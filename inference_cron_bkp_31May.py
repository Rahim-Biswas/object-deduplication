# """
# PieflyVisionX — AI/ML Inference Cron Job (Deduplication Pipeline version)

# Polls pvx_file for uploaded mobile_images, runs YOLO detection,
# fuses and deduplicates asset detections, and writes results back to the portal database.

# Flow per inspection:
#   1. Find one inspection that has 'uploaded' mobile_images in pvx_file.
#   2. Lock all 'uploaded' files for that inspection.
#   3. Mark status = 'processing'.
#   4. Process all files using the global deduplication pipeline:
#      - Download files
#      - Parse DJI pose
#      - YOLO detect (asset model)
#      - ReID embed & back-project to 3D
#      - Fuse & Deduplicate globally across the inspection
#   5. Write per-image rows to the `detection` table (for UI bounding-box display).
#   6. Upload ONLY unique-deduplicated annotated images to S3.
#   7. Update inspection.asset_counts JSONB with final per-component totals.
#   8. Mark processed.
#   9. On failure → mark status = 'failed'

# Environment variables (loaded from .env):
#   ...
# """

# import asyncio
# import json
# import logging
# import os
# import threading
# import urllib.parse
# import uuid
# import tempfile
# import yaml
# import gc
# from typing import Any, Dict, List, Optional
# from pathlib import Path

# import asyncpg
# import boto3
# import cv2
# import numpy as np
# from dotenv import load_dotenv

# # Pipeline imports
# from src.metadata import parse_dji_metadata
# from src.detector import TowerDetector
# from src.embedder import ReIDEmbedder
# from src.projector import Projector3D
# from src.fusion import DetectionFuser
# from src.deduplicator import Deduplicator
# from ultralytics import YOLO

# load_dotenv()

# # ---------------------------------------------------------------------------
# # Logging
# # ---------------------------------------------------------------------------

# logging.basicConfig(
#     level=logging.INFO,
#     format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
#     handlers=[
#         logging.StreamHandler(),
#         logging.FileHandler("inference_cron.log"),
#     ],
# )
# logger = logging.getLogger("visionx-cron")

# # ---------------------------------------------------------------------------
# # Config
# # ---------------------------------------------------------------------------

# ASSET_MODEL_PATH  = os.getenv("ASSET_MODEL_PATH",  "models/tower_asset_detection_v1.pt")
# DEFECT_MODEL_PATH = os.getenv("DEFECT_MODEL_PATH", "models/tower_defect_detection_v1.pt")
# INFERENCE_CONF    = float(os.getenv("INFERENCE_CONF", "0.8"))

# POLL_INTERVAL    = int(os.getenv("CRON_POLL_INTERVAL",  "60"))
# # Note: BATCH_SIZE is no longer used for fetching files, we process by inspection.
# MAX_CONCURRENT   = int(os.getenv("CRON_MAX_CONCURRENT",  "1")) # Number of concurrent inspections

# # ---------------------------------------------------------------------------
# # Pipeline Module Registry (thread-safe lazy load)
# # ---------------------------------------------------------------------------

# _modules: Dict[str, Any] = {
#     "detector": None,
#     "embedder": None,
#     "projector": None,
#     "fuser": None,
#     "deduper": None,
#     "defect_model": None
# }
# _module_lock = threading.Lock()


# def _get_device() -> str:
#     try:
#         import torch
#         return "cuda" if torch.cuda.is_available() else "cpu"
#     except Exception:
#         return "cpu"


# def _init_pipeline():
#     with _module_lock:
#         if _modules["detector"] is None:
#             logger.info("Initializing pipeline modules...")
#             cfg_path = "config.yaml"
#             if not os.path.exists(cfg_path):
#                 raise FileNotFoundError(f"{cfg_path} not found. Cannot initialize pipeline.")
            
#             with open(cfg_path, 'r') as f:
#                 cfg = yaml.safe_load(f)

#             # Assets use pipeline config
#             model_path = cfg["yolo"].get("model_path", ASSET_MODEL_PATH)
#             conf = cfg["yolo"].get("conf_threshold", INFERENCE_CONF)
#             iou = cfg["yolo"].get("iou_threshold", 0.45)
#             max_img_size = cfg.get("max_image_size", 1280)

#             logger.info(f"Loaded ASSET model:  {model_path} | conf={conf} | iou={iou}")
#             _modules["detector"] = TowerDetector(model_path, conf, iou, max_image_size=max_img_size)
            
#             # Defect uses env path and same default thresholds
#             defect_model_path = DEFECT_MODEL_PATH
#             defect_conf = conf
#             defect_iou = iou
#             logger.info(f"Loaded DEFECT model: {defect_model_path} | conf={defect_conf} | iou={defect_iou}")
#             _modules["defect_model"] = TowerDetector(defect_model_path, defect_conf, defect_iou, max_image_size=max_img_size)

#             _modules["embedder"] = ReIDEmbedder()
#             _modules["projector"] = Projector3D(cfg["camera"], cfg["component_sizes"])
#             _modules["fuser"] = DetectionFuser(_modules["embedder"], _modules["projector"])
#             _modules["deduper"] = Deduplicator(cfg.get("dedup", {}))
            
#     return _modules


# # ---------------------------------------------------------------------------
# # Database
# # ---------------------------------------------------------------------------

# def _build_dsn() -> str:
#     host = os.getenv("DB_HOST", "localhost")
#     port = os.getenv("DB_PORT", "5432")
#     db   = os.getenv("DB_NAME",     "visionx_qat").strip("'\"")
#     user = os.getenv("DB_USERNAME", "postgres").strip("'\"")
#     pwd  = os.getenv("DB_PASSWORD", "postgres").strip("'\"")
#     return (
#         f"postgresql://{urllib.parse.quote_plus(user)}:"
#         f"{urllib.parse.quote_plus(pwd)}@{host}:{port}/{db}"
#     )


# ENSURE_DETECTION_TABLE_SQL = """
# CREATE TABLE IF NOT EXISTS detection (
#     id             UUID         PRIMARY KEY DEFAULT gen_random_uuid(),
#     inspection_id  UUID         NOT NULL,
#     s3_url         VARCHAR(500) NOT NULL,
#     detection_type VARCHAR(50)  NOT NULL,
#     component_name VARCHAR(255),
#     detected_count INTEGER      DEFAULT 0,
#     created_by     VARCHAR(255) DEFAULT 'ai_ml_system',
#     created_date   TIMESTAMP    DEFAULT CURRENT_TIMESTAMP,
#     is_active      BOOLEAN      DEFAULT true,
#     is_deleted     BOOLEAN      DEFAULT false
# );
# """

# # Atomically claim all 'uploaded' mobile_images for ONE inspection.
# # pvx_file links to inspection via the pvx_inspection_file join table.
# CLAIM_INSPECTION_SQL = """
# WITH target_inspection AS (
#     SELECT insp_sub.id::text AS inspection_id
#     FROM pvx_file f_sub
#     JOIN pvx_inspection_file inf_sub ON inf_sub.file_id = f_sub.id
#     JOIN pvx_inspection insp_sub ON insp_sub.id::text = inf_sub.inspection_id::text
#     WHERE f_sub.status ILIKE 'uploaded'
#       AND (f_sub.is_deleted = false OR f_sub.is_deleted IS NULL)
#       AND (f_sub.s3_url ILIKE '%.jpg' OR f_sub.s3_url ILIKE '%.jpeg')
#     LIMIT 1
# )
# SELECT
#     f.id,
#     f.s3_url,
#     insp.id       AS inspection_id,
#     insp.tower_id
# FROM pvx_file f
# JOIN pvx_inspection_file inf ON inf.file_id = f.id
# JOIN pvx_inspection insp ON insp.id::text = inf.inspection_id::text
# WHERE f.status ILIKE 'uploaded'
#   AND (f.is_deleted = false OR f.is_deleted IS NULL)
#   AND (f.s3_url ILIKE '%.jpg' OR f.s3_url ILIKE '%.jpeg')
#   AND insp.id::text = (SELECT inspection_id FROM target_inspection)
# FOR UPDATE OF f SKIP LOCKED;
# """



# # ---------------------------------------------------------------------------
# # S3
# # ---------------------------------------------------------------------------

# def _build_s3_client():
#     return boto3.client(
#         "s3",
#         aws_access_key_id     = os.getenv("AWS_ACCESS_KEY_ID",     "").strip("'\""),
#         aws_secret_access_key = os.getenv("AWS_SECRET_ACCESS_KEY", "").strip("'\""),
#         region_name           = os.getenv("AWS_REGION", "ap-south-1"),
#     )


# def _s3_download_to_file(s3_client: Any, bucket: str, key: str, filepath: str) -> None:
#     s3_client.download_file(bucket, key, filepath)


# def _s3_upload(s3_client: Any, bucket: str, img: np.ndarray, key: str) -> None:
#     ok, encoded = cv2.imencode(".jpg", img)
#     if not ok:
#         raise ValueError("JPEG encoding failed")
#     s3_client.put_object(
#         Bucket=bucket, Key=key,
#         Body=encoded.tobytes(), ContentType="image/jpeg",
#     )


# # ---------------------------------------------------------------------------
# # Image annotation
# # ---------------------------------------------------------------------------

# def _annotate(img: np.ndarray, detections: List[Dict]) -> np.ndarray:
#     for d in detections:
#         x1, y1, x2, y2 = map(int, d["bbox"])
#         cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255), 3)
#         label = f"{d['class_name']} {d['confidence']:.2f}"
#         cv2.putText(img, label, (x1, y1 - 10),
#                     cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
#     return img


# # ---------------------------------------------------------------------------
# # Output S3 key
# # ---------------------------------------------------------------------------

# def _output_key(s3_url: str, tower_id: str, inspection_id: str, task_type: str) -> str:
#     folder   = os.getenv("FOLDER_NAME", "deep_learning").strip("'\"")
#     filename = s3_url.split("/")[-1]
#     base, ext = os.path.splitext(filename)
#     ext    = ext or ".jpg"
#     prefix = "defect_" if task_type == "defect" else "asset_"
#     clean  = base[len(prefix):] if base.startswith(prefix) else base
#     return f"{folder}/{tower_id}/{inspection_id}/detection/{prefix}{clean}{ext}"


# # ---------------------------------------------------------------------------
# # DB write helpers
# # ---------------------------------------------------------------------------

# async def _mark_processing(conn: asyncpg.Connection, file_id: uuid.UUID) -> None:
#     try:
#         await conn.execute(
#             """UPDATE pvx_file
#                SET status = 'processing', updated_date = NOW()
#                WHERE id = $1""",
#             file_id,
#         )
#     except asyncpg.exceptions.UndefinedColumnError:
#         await conn.execute(
#             "UPDATE pvx_file SET status = 'processing' WHERE id = $1", file_id
#         )


# async def _mark_processed(conn: asyncpg.Connection, file_id: uuid.UUID) -> None:
#     try:
#         await conn.execute(
#             """UPDATE pvx_file
#                SET status = 'processed', detected = true,
#                    updated_date = NOW(), updated_by = 'ai_ml_system'
#                WHERE id = $1""",
#             file_id,
#         )
#     except asyncpg.exceptions.UndefinedColumnError:
#         await conn.execute(
#             "UPDATE pvx_file SET status = 'processed', detected = true WHERE id = $1",
#             file_id,
#         )


# async def _mark_failed(conn: asyncpg.Connection, file_id: uuid.UUID) -> None:
#     try:
#         await conn.execute(
#             "UPDATE pvx_file SET status = 'failed', updated_date = NOW() WHERE id = $1",
#             file_id,
#         )
#     except asyncpg.exceptions.UndefinedColumnError:
#         await conn.execute(
#             "UPDATE pvx_file SET status = 'failed' WHERE id = $1", file_id
#         )


# async def _upsert_component(conn: asyncpg.Connection, class_name: str) -> uuid.UUID:
#     """Upsert a component into pvx_ground_base_component_master and return its UUID."""
#     name = class_name.title()
#     code = class_name.upper().replace(" ", "_")
#     row  = await conn.fetchrow(
#         """INSERT INTO pvx_ground_base_component_master
#              (id, component_name, component_code, created_by, created_date, is_active, is_deleted)
#            VALUES (gen_random_uuid(), $1, $2, 'ai_ml_system', NOW(), true, false)
#            ON CONFLICT (component_code)
#            DO UPDATE SET component_name = EXCLUDED.component_name
#            RETURNING id""",
#         name, code,
#     )
#     return row["id"]  # type: ignore[index]


# async def _insert_detection(
#     conn: asyncpg.Connection,
#     inspection_id: str,
#     s3_url: str,
#     detection_type: str,
#     component_name: str,
#     detected_count: int,
# ) -> None:
#     """Insert one row into the `detection` table for a single image + component."""
#     await conn.execute(
#         """INSERT INTO detection
#              (inspection_id, s3_url, detection_type,
#               component_name, detected_count,
#               created_by, created_date, is_active, is_deleted)
#            VALUES ($1::uuid, $2, $3, $4, $5,
#                    'ai_ml_system', NOW(), true, false)""",
#         inspection_id,
#         s3_url,
#         detection_type,
#         component_name.title(),
#         detected_count,
#     )


# async def _update_inspection_asset_counts(
#     conn: asyncpg.Connection,
#     inspection_id: str,
#     asset_counts: Dict[str, int],
# ) -> None:
#     """Update the pvx_inspection.asset_counts JSONB column with final deduplicated totals.
#     Format: {"<component_uuid>": <total_count>}
#     """
#     await conn.execute(
#         """UPDATE pvx_inspection
#                SET asset_counts = $1::jsonb
#              WHERE id = $2::uuid""",
#         json.dumps(asset_counts),
#         inspection_id,
#     )


# # ---------------------------------------------------------------------------
# # Core: process one inspection end-to-end
# # ---------------------------------------------------------------------------

# async def _process_inspection(
#     pool: asyncpg.Pool,
#     s3_client: Any,
#     rows: List[asyncpg.Record],
# ) -> None:
#     """
#     Process all mobile_images for one inspection end-to-end.

#     Asset flow (with global 3-D deduplication):
#       - Download → YOLO detect → fuse+embed → collect all detections
#       - Deduplicate globally across the inspection
#       - Upload ONLY the annotated images for unique (deduplicated) detections to S3
#       - Insert one `detection` row per image per component class
#       - Update inspection.asset_counts JSONB with final totals

#     Defect flow (per-image, no deduplication):
#       - Download → YOLO detect → annotate → upload to S3
#       - Insert one `detection` row per image per class
#     """
#     if not rows:
#         return

#     inspection_id = str(rows[0]["inspection_id"])
#     tower_id      = str(rows[0]["tower_id"])
#     bucket        = os.getenv("S3_BUCKET_NAME", "").strip("'\"")

#     logger.info(f"[{inspection_id}] START processing {len(rows)} file(s)")

#     # Mark all files as processing
#     async with pool.acquire() as conn:
#         for row in rows:
#             await _mark_processing(conn, row["id"])

#     try:
#         # Initialize pipeline modules (lazy load in thread)
#         modules = await asyncio.to_thread(_init_pipeline)

#         with tempfile.TemporaryDirectory() as tmpdir:

#             logger.info(f"[{inspection_id}] Processing {len(rows)} file(s) concurrently (assets + defects)")
#             detector = modules["detector"]
#             defect_model = modules["defect_model"]
#             fuser    = modules["fuser"]
#             deduper  = modules["deduper"]

#             # file_id → {s3_url, local_path, raw_dets}
#             file_info: Dict[str, Dict] = {}
#             all_fused_detections = []
            
#             img_semaphore = asyncio.Semaphore(5)

#             async def _process_image(idx: int, row: asyncpg.Record):
#                 async with img_semaphore:
#                     file_id    = str(row["id"])
#                     s3_url     = row["s3_url"]
#                     filename   = s3_url.split("/")[-1]
#                     local_path = os.path.join(tmpdir, f"{file_id}.jpg")

#                     logger.info(f"[{inspection_id}] [{idx}/{len(rows)}] Downloading: {filename}")

#                     try:
#                         await asyncio.to_thread(_s3_download_to_file, s3_client, bucket, s3_url, local_path)
                        
#                         pose_task = asyncio.to_thread(parse_dji_metadata, local_path)
#                         asset_task = asyncio.to_thread(detector.detect, local_path)
#                         defect_task = asyncio.to_thread(defect_model.detect, local_path)
                        
#                         pose, asset_dets, defect_dets = await asyncio.gather(pose_task, asset_task, defect_task)

#                         # --- DEFECT PIPELINE (No Deduplication) ---
#                         if defect_dets:
#                             defect_counts = {}
#                             for d in defect_dets:
#                                 defect_counts[d.class_name] = defect_counts.get(d.class_name, 0) + 1
#                             breakdown = ", ".join(f"{cls}={cnt}" for cls, cnt in defect_counts.items())
#                             logger.info(f"[{inspection_id}] [{idx}/{len(rows)}] {filename}: {len(defect_dets)} DEFECT(s) [{breakdown}]")
                            
#                             img = await asyncio.to_thread(cv2.imread, local_path)
#                             dict_dets = [{"class_name": d.class_name, "confidence": d.confidence, "bbox": d.bbox.tolist()} for d in defect_dets]
#                             annotated = await asyncio.to_thread(_annotate, img, dict_dets)
#                             defect_output_key = _output_key(s3_url, tower_id, inspection_id, "defect")
#                             await asyncio.to_thread(_s3_upload, s3_client, bucket, annotated, defect_output_key)
                            
#                             # Write defect detection rows
#                             async with pool.acquire() as conn:
#                                 async with conn.transaction():
#                                     for cls, cnt in defect_counts.items():
#                                         await _insert_detection(conn, inspection_id, defect_output_key, "defect", cls, cnt)

#                         # --- ASSET PIPELINE (With Deduplication) ---
#                         if asset_dets:
#                             asset_counts = {}
#                             for d in asset_dets:
#                                 asset_counts[d.class_name] = asset_counts.get(d.class_name, 0) + 1
#                             breakdown = ", ".join(f"{cls}={cnt}" for cls, cnt in asset_counts.items())
#                             logger.info(f"[{inspection_id}] [{idx}/{len(rows)}] {filename}: {len(asset_dets)} ASSET(s) [{breakdown}]")
                            
#                             fused_dets = await asyncio.to_thread(fuser.fuse, asset_dets, pose)
#                             for d in fused_dets:
#                                 d.frame_id = file_id
                                
#                             return file_id, s3_url, local_path, asset_dets, fused_dets
#                         else:
#                             logger.info(f"[{inspection_id}] [{idx}/{len(rows)}] {filename}: no ASSETS")
#                             return file_id, s3_url, local_path, [], []

#                     except Exception as e:
#                         logger.error(f"[{inspection_id}] [{idx}/{len(rows)}] Error processing {filename}: {e}")
#                         return None
#                     finally:
#                         gc.collect()

#             # Run all images concurrently (bounded by semaphore)
#             tasks = [_process_image(idx, row) for idx, row in enumerate(rows, 1)]
#             results = await asyncio.gather(*tasks)

#             for res in results:
#                 if res:
#                     file_id, s3_url, local_path, raw_dets, fused_dets = res
#                     file_info[file_id] = {
#                         "s3_url": s3_url,
#                         "local_path": local_path,
#                         "raw_dets": raw_dets,
#                     }
#                     all_fused_detections.extend(fused_dets)

#             # ------------------------------------------------------------------
#             # Global deduplication
#             # ------------------------------------------------------------------
#             logger.info(
#                 f"[{inspection_id}] Deduplicating {len(all_fused_detections)} total asset detections"
#             )
#             inventory = await asyncio.to_thread(deduper.deduplicate, all_fused_detections)

#             # Collect the set of file_ids that have at least one unique detection
#             unique_file_ids: set = set()
#             for unique_dets in inventory.values():
#                 for d in unique_dets:
#                     unique_file_ids.add(d.frame_id)

#             logger.info(
#                 f"[{inspection_id}] {len(unique_file_ids)} unique images after deduplication"
#             )

#             # ------------------------------------------------------------------
#             # Upload ONLY unique images and build per-image detection inserts
#             # ------------------------------------------------------------------
#             # file_id → output_s3_key (for unique images only)
#             unique_output_keys: Dict[str, str] = {}

#             for file_id in unique_file_ids:
#                 info       = file_info[file_id]
#                 raw_dets   = info["raw_dets"]
#                 local_path = info["local_path"]
#                 s3_url     = info["s3_url"]

#                 try:
#                     img = await asyncio.to_thread(cv2.imread, local_path)
#                     dict_dets = [
#                         {
#                             "class_name": d.class_name,
#                             "confidence": d.confidence,
#                             "bbox":       d.bbox.tolist(),
#                         }
#                         for d in raw_dets
#                     ]
#                     annotated  = await asyncio.to_thread(_annotate, img, dict_dets)
#                     output_key = _output_key(s3_url, tower_id, inspection_id, "asset")
#                     await asyncio.to_thread(_s3_upload, s3_client, bucket, annotated, output_key)
#                     unique_output_keys[file_id] = output_key
#                     logger.info(
#                         f"[{inspection_id}] Uploaded unique image → s3://{bucket}/{output_key}"
#                     )
#                 except Exception as e:
#                     logger.error(f"[{inspection_id}] Error uploading unique image [{file_id}]: {e}")

#             # ------------------------------------------------------------------
#             # Write DB records: `detection` rows + inspection.asset_counts
#             # ------------------------------------------------------------------
#             async with pool.acquire() as conn:
#                 async with conn.transaction():
#                     # Track {component_uuid: total_unique_count} for asset_counts
#                     asset_counts: Dict[str, int] = {}

#                     for class_name, unique_dets in inventory.items():
#                         comp_id = await _upsert_component(conn, class_name)
#                         comp_id_str = str(comp_id)

#                         # Accumulate total unique count for this component
#                         asset_counts[comp_id_str] = len(unique_dets)

#                         # Insert one detection row per unique image for this component
#                         # Group unique detections by frame_id to get per-image counts
#                         frame_class_counts: Dict[str, int] = {}
#                         for d in unique_dets:
#                             frame_class_counts[d.frame_id] = (
#                                 frame_class_counts.get(d.frame_id, 0) + 1
#                             )

#                         db_rows_written = 0
#                         for frame_id, img_count in frame_class_counts.items():
#                             output_key = unique_output_keys.get(frame_id, "")
#                             if not output_key:
#                                 continue  # image upload failed; skip
#                             await _insert_detection(
#                                 conn,
#                                 inspection_id,
#                                 output_key,
#                                 "asset",
#                                 class_name,
#                                 img_count,
#                             )
#                             db_rows_written += 1

#                         logger.info(
#                             f"[{inspection_id}] DB: '{class_name}' — "
#                             f"{len(unique_dets)} unique object(s), "
#                             f"{db_rows_written} detection row(s) inserted"
#                         )

#                     # Update the inspection.asset_counts JSONB column
#                     if asset_counts:
#                         await _update_inspection_asset_counts(
#                             conn, inspection_id, asset_counts
#                         )
#                         lines = "\n".join(
#                             f"  {cn:30s}: {len(ud):>4d} unique"
#                             for cn, ud in inventory.items()
#                         )
#                         total_unique = sum(len(ud) for ud in inventory.values())
#                         logger.info(
#                             f"[{inspection_id}] ===== FINAL ASSET COUNTS =====\n"
#                             + lines
#                             + f"\n  TOTAL                         : {total_unique:>4d} unique"
#                         )

#         # Mark all files as processed
#         async with pool.acquire() as conn:
#             for row in rows:
#                 await _mark_processed(conn, row["id"])

#         logger.info(f"[{inspection_id}] DONE")

#     except Exception as exc:
#         logger.error(f"[{inspection_id}] FAILED: {exc}", exc_info=True)
#         async with pool.acquire() as conn:
#             for row in rows:
#                 await _mark_failed(conn, row["id"])


# # ---------------------------------------------------------------------------
# # Cron tick: claim batch → process concurrently
# # ---------------------------------------------------------------------------

# async def _tick(
#     pool: asyncpg.Pool,
#     s3_client: Any,
#     semaphore: asyncio.Semaphore,
# ) -> None:
#     """Claim one inspection's 'uploaded' mobile_images and dispatch processing."""
#     # Claim the inspection batch atomically
#     async with pool.acquire() as conn:
#         async with conn.transaction():
#             rows = await conn.fetch(CLAIM_INSPECTION_SQL)
#             if rows:
#                 # Pre-mark as 'processing' inside the transaction so the lock
#                 # is released quickly while the heavy processing runs outside.
#                 ids = [r["id"] for r in rows]
#                 await conn.executemany(
#                     "UPDATE pvx_file SET status = 'processing' WHERE id = $1",
#                     [(fid,) for fid in ids],
#                 )

#     if not rows:
#         logger.info("[poll] No uploaded .jpg images found in pvx_file — sleeping for %ds.", POLL_INTERVAL)
#         return

#     logger.info(f"Claimed {len(rows)} file(s) for inspection {rows[0]['inspection_id']} for processing.")

#     async def _guarded(insp_rows: List[asyncpg.Record]) -> None:
#         async with semaphore:
#             await _process_inspection(pool, s3_client, insp_rows)

#     await asyncio.gather(_guarded(rows))


# # ---------------------------------------------------------------------------
# # Manual S3 Path Processing (No DB)
# # ---------------------------------------------------------------------------

# async def _process_manual_s3_path(s3_client: Any, bucket: str, prefix: str) -> None:
#     """
#     Process images from a manual S3 prefix (no DB).

#     Outputs are written to:
#         {FOLDER_NAME}/{input_folder_name}/detection/{filename}
#     where input_folder_name is the last segment of the input prefix.

#     Only UNIQUE (deduplicated) images are annotated and uploaded.
#     """
#     logger.info(f"Listing objects in s3://{bucket}/{prefix}")

#     # Ensure prefix ends with /
#     if not prefix.endswith("/"):
#         prefix += "/"

#     # Derive input folder name from the last non-empty segment of the prefix
#     # e.g. "towers/site_abc/images/" -> "images"
#     folder_name   = os.getenv("FOLDER_NAME", "deep_learning").strip("'\"")
#     input_folder  = [p for p in prefix.rstrip("/").split("/") if p][-1]
#     output_prefix = f"{folder_name}/{input_folder}/detection/"

#     response = await asyncio.to_thread(s3_client.list_objects_v2, Bucket=bucket, Prefix=prefix)

#     if "Contents" not in response:
#         logger.warning("No files found in the specified S3 path.")
#         return

#     s3_keys = [obj["Key"] for obj in response["Contents"] if obj["Key"].lower().endswith(".jpg")]

#     if not s3_keys:
#         logger.warning("No JPG images found in the specified S3 path.")
#         return

#     logger.info(f"Found {len(s3_keys)} images. Output will go to s3://{bucket}/{output_prefix}")

#     modules  = await asyncio.to_thread(_init_pipeline)
#     detector = modules["detector"]
#     fuser    = modules["fuser"]
#     deduper  = modules["deduper"]

#     # filename -> {local_path, raw_dets} -- store everything; upload ONLY unique ones later
#     file_info: Dict[str, Dict] = {}
#     all_fused_detections = []

#     with tempfile.TemporaryDirectory() as tmpdir:
#         # ------------------------------------------------------------------
#         # Pass 1: Concurrent download, detect, fuse
#         # ------------------------------------------------------------------
#         defect_model = modules["defect_model"]
#         img_semaphore = asyncio.Semaphore(5)
        
#         async def _process_manual_image(s3_url: str):
#             async with img_semaphore:
#                 filename   = s3_url.split("/")[-1]
#                 local_path = os.path.join(tmpdir, filename)
                
#                 logger.info(f"Downloading {s3_url}")
#                 try:
#                     await asyncio.to_thread(_s3_download_to_file, s3_client, bucket, s3_url, local_path)

#                     pose_task = asyncio.to_thread(parse_dji_metadata, local_path)
#                     asset_task = asyncio.to_thread(detector.detect, local_path)
#                     defect_task = asyncio.to_thread(defect_model.detect, local_path)
                    
#                     pose, asset_dets, defect_dets = await asyncio.gather(pose_task, asset_task, defect_task)

#                     # Handle DEFECTS
#                     if defect_dets:
#                         defect_counts = {}
#                         for d in defect_dets:
#                             defect_counts[d.class_name] = defect_counts.get(d.class_name, 0) + 1
#                         breakdown = ", ".join(f"{cls}={cnt}" for cls, cnt in defect_counts.items())
#                         logger.info(f"{filename}: {len(defect_dets)} DEFECT(s) [{breakdown}]")
                        
#                         img = await asyncio.to_thread(cv2.imread, local_path)
#                         dict_dets = [{"class_name": d.class_name, "confidence": d.confidence, "bbox": d.bbox.tolist()} for d in defect_dets]
#                         annotated = await asyncio.to_thread(_annotate, img, dict_dets)
#                         defect_output_key = f"{output_prefix}defect_{filename}"
#                         await asyncio.to_thread(_s3_upload, s3_client, bucket, annotated, defect_output_key)
                        
#                     # Handle ASSETS
#                     if asset_dets:
#                         asset_counts = {}
#                         for d in asset_dets:
#                             asset_counts[d.class_name] = asset_counts.get(d.class_name, 0) + 1
#                         breakdown = ", ".join(f"{cls}={cnt}" for cls, cnt in asset_counts.items())
#                         logger.info(f"{filename}: {len(asset_dets)} ASSET(s) [{breakdown}]")
                        
#                         fused_dets = await asyncio.to_thread(fuser.fuse, asset_dets, pose)
#                         for d in fused_dets:
#                             d.frame_id = filename
                            
#                         return filename, local_path, asset_dets, fused_dets
#                     else:
#                         logger.info(f"{filename}: no ASSETS")
#                         return filename, local_path, [], []

#                 except Exception as e:
#                     logger.error(f"Error processing {s3_url}: {e}")
#                     return None
#                 finally:
#                     gc.collect()

#         tasks = [_process_manual_image(s3_url) for s3_url in s3_keys]
#         results = await asyncio.gather(*tasks)
        
#         for res in results:
#             if res:
#                 filename, local_path, raw_dets, fused_dets = res
#                 file_info[filename] = {"local_path": local_path, "raw_dets": raw_dets}
#                 all_fused_detections.extend(fused_dets)

#         # ------------------------------------------------------------------
#         # Pass 2: global deduplication (Assets)
#         # ------------------------------------------------------------------
#         logger.info(f"Deduplicating {len(all_fused_detections)} total asset detections")
#         inventory = await asyncio.to_thread(deduper.deduplicate, all_fused_detections)

#         # Collect filenames that appear in any unique detection
#         unique_filenames: set = set()
#         for unique_dets in inventory.values():
#             for d in unique_dets:
#                 unique_filenames.add(d.frame_id)

#         total        = len(s3_keys)
#         unique_count = len(unique_filenames)
#         logger.info(f"{unique_count} unique images after deduplication (skipped {total - unique_count})")

#         # ------------------------------------------------------------------
#         # Pass 3: annotate and upload ONLY unique images
#         # ------------------------------------------------------------------
#         for filename in unique_filenames:
#             info       = file_info[filename]
#             local_path = info["local_path"]
#             raw_dets   = info["raw_dets"]
#             output_key = f"{output_prefix}{filename}"
#             try:
#                 img = await asyncio.to_thread(cv2.imread, local_path)
#                 dict_dets = [
#                     {"class_name": d.class_name, "confidence": d.confidence, "bbox": d.bbox.tolist()}
#                     for d in raw_dets
#                 ]
#                 annotated = await asyncio.to_thread(_annotate, img, dict_dets)
#                 logger.info(f"Uploading unique annotated image -> s3://{bucket}/{output_key}")
#                 await asyncio.to_thread(_s3_upload, s3_client, bucket, annotated, output_key)
#             except Exception as e:
#                 logger.error(f"Error uploading unique image {filename}: {e}")

#         # ------------------------------------------------------------------
#         # Upload inventory JSON to same output folder
#         # ------------------------------------------------------------------
#         report = {}
#         for cls_name, unique_dets in inventory.items():
#             report[cls_name] = {
#                 "count": len(unique_dets),
#                 "components": [
#                     {
#                         "best_frame":  d.frame_id,
#                         "confidence":  round(d.confidence, 3),
#                         "world_xyz_m": d.world_xyz.tolist() if d.world_xyz is not None else [0, 0, 0],
#                         "bbox_pixels": d.bbox.tolist(),
#                     }
#                     for d in unique_dets
#                 ],
#             }

#         report_key = f"{output_prefix}inventory.json"
#         logger.info(f"Uploading deduplication inventory -> s3://{bucket}/{report_key}")
#         await asyncio.to_thread(
#             s3_client.put_object,
#             Bucket=bucket,
#             Key=report_key,
#             Body=json.dumps(report, indent=2),
#             ContentType="application/json",
#         )

#         unique_counts = {k: len(v) for k, v in inventory.items()}
#         logger.info(f"Manual processing complete. Unique components: {unique_counts}")


# # ---------------------------------------------------------------------------
# # Entry point
# # ---------------------------------------------------------------------------

# async def main() -> None:
#     s3_client = _build_s3_client()
    
#     manual_path_mode = os.getenv("MANUAL_PATH_MODE", "false").lower() == "true"
#     manual_s3_path = os.getenv("MANUAL_S3_PATH", "").strip("'\"")
    
#     if manual_path_mode and manual_s3_path:
#         bucket = os.getenv("S3_BUCKET_NAME", "").strip("'\"")
#         logger.info(f"Running in MANUAL PATH MODE for S3 prefix: {manual_s3_path}")
#         await _process_manual_s3_path(s3_client, bucket, manual_s3_path)
#         return

#     dsn = _build_dsn()
#     logger.info("Connecting to PostgreSQL…")
#     pool = await asyncpg.create_pool(dsn=dsn, min_size=2, max_size=10)

#     async with pool.acquire() as conn:
#         await conn.execute(ENSURE_DETECTION_TABLE_SQL)
#         logger.info("`detection` table verified.")

#     semaphore = asyncio.Semaphore(MAX_CONCURRENT)

#     logger.info(
#         f"PieflyVisionX inference cron running | "
#         f"poll={POLL_INTERVAL}s  concurrency={MAX_CONCURRENT}"
#     )

#     try:
#         while True:
#             try:
#                 await _tick(pool, s3_client, semaphore)
#             except Exception as exc:
#                 logger.error(f"Tick-level error: {exc}", exc_info=True)
#             await asyncio.sleep(POLL_INTERVAL)
#     finally:
#         await pool.close()
#         logger.info("DB pool closed. Cron stopped.")


# if __name__ == "__main__":
#     asyncio.run(main())





"""
PieflyVisionX — AI/ML Inference Cron Job (Deduplication Pipeline version v2)

Polls pvx_file for uploaded drone_images, runs YOLO detection,
fuses and deduplicates asset detections, and writes results back to the portal database.

Flow per inspection:
  1. Find one inspection that has 'uploaded' drone_images in pvx_file.
  2. Lock all 'uploaded' files for that inspection.
  3. Mark status = 'processing'.
  4. Process all files using the global deduplication pipeline:
     - Download files
     - Parse DJI pose
     - YOLO detect (asset model)
     - ReID embed & back-project to 3D
     - Fuse & Deduplicate globally across the inspection
  5. Write per-image rows to the `detection` table (for UI bounding-box display).
  6. Upload ONLY unique-deduplicated annotated images to S3.
  7. Update inspection.asset_counts JSONB with final per-component totals.
  8. Mark processed.
  9. On failure → mark status = 'failed'

Environment variables (loaded from .env):
  ...
"""

import asyncio
import json
import logging
import os
import threading
import urllib.parse
import uuid
import tempfile
import yaml
import gc
from typing import Any, Dict, List, Optional
from pathlib import Path

import asyncpg
import boto3
import cv2
import numpy as np
from dotenv import load_dotenv

# Pipeline imports
from src.metadata import parse_dji_metadata
from src.detector import TowerDetector
from src.embedder import ReIDEmbedder
from src.projector import Projector3D
from src.fusion import DetectionFuser
from src.deduplicator import Deduplicator
from ultralytics import YOLO

load_dotenv()

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("inference_cron.log"),
    ],
)
logger = logging.getLogger("visionx-cron")

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

ASSET_MODEL_PATH  = os.getenv("ASSET_MODEL_PATH",  "models/tower_asset_detection_v1.pt")
DEFECT_MODEL_PATH = os.getenv("DEFECT_MODEL_PATH", "models/tower_defect_detection_v1.pt")
INFERENCE_CONF    = float(os.getenv("INFERENCE_CONF", "0.8"))

POLL_INTERVAL    = int(os.getenv("CRON_POLL_INTERVAL",  "60"))
MAX_CONCURRENT   = int(os.getenv("CRON_MAX_CONCURRENT",  "1"))

# ---------------------------------------------------------------------------
# Pipeline Module Registry (thread-safe lazy load)
# ---------------------------------------------------------------------------

_modules: Dict[str, Any] = {
    "detector": None,
    "embedder": None,
    "projector": None,
    "fuser": None,
    "deduper": None,
    "defect_model": None
}
_module_lock = threading.Lock()


def _get_device() -> str:
    try:
        import torch
        return "cuda" if torch.cuda.is_available() else "cpu"
    except Exception:
        return "cpu"


def _init_pipeline():
    with _module_lock:
        if _modules["detector"] is None:
            logger.info("Initializing pipeline modules...")
            cfg_path = "config.yaml"
            if not os.path.exists(cfg_path):
                raise FileNotFoundError(f"{cfg_path} not found. Cannot initialize pipeline.")

            with open(cfg_path, 'r') as f:
                cfg = yaml.safe_load(f)

            model_path = cfg["yolo"].get("model_path", ASSET_MODEL_PATH)
            conf = cfg["yolo"].get("conf_threshold", INFERENCE_CONF)
            iou = cfg["yolo"].get("iou_threshold", 0.45)
            max_img_size = cfg.get("max_image_size", 1280)

            logger.info(f"Loaded ASSET model:  {model_path} | conf={conf} | iou={iou}")
            _modules["detector"] = TowerDetector(model_path, conf, iou, max_image_size=max_img_size)

            defect_model_path = DEFECT_MODEL_PATH
            defect_conf = conf
            defect_iou = iou
            logger.info(f"Loaded DEFECT model: {defect_model_path} | conf={defect_conf} | iou={defect_iou}")
            _modules["defect_model"] = TowerDetector(defect_model_path, defect_conf, defect_iou, max_image_size=max_img_size)

            _modules["embedder"] = ReIDEmbedder()
            _modules["projector"] = Projector3D(cfg["camera"], cfg["component_sizes"])
            _modules["fuser"] = DetectionFuser(_modules["embedder"], _modules["projector"])
            _modules["deduper"] = Deduplicator(cfg.get("dedup", {}))

    return _modules


# ---------------------------------------------------------------------------
# Database
# ---------------------------------------------------------------------------

def _build_dsn() -> str:
    host = os.getenv("DB_HOST", "localhost")
    port = os.getenv("DB_PORT", "5432")
    db   = os.getenv("DB_NAME",     "visionx_qat").strip("'\"")
    user = os.getenv("DB_USERNAME", "postgres").strip("'\"")
    pwd  = os.getenv("DB_PASSWORD", "postgres").strip("'\"")
    return (
        f"postgresql://{urllib.parse.quote_plus(user)}:"
        f"{urllib.parse.quote_plus(pwd)}@{host}:{port}/{db}"
    )

ENSURE_DETECTION_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS pvx_detection (
    id             UUID         PRIMARY KEY DEFAULT gen_random_uuid(),
    inspection_id  UUID         NOT NULL,
    s3_url         VARCHAR(500) NOT NULL,
    meta_data      JSONB        NOT NULL,
    detection_type VARCHAR(50)  NOT NULL,
    component_name VARCHAR(255),
    detected_count INTEGER      DEFAULT 0,
    feedback       TEXT,
    correction_points TEXT,
    created_by     VARCHAR(255) DEFAULT 'ai_ml_system',
    created_date   TIMESTAMP    DEFAULT CURRENT_TIMESTAMP,
    is_active      BOOLEAN      DEFAULT true,
    is_deleted     BOOLEAN      DEFAULT false
);
"""


# FIX 1: Filter for drone images (file_type = 'image')
CLAIM_INSPECTION_SQL = """
WITH target_inspection AS (
    SELECT insp_sub.uuid::text AS inspection_uuid
    FROM pvx_file f_sub
    JOIN pvx_inspection_file inf_sub ON inf_sub.file_id = f_sub.id
    JOIN pvx_inspection insp_sub ON insp_sub.uuid::text = inf_sub.inspection_id::text
    WHERE f_sub.status ILIKE 'uploaded'
      AND f_sub.file_type = 'image'
      AND inf_sub.long_run_process_id IS NULL
      AND (f_sub.is_deleted = false OR f_sub.is_deleted IS NULL)
      AND (f_sub.s3_url ILIKE '%.jpg' OR f_sub.s3_url ILIKE '%.jpeg')
    LIMIT 1
)
SELECT
    f.id,
    f.s3_url,
    insp.uuid     AS inspection_uuid,
    insp.id       AS inspection_id,
    t.id          AS tower_id
FROM pvx_file f
JOIN pvx_inspection_file inf ON inf.file_id = f.id
JOIN pvx_inspection insp ON insp.uuid::text = inf.inspection_id::text
JOIN pvx_tower t ON t.uuid = insp.tower_id
WHERE f.status ILIKE 'uploaded'
  AND f.file_type = 'image'
  AND inf.long_run_process_id IS NULL
  AND (f.is_deleted = false OR f.is_deleted IS NULL)
  AND (f.s3_url ILIKE '%.jpg' OR f.s3_url ILIKE '%.jpeg')
  AND insp.uuid::text = (SELECT inspection_uuid FROM target_inspection)
FOR UPDATE OF f SKIP LOCKED;
"""


# ---------------------------------------------------------------------------
# S3
# ---------------------------------------------------------------------------

def _build_s3_client():
    return boto3.client(
        "s3",
        aws_access_key_id     = os.getenv("AWS_ACCESS_KEY_ID",     "").strip("'\""),
        aws_secret_access_key = os.getenv("AWS_SECRET_ACCESS_KEY", "").strip("'\""),
        region_name           = os.getenv("AWS_REGION", "ap-south-1"),
    )


def _s3_download_to_file(s3_client: Any, bucket: str, key: str, filepath: str) -> None:
    s3_client.download_file(bucket, key, filepath)


def _s3_upload(s3_client: Any, bucket: str, img: np.ndarray, key: str) -> None:
    ok, encoded = cv2.imencode(".jpg", img)
    if not ok:
        raise ValueError("JPEG encoding failed")
    s3_client.put_object(
        Bucket=bucket, Key=key,
        Body=encoded.tobytes(), ContentType="image/jpeg",
    )


# ---------------------------------------------------------------------------
# Image annotation
# ---------------------------------------------------------------------------

def _annotate(img: np.ndarray, detections: List[Dict]) -> np.ndarray:
    for d in detections:
        x1, y1, x2, y2 = map(int, d["bbox"])
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255), 3)
        label = f"{d['class_name']} {d['confidence']:.2f}"
        cv2.putText(img, label, (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
    return img


# ---------------------------------------------------------------------------
# Output S3 key
# ---------------------------------------------------------------------------

def _output_key(s3_url: str, tower_id: str, inspection_id: str, task_type: str) -> str:
    folder   = os.getenv("FOLDER_NAME", "deep_learning").strip("'\"")
    filename = s3_url.split("/")[-1]
    base, ext = os.path.splitext(filename)
    ext    = ext or ".jpg"
    prefix = "defect_" if task_type == "defect" else "asset_"
    clean  = base[len(prefix):] if base.startswith(prefix) else base
    return f"{folder}/{tower_id}/{inspection_id}/detection/{prefix}{clean}{ext}"


# ---------------------------------------------------------------------------
# DB write helpers
# ---------------------------------------------------------------------------

async def _mark_processing(conn: asyncpg.Connection, file_id: uuid.UUID) -> None:
    try:
        await conn.execute(
            """UPDATE pvx_file
               SET status = 'processing', updated_date = NOW()
               WHERE id = $1""",
            file_id,
        )
    except asyncpg.exceptions.UndefinedColumnError:
        await conn.execute(
            "UPDATE pvx_file SET status = 'processing' WHERE id = $1", file_id
        )


async def _mark_processed(conn: asyncpg.Connection, file_id: uuid.UUID) -> None:
    try:
        await conn.execute(
            """UPDATE pvx_file
               SET status = 'processed', detected = true,
                   updated_date = NOW(), updated_by = 'ai_ml_system'
               WHERE id = $1""",
            file_id,
        )
    except asyncpg.exceptions.UndefinedColumnError:
        await conn.execute(
            "UPDATE pvx_file SET status = 'processed', detected = true WHERE id = $1",
            file_id,
        )


async def _mark_failed(conn: asyncpg.Connection, file_id: uuid.UUID) -> None:
    try:
        await conn.execute(
            "UPDATE pvx_file SET status = 'failed', updated_date = NOW() WHERE id = $1",
            file_id,
        )
    except asyncpg.exceptions.UndefinedColumnError:
        await conn.execute(
            "UPDATE pvx_file SET status = 'failed' WHERE id = $1", file_id
        )


async def _upsert_component(
    conn: asyncpg.Connection, 
    class_name: str, 
    is_defect: bool = False,
    parent_id: Optional[uuid.UUID] = None
) -> uuid.UUID:
    """Upsert a component into pvx_ground_base_component_master and return its UUID.
    Robustly handles both "componentName"/"componentCode" (camelCase) and 
    component_name/component_code (snake_case) schema versions.
    """
    name = class_name.title()
    code = class_name.upper().replace(" ", "_")
    description = "AUTOMATIC_DEFECT_GROUP" if is_defect else None
    
    # Resolve parent code if needed
    if is_defect and parent_id:
        try:
            # Try camelCase first
            parent_row = await conn.fetchrow(
                'SELECT "componentCode" FROM pvx_ground_base_component_master WHERE id = $1',
                parent_id
            )
            if parent_row:
                code = f"{parent_row['componentCode']}_{code}"
        except asyncpg.exceptions.UndefinedColumnError:
            # Fallback to snake_case
            parent_row = await conn.fetchrow(
                'SELECT component_code FROM pvx_ground_base_component_master WHERE id = $1',
                parent_id
            )
            if parent_row:
                code = f"{parent_row['component_code']}_{code}"

    # Manual upsert: Try to find existing by code
    existing = None
    try:
        existing = await conn.fetchrow(
            'SELECT id FROM pvx_ground_base_component_master WHERE "componentCode" = $1',
            code
        )
    except asyncpg.exceptions.UndefinedColumnError:
        existing = await conn.fetchrow(
            'SELECT id FROM pvx_ground_base_component_master WHERE component_code = $1',
            code
        )

    if existing:
        comp_id = existing["id"]
        try:
            # Try camelCase Update
            await conn.execute(
                """UPDATE pvx_ground_base_component_master
                   SET "componentName" = $1,
                       description = COALESCE(description, $2),
                       parent_component_row_id = COALESCE(parent_component_row_id, $3),
                       updated_by = 'ai_ml_system',
                       updated_date = NOW()
                   WHERE id = $4""",
                name, description, parent_id, comp_id
            )
        except asyncpg.exceptions.UndefinedColumnError:
            # Fallback to snake_case Update
            await conn.execute(
                """UPDATE pvx_ground_base_component_master
                   SET component_name = $1,
                       description = COALESCE(description, $2),
                       parent_component_row_id = COALESCE(parent_component_row_id, $3),
                       updated_by = 'ai_ml_system',
                       updated_date = NOW()
                   WHERE id = $4""",
                name, description, parent_id, comp_id
            )
        return comp_id
    else:
        new_id = uuid.uuid4()
        try:
            # Try camelCase Insert
            await conn.execute(
                """INSERT INTO pvx_ground_base_component_master
                     (id, "componentName", "componentCode", description, 
                      parent_component_row_id, created_by, created_date, is_active, is_deleted)
                   VALUES ($1, $2, $3, $4, $5, 'ai_ml_system', NOW(), true, false)""",
                new_id, name, code, description, parent_id
            )
        except asyncpg.exceptions.UndefinedColumnError:
            # Fallback to snake_case Insert
            await conn.execute(
                """INSERT INTO pvx_ground_base_component_master
                     (id, component_name, component_code, description, 
                      parent_component_row_id, created_by, created_date, is_active, is_deleted)
                   VALUES ($1, $2, $3, $4, $5, 'ai_ml_system', NOW(), true, false)""",
                new_id, name, code, description, parent_id
            )
        return new_id


async def _insert_detection(
    conn: asyncpg.Connection,
    inspection_id: str,
    s3_url: str,
    detection_type: str,
    component_name: str,
    detected_count: int,
    meta_data: Dict = None
) -> None:
    """Insert one row into the `pvx_detection` table for a single image + component."""
    if meta_data is None:
        meta_data = {}
    await conn.execute(
        """INSERT INTO pvx_detection
             (inspection_id, s3_url, meta_data, detection_type,
              component_name, detected_count,
              created_by, created_date, is_active, is_deleted)
           VALUES ($1::uuid, $2, $3::jsonb, $4, $5, $6,
                   'ai_ml_system', NOW(), true, false)""",
        inspection_id,
        s3_url,
        json.dumps(meta_data),
        detection_type,
        component_name.title(),
        detected_count,
    )


async def _update_inspection_asset_counts(
    conn: asyncpg.Connection,
    inspection_id: str,
    asset_counts: Dict[str, int],
) -> None:
    """Update the pvx_inspection.asset_counts JSONB column with final deduplicated totals.
    Format: {"<component_uuid>": <total_count>}
    """
    await conn.execute(
        """UPDATE pvx_inspection
               SET asset_counts = $1::jsonb
             WHERE uuid = $2::uuid""",
        json.dumps(asset_counts),
        inspection_id,
    )


# ---------------------------------------------------------------------------
# Core: process one inspection end-to-end
# ---------------------------------------------------------------------------

async def _process_inspection(
    pool: asyncpg.Pool,
    s3_client: Any,
    rows: List[asyncpg.Record],
) -> None:
    """
    Process all drone_images for one inspection end-to-end.

    Asset flow (with global 3-D deduplication):
      - Download → YOLO detect → fuse+embed → collect all detections
      - Deduplicate globally across the inspection
      - Upload ONLY the annotated images for unique (deduplicated) detections to S3
      - Insert one `detection` row per image per component class
      - Update inspection.asset_counts JSONB with final totals

    Defect flow (per-image, no deduplication):
      - Download → YOLO detect → annotate → upload to S3
      - Insert one `detection` row per image per class
    """
    if not rows:
        return

    inspection_uuid = str(rows[0]["inspection_uuid"])
    inspection_id   = str(rows[0]["inspection_id"])
    tower_id        = str(rows[0]["tower_id"])
    bucket          = os.getenv("S3_BUCKET_NAME", "").strip("'\"")

    logger.info(f"[{inspection_id}] START processing {len(rows)} file(s)")

    async with pool.acquire() as conn:
        for row in rows:
            await _mark_processing(conn, row["id"])

    try:
        modules = await asyncio.to_thread(_init_pipeline)

        with tempfile.TemporaryDirectory() as tmpdir:

            logger.info(f"[{inspection_id}] Processing {len(rows)} file(s) concurrently (assets + defects)")
            detector = modules["detector"]
            defect_model = modules["defect_model"]
            fuser    = modules["fuser"]
            deduper  = modules["deduper"]

            file_info: Dict[str, Dict] = {}
            all_fused_detections = []

            # Batch processing: process 10 images at a time
            BATCH_SIZE = 10
            for i in range(0, len(rows), BATCH_SIZE):
                batch_rows = rows[i : i + BATCH_SIZE]
                logger.info(f"[{inspection_id}] Processing batch {i//BATCH_SIZE + 1}/{(len(rows)-1)//BATCH_SIZE + 1} ({len(batch_rows)} file(s))")
                
                async def _process_image(idx_in_batch: int, row: asyncpg.Record):
                    absolute_idx = i + idx_in_batch + 1
                    file_id    = str(row["id"])
                    s3_url     = row["s3_url"]
                    filename   = s3_url.split("/")[-1]
                    local_path = os.path.join(tmpdir, f"{file_id}.jpg")

                    logger.info(f"[{inspection_id}] [{absolute_idx}/{len(rows)}] Downloading: {filename}")

                    try:
                        await asyncio.to_thread(_s3_download_to_file, s3_client, bucket, s3_url, local_path)

                        pose_task   = asyncio.to_thread(parse_dji_metadata, local_path)
                        asset_task  = asyncio.to_thread(detector.detect, local_path)
                        defect_task = asyncio.to_thread(defect_model.detect, local_path)

                        pose, asset_dets, defect_dets = await asyncio.gather(pose_task, asset_task, defect_task)

                        # --- PREPARE DATA ---
                        res = {
                            "file_id": file_id,
                            "s3_url": s3_url,
                            "local_path": local_path,
                            "asset_dets": asset_dets,
                            "defect_dets": defect_dets,
                            "pose": pose,
                            "fused_dets": []
                        }

                        if asset_dets:
                            fused_dets = await asyncio.to_thread(fuser.fuse, asset_dets, pose)
                            for d in fused_dets:
                                d.frame_id = file_id
                            res["fused_dets"] = fused_dets

                        return res

                    except Exception as e:
                        logger.error(f"[{inspection_id}] [{absolute_idx}/{len(rows)}] Error processing {filename}: {e}")
                        return None
                    finally:
                        gc.collect()

                batch_tasks = [_process_image(idx, row) for idx, row in enumerate(batch_rows)]
                batch_results = await asyncio.gather(*batch_tasks)

                for res in batch_results:
                    if res:
                        f_id = res["file_id"]
                        file_info[f_id] = res
                        all_fused_detections.extend(res["fused_dets"])

            # ------------------------------------------------------------------
            # Global deduplication for Assets
            # ------------------------------------------------------------------
            logger.info(f"[{inspection_id}] Deduplicating {len(all_fused_detections)} total asset detections")
            asset_inventory = await asyncio.to_thread(deduper.deduplicate, all_fused_detections)

            # Map from frame_id -> primary asset class name (highest confidence raw detection)
            frame_to_primary_asset: Dict[str, str] = {}
            for f_id, info in file_info.items():
                if info["asset_dets"]:
                    primary = max(info["asset_dets"], key=lambda d: d.confidence)
                    frame_to_primary_asset[f_id] = primary.class_name

            # ------------------------------------------------------------------
            # Process and Upload
            # ------------------------------------------------------------------
            final_asset_counts: Dict[str, int] = {} # UUID -> count
            
            # 1. Assets
            unique_asset_frame_ids = set()
            for class_name, unique_dets in asset_inventory.items():
                for d in unique_dets:
                    unique_asset_frame_ids.add(d.frame_id)

            unique_asset_output_keys: Dict[str, str] = {}
            async with pool.acquire() as conn:
                async with conn.transaction():
                    for class_name, unique_dets in asset_inventory.items():
                        comp_id = await _upsert_component(conn, class_name, is_defect=False)
                        final_asset_counts[str(comp_id)] = len(unique_dets)

                        # Group by frame for detection rows
                        frame_counts = {}
                        for d in unique_dets:
                            frame_counts[d.frame_id] = frame_counts.get(d.frame_id, 0) + 1
                        
                        for f_id, count in frame_counts.items():
                            info = file_info[f_id]
                            if f_id not in unique_asset_output_keys:
                                img = await asyncio.to_thread(cv2.imread, info["local_path"])
                                dict_dets = [{"class_name": d.class_name, "confidence": d.confidence, "bbox": d.bbox.tolist()} for d in info["asset_dets"]]
                                annotated = await asyncio.to_thread(_annotate, img, dict_dets)
                                out_key = _output_key(info["s3_url"], tower_id, inspection_id, "asset")
                                await asyncio.to_thread(_s3_upload, s3_client, bucket, annotated, out_key)
                                unique_asset_output_keys[f_id] = out_key
                            
                            # Prepare metadata: all asset detections of this class in this frame
                            meta = {
                                "detections": [
                                    {"class_name": d.class_name, "confidence": float(d.confidence), "bbox": d.bbox.tolist()}
                                    for d in info["asset_dets"] if d.class_name == class_name
                                ]
                            }
                            await _insert_detection(conn, inspection_uuid, unique_asset_output_keys[f_id], "asset", class_name, count, meta)

            # 2. Defects
            defect_class_counts: Dict[str, int] = {} # class_name -> total detections
            async with pool.acquire() as conn:
                async with conn.transaction():
                    for f_id, info in file_info.items():
                        if info["defect_dets"]:
                            # Find parent asset ID
                            parent_asset_id = None
                            if f_id in frame_to_primary_asset:
                                parent_asset_id = await _upsert_component(conn, frame_to_primary_asset[f_id], is_defect=False)

                            img = await asyncio.to_thread(cv2.imread, info["local_path"])
                            dict_dets = [{"class_name": d.class_name, "confidence": d.confidence, "bbox": d.bbox.tolist()} for d in info["defect_dets"]]
                            annotated = await asyncio.to_thread(_annotate, img, dict_dets)
                            out_key = _output_key(info["s3_url"], tower_id, inspection_id, "defect")
                            await asyncio.to_thread(_s3_upload, s3_client, bucket, annotated, out_key)

                            counts_in_frame = {}
                            for d in info["defect_dets"]:
                                counts_in_frame[d.class_name] = counts_in_frame.get(d.class_name, 0) + 1
                                defect_class_counts[d.class_name] = defect_class_counts.get(d.class_name, 0) + 1
                            
                            for cls, count in counts_in_frame.items():
                                # Prepare metadata: all defect detections of this class in this frame
                                meta = {
                                    "detections": [
                                        {"class_name": d.class_name, "confidence": float(d.confidence), "bbox": d.bbox.tolist()}
                                        for d in info["defect_dets"] if d.class_name == cls
                                    ]
                                }
                                await _insert_detection(conn, inspection_uuid, out_key, "defect", cls, count, meta)
                                # Ensure defect component exists under the parent asset
                                await _upsert_component(conn, cls, is_defect=True, parent_id=parent_asset_id)

                    # Now update defect totals in final_asset_counts
                    for cls, total in defect_class_counts.items():
                        # Root defect component for the overall inspection count
                        comp_id = await _upsert_component(conn, cls, is_defect=True)
                        final_asset_counts[str(comp_id)] = total

            # 3. Update Inspection asset_counts
            if final_asset_counts:
                async with pool.acquire() as conn:
                    await _update_inspection_asset_counts(conn, inspection_uuid, final_asset_counts)
                logger.info(f"[{inspection_id}] Updated inspection asset_counts with {len(final_asset_counts)} items")


        async with pool.acquire() as conn:
            for row in rows:
                await _mark_processed(conn, row["id"])

        logger.info(f"[{inspection_id}] DONE")

    except Exception as exc:
        logger.error(f"[{inspection_id}] FAILED: {exc}", exc_info=True)
        async with pool.acquire() as conn:
            for row in rows:
                await _mark_failed(conn, row["id"])


# ---------------------------------------------------------------------------
# Cron tick: claim batch → process concurrently
# ---------------------------------------------------------------------------

async def _tick(
    pool: asyncpg.Pool,
    s3_client: Any,
    semaphore: asyncio.Semaphore,
) -> None:
    """Claim one inspection's 'uploaded' drone_images and dispatch processing."""
    async with pool.acquire() as conn:
        async with conn.transaction():
            rows = await conn.fetch(CLAIM_INSPECTION_SQL)
            if rows:
                ids = [r["id"] for r in rows]
                await conn.executemany(
                    "UPDATE pvx_file SET status = 'processing' WHERE id = $1",
                    [(fid,) for fid in ids],
                )

    if not rows:
        logger.info("[poll] No uploaded .jpg images found in pvx_file — sleeping for %ds.", POLL_INTERVAL)
        return

    logger.info(f"Claimed {len(rows)} file(s) for inspection {rows[0]['inspection_id']} for processing.")

    async def _guarded(insp_rows: List[asyncpg.Record]) -> None:
        async with semaphore:
            await _process_inspection(pool, s3_client, insp_rows)

    await asyncio.gather(_guarded(rows))


# ---------------------------------------------------------------------------
# Manual S3 Path Processing
# FIX 2: Now writes detection rows and updates inspection.asset_counts so the
# Dashboard and Tower Details page reflect manual-mode results just like DB mode.
# Requires MANUAL_INSPECTION_ID env var to link results to the correct inspection.
# ---------------------------------------------------------------------------

async def _process_manual_s3_path(
    pool: asyncpg.Pool,
    s3_client: Any,
    bucket: str,
    prefix: str,
) -> None:
    """
    Process images from a manual S3 prefix and write all results to the DB.

    Required env var:
      MANUAL_INSPECTION_ID — UUID of the pvx_inspection row to link results to.

    Outputs annotated images to:
        {FOLDER_NAME}/{tower_id}/{inspection_id}/detection/{filename}
    Only UNIQUE (deduplicated) asset images are annotated and uploaded.
    """
    inspection_uuid = os.getenv("MANUAL_INSPECTION_ID", "").strip("'\"")
    if not inspection_uuid:
        raise ValueError(
            "MANUAL_INSPECTION_ID env var is required for manual path mode "
            "so results can be linked to the correct inspection."
        )

    # Resolve human-readable IDs from the inspection and tower tables.
    async with pool.acquire() as conn:
        row = await conn.fetchrow(
            """SELECT i.id as inspection_id, t.id as tower_id 
               FROM pvx_inspection i 
               JOIN pvx_tower t ON t.uuid = i.tower_id 
               WHERE i.uuid = $1::uuid""",
            inspection_uuid,
        )
    if not row:
        raise ValueError(f"No pvx_inspection found for uuid={inspection_uuid}")
    
    inspection_id = str(row["inspection_id"])
    tower_id      = str(row["tower_id"])

    logger.info(
        f"[manual] inspection_id={inspection_id} tower_id={tower_id} "
        f"s3://{bucket}/{prefix}"
    )

    if not prefix.endswith("/"):
        prefix += "/"

    response = await asyncio.to_thread(s3_client.list_objects_v2, Bucket=bucket, Prefix=prefix)

    if "Contents" not in response:
        logger.warning("No files found in the specified S3 path.")
        return

    s3_keys = [obj["Key"] for obj in response["Contents"] if obj["Key"].lower().endswith(".jpg")]

    if not s3_keys:
        logger.warning("No JPG images found in the specified S3 path.")
        return

    logger.info(f"[manual] Found {len(s3_keys)} images.")

    modules      = await asyncio.to_thread(_init_pipeline)
    detector     = modules["detector"]
    defect_model = modules["defect_model"]
    fuser        = modules["fuser"]
    deduper      = modules["deduper"]

    file_info: Dict[str, Dict] = {}
    all_fused_detections = []

    with tempfile.TemporaryDirectory() as tmpdir:
        # Batch processing: process 10 images at a time
        BATCH_SIZE = 10
        for i in range(0, len(s3_keys), BATCH_SIZE):
            batch_keys = s3_keys[i : i + BATCH_SIZE]
            logger.info(f"[manual] Processing batch {i//BATCH_SIZE + 1}/{(len(s3_keys)-1)//BATCH_SIZE + 1} ({len(batch_keys)} file(s))")

            async def _process_manual_image(idx_in_batch: int, s3_url: str):
                absolute_idx = i + idx_in_batch + 1
                filename   = s3_url.split("/")[-1]
                local_path = os.path.join(tmpdir, filename)

                logger.info(f"[manual] [{absolute_idx}/{len(s3_keys)}] Downloading {s3_url}")
                try:
                    await asyncio.to_thread(_s3_download_to_file, s3_client, bucket, s3_url, local_path)

                    pose_task   = asyncio.to_thread(parse_dji_metadata, local_path)
                    asset_task  = asyncio.to_thread(detector.detect, local_path)
                    defect_task = asyncio.to_thread(defect_model.detect, local_path)

                    pose, asset_dets, defect_dets = await asyncio.gather(pose_task, asset_task, defect_task)

                    res = {
                        "s3_url": s3_url,
                        "filename": filename,
                        "local_path": local_path,
                        "asset_dets": asset_dets,
                        "defect_dets": defect_dets,
                        "pose": pose,
                        "fused_dets": []
                    }

                    if asset_dets:
                        fused_dets = await asyncio.to_thread(fuser.fuse, asset_dets, pose)
                        for d in fused_dets:
                            d.frame_id = filename
                        res["fused_dets"] = fused_dets

                    return res

                except Exception as e:
                    logger.error(f"[manual] Error processing {s3_url}: {e}")
                    return None
                finally:
                    gc.collect()

            batch_tasks = [_process_manual_image(idx, s3_url) for idx, s3_url in enumerate(batch_keys)]
            batch_results = await asyncio.gather(*batch_tasks)

            for res in batch_results:
                if res:
                    f_name = res["filename"]
                    file_info[f_name] = res
                    all_fused_detections.extend(res["fused_dets"])

        # ------------------------------------------------------------------
        # Global deduplication
        # ------------------------------------------------------------------
        logger.info(f"[manual] Deduplicating {len(all_fused_detections)} total asset detections")
        asset_inventory = await asyncio.to_thread(deduper.deduplicate, all_fused_detections)

        # Map from filename -> primary asset class name
        frame_to_primary_asset: Dict[str, str] = {}
        for f_name, info in file_info.items():
            if info["asset_dets"]:
                primary = max(info["asset_dets"], key=lambda d: d.confidence)
                frame_to_primary_asset[f_name] = primary.class_name

        # ------------------------------------------------------------------
        # Process and Upload
        # ------------------------------------------------------------------
        final_asset_counts: Dict[str, int] = {}
        
        # 1. Assets
        unique_asset_output_keys: Dict[str, str] = {}
        async with pool.acquire() as conn:
            async with conn.transaction():
                for class_name, unique_dets in asset_inventory.items():
                    comp_id = await _upsert_component(conn, class_name, is_defect=False)
                    final_asset_counts[str(comp_id)] = len(unique_dets)

                    frame_counts = {}
                    for d in unique_dets:
                        frame_counts[d.frame_id] = frame_counts.get(d.frame_id, 0) + 1
                    
                    for f_name, count in frame_counts.items():
                        info = file_info[f_name]
                        if f_name not in unique_asset_output_keys:
                            img = await asyncio.to_thread(cv2.imread, info["local_path"])
                            dict_dets = [{"class_name": d.class_name, "confidence": d.confidence, "bbox": d.bbox.tolist()} for d in info["asset_dets"]]
                            annotated = await asyncio.to_thread(_annotate, img, dict_dets)
                            out_key = _output_key(info["s3_url"], tower_id, inspection_id, "asset")
                            await asyncio.to_thread(_s3_upload, s3_client, bucket, annotated, out_key)
                            unique_asset_output_keys[f_name] = out_key
                        
                        # Prepare metadata: all asset detections of this class in this frame
                        meta = {
                            "detections": [
                                {"class_name": d.class_name, "confidence": float(d.confidence), "bbox": d.bbox.tolist()}
                                for d in info["asset_dets"] if d.class_name == class_name
                            ]
                        }
                        await _insert_detection(conn, inspection_uuid, unique_asset_output_keys[f_name], "asset", class_name, count, meta)

        # 2. Defects
        defect_class_counts: Dict[str, int] = {}
        async with pool.acquire() as conn:
            async with conn.transaction():
                for f_name, info in file_info.items():
                    if info["defect_dets"]:
                        parent_asset_id = None
                        if f_name in frame_to_primary_asset:
                            parent_asset_id = await _upsert_component(conn, frame_to_primary_asset[f_name], is_defect=False)

                        img = await asyncio.to_thread(cv2.imread, info["local_path"])
                        dict_dets = [{"class_name": d.class_name, "confidence": d.confidence, "bbox": d.bbox.tolist()} for d in info["defect_dets"]]
                        annotated = await asyncio.to_thread(_annotate, img, dict_dets)
                        out_key = _output_key(info["s3_url"], tower_id, inspection_id, "defect")
                        await asyncio.to_thread(_s3_upload, s3_client, bucket, annotated, out_key)

                        counts_in_frame = {}
                        for d in info["defect_dets"]:
                            counts_in_frame[d.class_name] = counts_in_frame.get(d.class_name, 0) + 1
                            defect_class_counts[d.class_name] = defect_class_counts.get(d.class_name, 0) + 1
                        
                        for cls, count in counts_in_frame.items():
                            # Prepare metadata: all defect detections of this class in this frame
                            meta = {
                                "detections": [
                                    {"class_name": d.class_name, "confidence": float(d.confidence), "bbox": d.bbox.tolist()}
                                    for d in info["defect_dets"] if d.class_name == cls
                                ]
                            }
                            await _insert_detection(conn, inspection_uuid, out_key, "defect", cls, count, meta)
                            await _upsert_component(conn, cls, is_defect=True, parent_id=parent_asset_id)

                for cls, total in defect_class_counts.items():
                    comp_id = await _upsert_component(conn, cls, is_defect=True)
                    final_asset_counts[str(comp_id)] = total

        # 3. Update Inspection asset_counts
        if final_asset_counts:
            async with pool.acquire() as conn:
                await _update_inspection_asset_counts(conn, inspection_uuid, final_asset_counts)
            logger.info(f"[manual] Updated inspection asset_counts with {len(final_asset_counts)} items")

        logger.info(f"[manual] DONE — inspection_id={inspection_id}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

async def main() -> None:
    s3_client = _build_s3_client()

    manual_path_mode = os.getenv("MANUAL_PATH_MODE", "false").lower() == "true"
    manual_s3_path   = os.getenv("MANUAL_S3_PATH",   "").strip("'\"")

    if manual_path_mode and manual_s3_path:
        bucket = os.getenv("S3_BUCKET_NAME", "").strip("'\"")
        logger.info(f"Running in MANUAL PATH MODE for S3 prefix: {manual_s3_path}")

        dsn  = _build_dsn()
        pool = await asyncpg.create_pool(dsn=dsn, min_size=2, max_size=10)
        async with pool.acquire() as conn:
            await conn.execute(ENSURE_DETECTION_TABLE_SQL)

        try:
            await _process_manual_s3_path(pool, s3_client, bucket, manual_s3_path)
        finally:
            await pool.close()
        return

    dsn = _build_dsn()
    logger.info("Connecting to PostgreSQL…")
    pool = await asyncpg.create_pool(dsn=dsn, min_size=2, max_size=10)

    async with pool.acquire() as conn:
        await conn.execute(ENSURE_DETECTION_TABLE_SQL)
        logger.info("`detection` table verified.")

    semaphore = asyncio.Semaphore(MAX_CONCURRENT)

    logger.info(
        f"PieflyVisionX inference cron running | "
        f"poll={POLL_INTERVAL}s  concurrency={MAX_CONCURRENT}"
    )

    try:
        while True:
            try:
                await _tick(pool, s3_client, semaphore)
            except Exception as exc:
                logger.error(f"Tick-level error: {exc}", exc_info=True)
            await asyncio.sleep(POLL_INTERVAL)
    finally:
        await pool.close()
        logger.info("DB pool closed. Cron stopped.")


if __name__ == "__main__":
    asyncio.run(main())
