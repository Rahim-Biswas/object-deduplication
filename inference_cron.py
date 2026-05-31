"""
PieflyVisionX — AI/ML Inference Cron Job (Deduplication Pipeline version)

Polls pvx_file for uploaded mobile_images, runs YOLO detection,
fuses and deduplicates asset detections, and writes results back to the portal database.

Flow per inspection:
  1. Find one inspection that has 'uploaded' mobile_images in pvx_file.
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
# Note: BATCH_SIZE is no longer used for fetching files, we process by inspection.
MAX_CONCURRENT   = int(os.getenv("CRON_MAX_CONCURRENT",  "1")) # Number of concurrent inspections

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

            # Assets use pipeline config
            model_path = cfg["yolo"].get("model_path", ASSET_MODEL_PATH)
            conf = cfg["yolo"].get("conf_threshold", INFERENCE_CONF)
            iou = cfg["yolo"].get("iou_threshold", 0.45)
            max_img_size = cfg.get("max_image_size", 1280)

            _modules["detector"] = TowerDetector(model_path, conf, iou, max_image_size=max_img_size)
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
CREATE TABLE IF NOT EXISTS detection (
    id             UUID         PRIMARY KEY DEFAULT gen_random_uuid(),
    inspection_id  UUID         NOT NULL,
    s3_url         VARCHAR(500) NOT NULL,
    detection_type VARCHAR(50)  NOT NULL,
    component_name VARCHAR(255),
    detected_count INTEGER      DEFAULT 0,
    created_by     VARCHAR(255) DEFAULT 'ai_ml_system',
    created_date   TIMESTAMP    DEFAULT CURRENT_TIMESTAMP,
    is_active      BOOLEAN      DEFAULT true,
    is_deleted     BOOLEAN      DEFAULT false
);
"""

# Atomically claim all 'uploaded' mobile_images for ONE inspection.
# pvx_file links to inspection via the pvx_inspection_file join table.
CLAIM_INSPECTION_SQL = """
WITH target_inspection AS (
    SELECT insp_sub.id::text AS inspection_id
    FROM pvx_file f_sub
    JOIN pvx_inspection_file inf_sub ON inf_sub.file_id = f_sub.id
    JOIN pvx_inspection insp_sub ON insp_sub.id::text = inf_sub.inspection_id::text
    WHERE f_sub.status    = 'uploaded'
      AND f_sub.file_type = 'mobile_images'
      AND f_sub.is_deleted = false
    LIMIT 1
)
SELECT
    f.id,
    f.s3_url,
    insp.id       AS inspection_id,
    insp.tower_id
FROM pvx_file f
JOIN pvx_inspection_file inf ON inf.file_id = f.id
JOIN pvx_inspection insp ON insp.id::text = inf.inspection_id::text
WHERE f.status     = 'uploaded'
  AND f.file_type  = 'mobile_images'
  AND f.is_deleted = false
  AND insp.id::text = (SELECT inspection_id FROM target_inspection)
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


async def _upsert_component(conn: asyncpg.Connection, class_name: str) -> uuid.UUID:
    """Upsert a component into pvx_ground_base_component_master and return its UUID."""
    name = class_name.title()
    code = class_name.upper().replace(" ", "_")
    row  = await conn.fetchrow(
        """INSERT INTO pvx_ground_base_component_master
             (id, component_name, component_code, created_by, created_date, is_active, is_deleted)
           VALUES (gen_random_uuid(), $1, $2, 'ai_ml_system', NOW(), true, false)
           ON CONFLICT (component_code)
           DO UPDATE SET component_name = EXCLUDED.component_name
           RETURNING id""",
        name, code,
    )
    return row["id"]  # type: ignore[index]


async def _insert_detection(
    conn: asyncpg.Connection,
    inspection_id: str,
    s3_url: str,
    detection_type: str,
    component_name: str,
    detected_count: int,
) -> None:
    """Insert one row into the `detection` table for a single image + component."""
    await conn.execute(
        """INSERT INTO detection
             (inspection_id, s3_url, detection_type,
              component_name, detected_count,
              created_by, created_date, is_active, is_deleted)
           VALUES ($1::uuid, $2, $3, $4, $5,
                   'ai_ml_system', NOW(), true, false)""",
        inspection_id,
        s3_url,
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
             WHERE id = $2::uuid""",
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
    Process all mobile_images for one inspection end-to-end.

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

    inspection_id = str(rows[0]["inspection_id"])
    tower_id      = str(rows[0]["tower_id"])
    bucket        = os.getenv("S3_BUCKET_NAME", "").strip("'\"")

    logger.info(f"[{inspection_id}] START processing {len(rows)} file(s)")

    # Mark all files as processing
    async with pool.acquire() as conn:
        for row in rows:
            await _mark_processing(conn, row["id"])

    try:
        # Initialize pipeline modules (lazy load in thread)
        modules = await asyncio.to_thread(_init_pipeline)

        with tempfile.TemporaryDirectory() as tmpdir:

            # ------------------------------------------------------------------
            # ASSETS — global deduplication pipeline
            # ------------------------------------------------------------------
            logger.info(f"[{inspection_id}] Processing {len(rows)} asset file(s) with deduplication")
            detector = modules["detector"]
            fuser    = modules["fuser"]
            deduper  = modules["deduper"]

            # file_id → {s3_url, local_path, raw_dets, annotated_img}
            file_info: Dict[str, Dict] = {}
            all_fused_detections = []

            for row in rows:
                file_id    = str(row["id"])
                s3_url     = row["s3_url"]
                local_path = os.path.join(tmpdir, f"{file_id}.jpg")

                file_info[file_id] = {
                    "s3_url":     s3_url,
                    "local_path": local_path,
                    "raw_dets":   [],
                }

                try:
                    await asyncio.to_thread(
                        _s3_download_to_file, s3_client, bucket, s3_url, local_path
                    )
                    pose = await asyncio.to_thread(parse_dji_metadata, local_path)
                    dets = await asyncio.to_thread(detector.detect, local_path)

                    if dets:
                        fused_dets = await asyncio.to_thread(fuser.fuse, dets, pose)
                        for d in fused_dets:
                            d.frame_id = file_id
                        all_fused_detections.extend(fused_dets)
                        file_info[file_id]["raw_dets"] = dets

                except Exception as e:
                    logger.error(f"[{file_id}] Error in detection pass: {e}")
                finally:
                    gc.collect()

            # ------------------------------------------------------------------
            # Global deduplication
            # ------------------------------------------------------------------
            logger.info(
                f"[{inspection_id}] Deduplicating {len(all_fused_detections)} total asset detections"
            )
            inventory = await asyncio.to_thread(deduper.deduplicate, all_fused_detections)

            # Collect the set of file_ids that have at least one unique detection
            unique_file_ids: set = set()
            for unique_dets in inventory.values():
                for d in unique_dets:
                    unique_file_ids.add(d.frame_id)

            logger.info(
                f"[{inspection_id}] {len(unique_file_ids)} unique images after deduplication"
            )

            # ------------------------------------------------------------------
            # Upload ONLY unique images and build per-image detection inserts
            # ------------------------------------------------------------------
            # file_id → output_s3_key (for unique images only)
            unique_output_keys: Dict[str, str] = {}

            for file_id in unique_file_ids:
                info       = file_info[file_id]
                raw_dets   = info["raw_dets"]
                local_path = info["local_path"]
                s3_url     = info["s3_url"]

                try:
                    img = await asyncio.to_thread(cv2.imread, local_path)
                    dict_dets = [
                        {
                            "class_name": d.class_name,
                            "confidence": d.confidence,
                            "bbox":       d.bbox.tolist(),
                        }
                        for d in raw_dets
                    ]
                    annotated  = await asyncio.to_thread(_annotate, img, dict_dets)
                    output_key = _output_key(s3_url, tower_id, inspection_id, "asset")
                    await asyncio.to_thread(_s3_upload, s3_client, bucket, annotated, output_key)
                    unique_output_keys[file_id] = output_key
                    logger.debug(f"[{file_id}] Unique annotated image uploaded → {output_key}")
                except Exception as e:
                    logger.error(f"[{file_id}] Error uploading unique image: {e}")

            # ------------------------------------------------------------------
            # Write DB records: `detection` rows + inspection.asset_counts
            # ------------------------------------------------------------------
            async with pool.acquire() as conn:
                async with conn.transaction():
                    # Track {component_uuid: total_unique_count} for asset_counts
                    asset_counts: Dict[str, int] = {}

                    for class_name, unique_dets in inventory.items():
                        comp_id = await _upsert_component(conn, class_name)
                        comp_id_str = str(comp_id)

                        # Accumulate total unique count for this component
                        asset_counts[comp_id_str] = len(unique_dets)

                        # Insert one detection row per unique image for this component
                        # Group unique detections by frame_id to get per-image counts
                        frame_class_counts: Dict[str, int] = {}
                        for d in unique_dets:
                            frame_class_counts[d.frame_id] = (
                                frame_class_counts.get(d.frame_id, 0) + 1
                            )

                        for frame_id, img_count in frame_class_counts.items():
                            output_key = unique_output_keys.get(frame_id, "")
                            if not output_key:
                                continue  # image upload failed; skip
                            await _insert_detection(
                                conn,
                                inspection_id,
                                output_key,
                                "asset",
                                class_name,
                                img_count,
                            )

                    # Update the inspection.asset_counts JSONB column
                    if asset_counts:
                        await _update_inspection_asset_counts(
                            conn, inspection_id, asset_counts
                        )
                        logger.info(
                            f"[{inspection_id}] Updated asset_counts: {asset_counts}"
                        )

        # Mark all files as processed
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
    """Claim one inspection's 'uploaded' mobile_images and dispatch processing."""
    # Claim the inspection batch atomically
    async with pool.acquire() as conn:
        async with conn.transaction():
            rows = await conn.fetch(CLAIM_INSPECTION_SQL)
            if rows:
                # Pre-mark as 'processing' inside the transaction so the lock
                # is released quickly while the heavy processing runs outside.
                ids = [r["id"] for r in rows]
                await conn.executemany(
                    "UPDATE pvx_file SET status = 'processing' WHERE id = $1",
                    [(fid,) for fid in ids],
                )

    if not rows:
        logger.debug("No uploaded mobile_images found — sleeping.")
        return

    logger.info(f"Claimed {len(rows)} file(s) for inspection {rows[0]['inspection_id']} for processing.")

    async def _guarded(insp_rows: List[asyncpg.Record]) -> None:
        async with semaphore:
            await _process_inspection(pool, s3_client, insp_rows)

    await asyncio.gather(_guarded(rows))


# ---------------------------------------------------------------------------
# Manual S3 Path Processing (No DB)
# ---------------------------------------------------------------------------

async def _process_manual_s3_path(s3_client: Any, bucket: str, prefix: str) -> None:
    """
    Process images from a manual S3 prefix (no DB).

    Outputs are written to:
        {FOLDER_NAME}/{input_folder_name}/detection/{filename}
    where input_folder_name is the last segment of the input prefix.

    Only UNIQUE (deduplicated) images are annotated and uploaded.
    """
    logger.info(f"Listing objects in s3://{bucket}/{prefix}")

    # Ensure prefix ends with /
    if not prefix.endswith("/"):
        prefix += "/"

    # Derive input folder name from the last non-empty segment of the prefix
    # e.g. "towers/site_abc/images/" -> "images"
    folder_name   = os.getenv("FOLDER_NAME", "deep_learning").strip("'\"")
    input_folder  = [p for p in prefix.rstrip("/").split("/") if p][-1]
    output_prefix = f"{folder_name}/{input_folder}/detection/"

    response = await asyncio.to_thread(s3_client.list_objects_v2, Bucket=bucket, Prefix=prefix)

    if "Contents" not in response:
        logger.warning("No files found in the specified S3 path.")
        return

    s3_keys = [obj["Key"] for obj in response["Contents"] if obj["Key"].lower().endswith(".jpg")]

    if not s3_keys:
        logger.warning("No JPG images found in the specified S3 path.")
        return

    logger.info(f"Found {len(s3_keys)} images. Output will go to s3://{bucket}/{output_prefix}")

    modules  = await asyncio.to_thread(_init_pipeline)
    detector = modules["detector"]
    fuser    = modules["fuser"]
    deduper  = modules["deduper"]

    # filename -> {local_path, raw_dets} -- store everything; upload ONLY unique ones later
    file_info: Dict[str, Dict] = {}
    all_fused_detections = []

    with tempfile.TemporaryDirectory() as tmpdir:
        # ------------------------------------------------------------------
        # Pass 1: download, detect, fuse -- do NOT upload yet
        # ------------------------------------------------------------------
        for s3_url in s3_keys:
            filename   = s3_url.split("/")[-1]
            local_path = os.path.join(tmpdir, filename)
            file_info[filename] = {"local_path": local_path, "raw_dets": []}

            try:
                logger.info(f"Downloading {s3_url}")
                await asyncio.to_thread(_s3_download_to_file, s3_client, bucket, s3_url, local_path)

                pose = await asyncio.to_thread(parse_dji_metadata, local_path)
                dets = await asyncio.to_thread(detector.detect, local_path)

                if dets:
                    fused_dets = await asyncio.to_thread(fuser.fuse, dets, pose)
                    for d in fused_dets:
                        d.frame_id = filename
                    all_fused_detections.extend(fused_dets)
                    file_info[filename]["raw_dets"] = dets

            except Exception as e:
                logger.error(f"Error processing {s3_url}: {e}")
            finally:
                gc.collect()

        # ------------------------------------------------------------------
        # Pass 2: global deduplication
        # ------------------------------------------------------------------
        logger.info(f"Deduplicating {len(all_fused_detections)} total asset detections")
        inventory = await asyncio.to_thread(deduper.deduplicate, all_fused_detections)

        # Collect filenames that appear in any unique detection
        unique_filenames: set = set()
        for unique_dets in inventory.values():
            for d in unique_dets:
                unique_filenames.add(d.frame_id)

        total        = len(s3_keys)
        unique_count = len(unique_filenames)
        logger.info(f"{unique_count} unique images after deduplication (skipped {total - unique_count})")

        # ------------------------------------------------------------------
        # Pass 3: annotate and upload ONLY unique images
        # ------------------------------------------------------------------
        for filename in unique_filenames:
            info       = file_info[filename]
            local_path = info["local_path"]
            raw_dets   = info["raw_dets"]
            output_key = f"{output_prefix}{filename}"
            try:
                img = await asyncio.to_thread(cv2.imread, local_path)
                dict_dets = [
                    {"class_name": d.class_name, "confidence": d.confidence, "bbox": d.bbox.tolist()}
                    for d in raw_dets
                ]
                annotated = await asyncio.to_thread(_annotate, img, dict_dets)
                logger.info(f"Uploading unique annotated image -> s3://{bucket}/{output_key}")
                await asyncio.to_thread(_s3_upload, s3_client, bucket, annotated, output_key)
            except Exception as e:
                logger.error(f"Error uploading unique image {filename}: {e}")

        # ------------------------------------------------------------------
        # Upload inventory JSON to same output folder
        # ------------------------------------------------------------------
        report = {}
        for cls_name, unique_dets in inventory.items():
            report[cls_name] = {
                "count": len(unique_dets),
                "components": [
                    {
                        "best_frame":  d.frame_id,
                        "confidence":  round(d.confidence, 3),
                        "world_xyz_m": d.world_xyz.tolist() if d.world_xyz is not None else [0, 0, 0],
                        "bbox_pixels": d.bbox.tolist(),
                    }
                    for d in unique_dets
                ],
            }

        report_key = f"{output_prefix}inventory.json"
        logger.info(f"Uploading deduplication inventory -> s3://{bucket}/{report_key}")
        await asyncio.to_thread(
            s3_client.put_object,
            Bucket=bucket,
            Key=report_key,
            Body=json.dumps(report, indent=2),
            ContentType="application/json",
        )

        unique_counts = {k: len(v) for k, v in inventory.items()}
        logger.info(f"Manual processing complete. Unique components: {unique_counts}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

async def main() -> None:
    s3_client = _build_s3_client()
    
    manual_path_mode = os.getenv("MANUAL_PATH_MODE", "false").lower() == "true"
    manual_s3_path = os.getenv("MANUAL_S3_PATH", "").strip("'\"")
    
    if manual_path_mode and manual_s3_path:
        bucket = os.getenv("S3_BUCKET_NAME", "").strip("'\"")
        logger.info(f"Running in MANUAL PATH MODE for S3 prefix: {manual_s3_path}")
        await _process_manual_s3_path(s3_client, bucket, manual_s3_path)
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
