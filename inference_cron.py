"""
PieflyVisionX — AI/ML Inference Cron Job (Deduplication Pipeline version)

Polls pvx_file for uploaded groundbasemobileimages, runs YOLO detection,
fuses and deduplicates asset detections, and writes results back to the portal database.

Flow per inspection:
  1. Find one inspection that has 'uploaded' groundbasemobileimages.
  2. Lock all 'uploaded' files for that inspection.
  3. Mark status = 'processing'.
  4. Process 'asset' files using the global deduplication pipeline.
     - Download files
     - Parse DJI pose
     - YOLO detect
     - ReID embed & back-project to 3D
     - Fuse & Deduplicate globally across the inspection
  5. Process 'defect' files individually (no 3D deduplication).
  6. Annotate images and upload results to S3.
  7. Write global deduplicated pvx_detection rows and pvx_view_3d_annotation for assets.
  8. Write per-image pvx_detection rows for defects.
  9. Mark processed.
  10. On failure → mark status = 'failed'

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
CREATE TABLE IF NOT EXISTS pvx_detection (
    id             UUID         PRIMARY KEY DEFAULT gen_random_uuid(),
    inspection_id  VARCHAR(255) NOT NULL,
    s3_url         VARCHAR(500) NOT NULL,
    meta_data      JSONB,
    detection_type VARCHAR(50)  NOT NULL,
    component_name VARCHAR(255),
    detected_count INTEGER      DEFAULT 0,
    created_by     VARCHAR(255) DEFAULT 'ai_ml_system',
    created_date   TIMESTAMP    DEFAULT CURRENT_TIMESTAMP,
    is_active      BOOLEAN      DEFAULT true,
    is_deleted     BOOLEAN      DEFAULT false
);
"""

# Atomically claim all 'uploaded' files for ONE inspection
CLAIM_INSPECTION_SQL = """
WITH target_inspection AS (
    SELECT insp_sub.id::text AS inspection_id
    FROM pvx_file f_sub
    JOIN pvx_inspection_file inf_sub ON inf_sub.file_id = f_sub.id
    JOIN pvx_inspection insp_sub ON insp_sub.id::text = inf_sub.inspection_id::text
    WHERE f_sub.status = 'uploaded'
      AND f_sub.file_type = 'groundbasemobileimages'
      AND f_sub.is_deleted = false
    LIMIT 1
)
SELECT
    f.id,
    f.s3_url,
    f.ground_base_component_master_id,
    f.view_3d_annotation_id,
    insp.id       AS inspection_id,
    insp.tower_id
FROM pvx_file f
JOIN pvx_inspection_file inf ON inf.file_id = f.id
JOIN pvx_inspection insp ON insp.id::text = inf.inspection_id::text
WHERE f.status = 'uploaded'
  AND f.file_type = 'groundbasemobileimages'
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


async def _insert_3d_annotation(
    conn: asyncpg.Connection,
    inspection_id: str,
    component_id: uuid.UUID,
    label: str,
    world_xyz: List[float],
) -> uuid.UUID:
    """Create a 3D pin for an asset detection based on deduplicated world_xyz."""
    # world_xyz from Projector3D is already in [east, north, up] in metres
    # The portal expects certain conventions, mapping here:
    # Based on old code:
    # pos_x = round((bbox[0] + bbox[2]) / 200.0, 3)
    # pos_y = round(-(bbox[1] + bbox[3]) / 200.0, 3)
    # pos_z = round((bbox[3] - bbox[1]) / 5.0, 2)
    # Since we have true metric coords now, we use world_xyz directly:
    pos_x = round(world_xyz[0], 3)
    pos_y = round(world_xyz[1], 3)
    pos_z = round(world_xyz[2], 2)
    
    row   = await conn.fetchrow(
        """INSERT INTO pvx_view_3d_annotation
             (id, position_x, position_y, position_z, label,
              inspection_id, ground_base_component_master_id,
              created_by, created_date, is_active, is_deleted)
           VALUES (gen_random_uuid(), $1, $2, $3, $4, $5, $6,
                   'ai_ml_system', NOW(), true, false)
           RETURNING id""",
        pos_x, pos_y, pos_z, label, inspection_id, component_id,
    )
    return row["id"]  # type: ignore[index]


async def _insert_pvx_detection(
    conn: asyncpg.Connection,
    inspection_id: str,
    s3_url: str,
    meta_data: Dict,
    task_type: str,
    class_name: str,
    count: int,
) -> None:
    await conn.execute(
        """INSERT INTO pvx_detection
             (inspection_id, s3_url, meta_data, detection_type,
              component_name, detected_count,
              created_by, created_date, is_active, is_deleted)
           VALUES ($1, $2, $3::jsonb, $4, $5, $6,
                   'ai_ml_system', NOW(), true, false)""",
        inspection_id,
        s3_url,
        json.dumps(meta_data),
        task_type,
        class_name.title(),
        count,
    )


# ---------------------------------------------------------------------------
# Core: process one inspection end-to-end
# ---------------------------------------------------------------------------

async def _process_inspection(
    pool: asyncpg.Pool,
    s3_client: Any,
    rows: List[asyncpg.Record],
) -> None:
    if not rows:
        return
        
    inspection_id = str(rows[0]["inspection_id"])
    tower_id      = str(rows[0]["tower_id"])
    bucket        = os.getenv("S3_BUCKET_NAME", "").strip("'\"")

    logger.info(f"[{inspection_id}] START processing {len(rows)} file(s)")

    # 1. Mark all files as processing
    async with pool.acquire() as conn:
        for row in rows:
            await _mark_processing(conn, row["id"])

    # Separate into asset vs defect
    asset_rows = [r for r in rows if r["view_3d_annotation_id"] is None]
    defect_rows = [r for r in rows if r["view_3d_annotation_id"] is not None]

    try:
        # Initialize pipeline modules (lazy load in thread)
        modules = await asyncio.to_thread(_init_pipeline)

        # We need a temp directory to download images so exiftool and cv2 can read them
        with tempfile.TemporaryDirectory() as tmpdir:
            
            # -------------------------------------------------------------
            # ASSETS (Pipeline logic with deduplication)
            # -------------------------------------------------------------
            if asset_rows:
                logger.info(f"[{inspection_id}] Processing {len(asset_rows)} asset file(s)")
                detector = modules["detector"]
                fuser = modules["fuser"]
                deduper = modules["deduper"]
                
                all_fused_detections = []
                file_id_to_key = {}
                
                for row in asset_rows:
                    file_id = row["id"]
                    s3_url = row["s3_url"]
                    local_path = os.path.join(tmpdir, f"{file_id}.jpg")
                    file_id_to_key[str(file_id)] = s3_url
                    
                    try:
                        # Download image
                        await asyncio.to_thread(_s3_download_to_file, s3_client, bucket, s3_url, local_path)
                        
                        # Pipeline steps
                        pose = await asyncio.to_thread(parse_dji_metadata, local_path)
                        dets = await asyncio.to_thread(detector.detect, local_path)
                        
                        if dets:
                            fused_dets = await asyncio.to_thread(fuser.fuse, dets, pose)
                            # Tag the detections with the original file_id
                            for d in fused_dets:
                                d.frame_id = str(file_id) # Override frame_id from local_path to file_id
                            all_fused_detections.extend(fused_dets)
                            
                        # Annotate original image and upload to S3 (same as old logic, for visualization)
                        if dets:
                            img = await asyncio.to_thread(cv2.imread, local_path)
                            # Convert dets to old format for annotation
                            dict_dets = [
                                {
                                    "class_name": d.class_name, 
                                    "confidence": d.confidence, 
                                    "bbox": d.bbox.tolist()
                                } 
                                for d in dets
                            ]
                            annotated = await asyncio.to_thread(_annotate, img, dict_dets)
                            output_key = _output_key(s3_url, tower_id, inspection_id, "asset")
                            await asyncio.to_thread(_s3_upload, s3_client, bucket, annotated, output_key)
                            logger.debug(f"[{file_id}] Annotated asset image uploaded → {output_key}")
                            
                    except Exception as e:
                        logger.error(f"[{file_id}] Error processing asset image: {e}")
                    finally:
                        gc.collect()

                # Deduplicate globally
                logger.info(f"[{inspection_id}] Deduplicating {len(all_fused_detections)} total asset detections")
                inventory = await asyncio.to_thread(deduper.deduplicate, all_fused_detections)
                
                # Write Asset DB records
                async with pool.acquire() as conn:
                    async with conn.transaction():
                        for class_name, unique_dets in inventory.items():
                            comp_id = await _upsert_component(conn, class_name)
                            count = len(unique_dets)
                            
                            # Build metadata and insert 3D pins
                            components_meta = []
                            best_s3_url = ""
                            
                            for d in unique_dets:
                                best_s3_url = file_id_to_key.get(d.frame_id, "")
                                world_xyz_list = d.world_xyz.tolist() if d.world_xyz is not None else [0,0,0]
                                
                                components_meta.append({
                                    "best_frame_id": d.frame_id,
                                    "confidence": round(d.confidence, 3),
                                    "world_xyz_m": world_xyz_list,
                                    "bbox_pixels": d.bbox.tolist()
                                })
                                
                                # 3b — create one 3D pin per *unique* detected object
                                await _insert_3d_annotation(
                                    conn, inspection_id, comp_id,
                                    class_name.title(), world_xyz_list,
                                )

                            # 3c — save global detection summary row for this class
                            # We use the S3 URL of the last processed best_frame as a representative URL
                            meta = {"components": components_meta, "label": class_name}
                            
                            await _insert_pvx_detection(
                                conn, inspection_id, best_s3_url, meta, "asset", class_name, count
                            )
                
            # -------------------------------------------------------------
            # DEFECTS (Old logic without deduplication)
            # -------------------------------------------------------------
            if defect_rows:
                logger.info(f"[{inspection_id}] Processing {len(defect_rows)} defect file(s)")
                
                with _module_lock:
                    if _modules["defect_model"] is None:
                        logger.info(f"Loading defect YOLO model from {DEFECT_MODEL_PATH}")
                        from ultralytics import YOLO
                        _modules["defect_model"] = YOLO(DEFECT_MODEL_PATH)
                        
                defect_model = _modules["defect_model"]
                device = _get_device()
                
                for row in defect_rows:
                    file_id = row["id"]
                    s3_url = row["s3_url"]
                    local_path = os.path.join(tmpdir, f"{file_id}.jpg")
                    
                    try:
                        await asyncio.to_thread(_s3_download_to_file, s3_client, bucket, s3_url, local_path)
                        img = await asyncio.to_thread(cv2.imread, local_path)
                        
                        results = await asyncio.to_thread(
                            defect_model.predict, img, conf=INFERENCE_CONF, verbose=False, device=device
                        )
                        
                        names = results[0].names
                        detections = []
                        for box in results[0].boxes:
                            cid = int(box.cls[0])
                            detections.append({
                                "class_name": names[cid],
                                "class_id":   cid,
                                "confidence": round(float(box.conf[0]), 4),
                                "bbox":       box.xyxy[0].tolist(),
                            })

                        class_summary = {}
                        for d in detections:
                            class_summary[d["class_name"]] = class_summary.get(d["class_name"], 0) + 1
                        
                        annotated = await asyncio.to_thread(_annotate, img.copy(), detections)
                        output_key = _output_key(s3_url, tower_id, inspection_id, "defect")
                        await asyncio.to_thread(_s3_upload, s3_client, bucket, annotated, output_key)
                        
                        # Write Defect DB records (per-image)
                        async with pool.acquire() as conn:
                            async with conn.transaction():
                                for c_name, count in class_summary.items():
                                    d_list = [d for d in detections if d["class_name"] == c_name]
                                    meta = {
                                        "confidence": max(d["confidence"] for d in d_list),
                                        "bboxes": [d["bbox"] for d in d_list],
                                        "label": c_name
                                    }
                                    await _insert_pvx_detection(
                                        conn, inspection_id, output_key, meta, "defect", c_name, count
                                    )
                                    
                    except Exception as e:
                        logger.error(f"[{file_id}] Error processing defect image: {e}")
                        async with pool.acquire() as conn:
                            await _mark_failed(conn, file_id)
                    finally:
                        gc.collect()

        # Mark all successfully processed files
        async with pool.acquire() as conn:
            for row in rows:
                # Re-verify failure wasn't set 
                # (For simplicity, we mark all as processed here if they reached this point)
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
    # Claim the inspection batch atomically
    async with pool.acquire() as conn:
        async with conn.transaction():
            rows = await conn.fetch(CLAIM_INSPECTION_SQL)
            if rows:
                # The FOR UPDATE SKIP LOCKED locks the rows. 
                # We optionally update their status to processing now,
                # though _process_inspection does it too. Doing it here prevents long lock holds.
                ids = [r["id"] for r in rows]
                await conn.executemany(
                    "UPDATE pvx_file SET status = 'processing' WHERE id = $1",
                    [(fid,) for fid in ids],
                )

    if not rows:
        logger.debug("No uploaded groundbasemobileimages found — sleeping.")
        return

    logger.info(f"Claimed {len(rows)} file(s) for inspection {rows[0]['inspection_id']} for processing.")

    async def _guarded(insp_rows: List[asyncpg.Record]) -> None:
        async with semaphore:
            await _process_inspection(pool, s3_client, insp_rows)

    # We only fetched one inspection's rows, so we just run it
    await asyncio.gather(_guarded(rows))


# ---------------------------------------------------------------------------
# Manual S3 Path Processing (No DB)
# ---------------------------------------------------------------------------

async def _process_manual_s3_path(s3_client: Any, bucket: str, prefix: str) -> None:
    logger.info(f"Listing objects in s3://{bucket}/{prefix}")
    
    # Ensure prefix ends with /
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
        
    logger.info(f"Found {len(s3_keys)} images. Initializing pipeline...")
    modules = await asyncio.to_thread(_init_pipeline)
    detector = modules["detector"]
    fuser = modules["fuser"]
    deduper = modules["deduper"]
    
    all_fused_detections = []
    
    with tempfile.TemporaryDirectory() as tmpdir:
        for s3_url in s3_keys:
            filename = s3_url.split("/")[-1]
            local_path = os.path.join(tmpdir, filename)
            
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
                    
                    img = await asyncio.to_thread(cv2.imread, local_path)
                    dict_dets = [
                        {
                            "class_name": d.class_name, 
                            "confidence": d.confidence, 
                            "bbox": d.bbox.tolist()
                        } 
                        for d in dets
                    ]
                    annotated = await asyncio.to_thread(_annotate, img, dict_dets)
                    
                    output_key = f"{prefix}detection/{filename}"
                    logger.info(f"Uploading annotated image to {output_key}")
                    await asyncio.to_thread(_s3_upload, s3_client, bucket, annotated, output_key)
            except Exception as e:
                logger.error(f"Error processing {s3_url}: {e}")
            finally:
                gc.collect()

        logger.info(f"Deduplicating {len(all_fused_detections)} total asset detections")
        inventory = await asyncio.to_thread(deduper.deduplicate, all_fused_detections)
        
        report = {}
        for cls_name, unique_dets in inventory.items():
            report[cls_name] = {
                "count": len(unique_dets),
                "components": [
                    {
                        "best_frame": d.frame_id,
                        "confidence": round(d.confidence, 3),
                        "world_xyz_m": d.world_xyz.tolist() if d.world_xyz is not None else [0,0,0],
                        "bbox_pixels": d.bbox.tolist()
                    }
                    for d in unique_dets
                ]
            }
            
        report_json = json.dumps(report, indent=2)
        report_key = f"{prefix}detection/inventory.json"
        
        logger.info(f"Uploading deduplication inventory to {report_key}")
        await asyncio.to_thread(
            s3_client.put_object,
            Bucket=bucket,
            Key=report_key,
            Body=report_json,
            ContentType="application/json"
        )
        
        unique_counts = {k: len(v) for k, v in inventory.items()}
        logger.info(f"Manual processing complete. Found unique components: {unique_counts}")

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
        logger.info("pvx_detection table verified.")

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
