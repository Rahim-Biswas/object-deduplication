#!/usr/bin/env python3
"""Quick test to verify the pipeline works with smaller images."""

import yaml
from pathlib import Path
from src.metadata import parse_dji_metadata
from src.detector import TowerDetector
from src.embedder import ReIDEmbedder
from src.projector import Projector3D
from src.fusion import DetectionFuser

cfg = yaml.safe_load(open("config.yaml"))

detector  = TowerDetector(
    cfg["yolo"]["model_path"],
    cfg["yolo"]["conf_threshold"],
    cfg["yolo"]["iou_threshold"],
    max_image_size=cfg.get("max_image_size", 640)
)
embedder  = ReIDEmbedder()
projector = Projector3D(cfg["camera"], cfg["component_sizes"])
fuser     = DetectionFuser(embedder, projector)

# Test on first 5 images only
images = sorted(list(Path("data/images").glob("*.JPG")) + list(Path("data/images").glob("*.jpg")))[:5]

print(f"Testing on {len(images)} images...")
total_dets = 0

for i, img_path in enumerate(images, 1):
    try:
        print(f"\n{i}. {img_path.name}...", end=" ")
        
        # Detect
        dets = detector.detect(str(img_path))
        print(f"detected {len(dets)}", end=" ")
        
        if not dets:
            print("✓")
            continue
        
        # Parse metadata
        pose = parse_dji_metadata(str(img_path))
        print("pose parsed", end=" ")
        
        # Fuse
        dets = fuser.fuse(dets, pose)
        print(f"fused ✓")
        total_dets += len(dets)
        
    except Exception as e:
        print(f"ERROR: {type(e).__name__}: {str(e)[:50]}")

print(f"\nTotal detections: {total_dets}")
