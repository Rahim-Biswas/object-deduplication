#!/usr/bin/env python3
"""Debug script to diagnose pipeline issues."""

import yaml
from pathlib import Path
from src.detector import TowerDetector
from src.metadata import parse_dji_metadata

# Load config
cfg = yaml.safe_load(open("config.yaml"))

print("=" * 60)
print("DEBUG: Pipeline Diagnostic")
print("=" * 60)

# 1. Check images exist
images_dir = Path("data/images")
images = sorted(
    list(images_dir.glob("*.JPG")) +
    list(images_dir.glob("*.jpg"))
)
print(f"\n1. Images found: {len(images)}")
if images:
    print(f"   First 3: {[img.name for img in images[:3]]}")
    print(f"   Last 3: {[img.name for img in images[-3:]]}")

# 2. Check model exists
model_path = Path(cfg["yolo"]["model_path"])
print(f"\n2. Model path: {model_path}")
print(f"   Exists: {model_path.exists()}")
if model_path.exists():
    print(f"   Size: {model_path.stat().st_size / 1e6:.1f} MB")

# 3. Test detector on first image
print(f"\n3. Testing detector on first image...")
if images:
    first_img = str(images[0])
    print(f"   Image: {Path(first_img).name}")
    
    detector = TowerDetector(
        cfg["yolo"]["model_path"],
        cfg["yolo"]["conf_threshold"],
        cfg["yolo"]["iou_threshold"]
    )
    print(f"   Confidence threshold: {detector.conf}")
    print(f"   IOU threshold: {detector.iou}")
    
    try:
        dets = detector.detect(first_img)
        print(f"   Detections: {len(dets)}")
        if dets:
            for det in dets[:3]:
                print(f"     - {det.class_name}: conf={det.confidence:.3f}")
    except Exception as e:
        print(f"   ERROR: {e}")

# 4. Test metadata parser on first image
print(f"\n4. Testing metadata parser on first image...")
if images:
    first_img = str(images[0])
    try:
        pose = parse_dji_metadata(first_img)
        print(f"   Success! Pose: lat={pose.lat:.6f}, lon={pose.lon:.6f}, alt={pose.alt_abs:.1f}m")
    except FileNotFoundError as e:
        print(f"   ERROR: {e}")
        print(f"   (exiftool not installed or not in PATH)")
    except Exception as e:
        print(f"   ERROR: {e}")

# 5. Test on a sample of images
print(f"\n5. Testing detection on first 10 images...")
if images:
    detector = TowerDetector(
        cfg["yolo"]["model_path"],
        cfg["yolo"]["conf_threshold"],
        cfg["yolo"]["iou_threshold"]
    )
    
    total_dets = 0
    successful = 0
    for img_path in images[:10]:
        try:
            dets = detector.detect(str(img_path))
            total_dets += len(dets)
            successful += 1
        except Exception as e:
            print(f"   ERROR on {img_path.name}: {e}")
    
    print(f"   Processed: {successful}/10")
    print(f"   Total detections: {total_dets}")
    print(f"   Avg per image: {total_dets/successful if successful > 0 else 0:.1f}")

print("\n" + "=" * 60)
