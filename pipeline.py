import yaml
import json
from pathlib import Path
from tqdm import tqdm
import gc  # For memory cleanup

from src.metadata import parse_dji_metadata
from src.detector import TowerDetector
from src.embedder import ReIDEmbedder
from src.projector import Projector3D
from src.fusion import DetectionFuser
from src.deduplicator import Deduplicator


def run_pipeline(image_dir: str, config_path: str = "config.yaml"):
    # Load config
    cfg = yaml.safe_load(open(config_path))

    # Initialise all modules
    detector  = TowerDetector(
        cfg["yolo"]["model_path"],
        cfg["yolo"]["conf_threshold"],
        cfg["yolo"]["iou_threshold"],
        max_image_size=cfg.get("max_image_size", 1280)  # Resize to 1280px for memory efficiency
    )
    embedder  = ReIDEmbedder()
    projector = Projector3D(cfg["camera"], cfg["component_sizes"])
    fuser     = DetectionFuser(embedder, projector)
    deduper   = Deduplicator(cfg.get("dedup", {}))

    # Collect all images (supports JPG and jpg)
    images = sorted(
        list(Path(image_dir).glob("*.JPG")) +
        list(Path(image_dir).glob("*.jpg"))
    )
    print(f"Found {len(images)} images in {image_dir}")

    all_detections = []

    for img_path in tqdm(images, desc="Processing frames"):
        try:
            # Step 1: Parse drone pose from EXIF
            pose = parse_dji_metadata(str(img_path))

            # Step 2: Run YOLO
            dets = detector.detect(str(img_path))
            if not dets:
                continue

            # Step 3: Embed + back-project to 3D
            dets = fuser.fuse(dets, pose)
            all_detections.extend(dets)

        except Exception as e:
            print(f"  Warning: skipped {img_path.name} — {e}")
            continue
        finally:
            # Clean up memory periodically
            gc.collect()

    print(f"\nTotal raw detections across all frames: {len(all_detections)}")

    # Step 4: Deduplicate globally across all frames
    inventory = deduper.deduplicate(all_detections)

    # Step 5: Build JSON output
    report = {}
    for cls_name, unique_dets in inventory.items():
        report[cls_name] = {
            "count": len(unique_dets),
            "components": [
                {
                    "best_frame": Path(d.frame_id).name,
                    "confidence": round(d.confidence, 3),
                    "world_xyz_m": d.world_xyz.tolist(),
                    "bbox_pixels": d.bbox.tolist()
                }
                for d in unique_dets
            ]
        }

    Path("outputs").mkdir(exist_ok=True)
    with open("outputs/inventory.json", "w") as f:
        json.dump(report, f, indent=2)

    # Print summary
    print("\n===== COMPONENT INVENTORY =====")
    total = 0
    for cls_name, data in report.items():
        print(f"  {cls_name}: {data['count']} unique components")
        total += data["count"]
    print(f"  TOTAL: {total} components")
    print("Output saved to outputs/inventory.json")

    return report


if __name__ == "__main__":
    run_pipeline("data/images")