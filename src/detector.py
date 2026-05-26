from ultralytics import YOLO
import numpy as np
import cv2
from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class Detection:
    frame_id: str
    class_name: str
    confidence: float
    bbox: np.ndarray          # [x1, y1, x2, y2] in pixels
    crop: np.ndarray          # BGR image crop (padded)
    centroid_px: np.ndarray   # [cx, cy] in pixels
    embedding: Optional[np.ndarray] = None
    world_xyz: Optional[np.ndarray] = None


class TowerDetector:
    def __init__(self, model_path: str, conf: float = 0.35, iou: float = 0.45, 
                 max_image_size: Optional[int] = 1280):
        """
        Initialize detector.
        
        Args:
            model_path: Path to YOLO model
            conf: Confidence threshold
            iou: IOU threshold
            max_image_size: Max dimension to resize image to (preserve aspect). 
                           None to skip resizing.
        """
        self.model = YOLO(model_path)
        self.conf  = conf
        self.iou   = iou
        self.max_image_size = max_image_size

    def detect(self, image_path: str) -> List[Detection]:
        img = cv2.imread(image_path)
        if img is None:
            raise FileNotFoundError(f"Could not read image: {image_path}")

        # Store original size for coordinate mapping
        orig_h, orig_w = img.shape[:2]
        scale_x, scale_y = 1.0, 1.0
        
        # Resize if image is too large (for memory efficiency)
        if self.max_image_size and max(orig_h, orig_w) > self.max_image_size:
            scale = self.max_image_size / max(orig_h, orig_w)
            new_h = int(orig_h * scale)
            new_w = int(orig_w * scale)
            img_resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)
            scale_x = orig_w / new_w
            scale_y = orig_h / new_h
            detect_img = img_resized
        else:
            detect_img = img

        results = self.model.predict(
            detect_img, conf=self.conf, iou=self.iou, verbose=False
        )[0]

        detections = []
        for box in results.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
            
            # Scale coordinates back to original image size
            x1, y1, x2, y2 = int(x1 * scale_x), int(y1 * scale_y), int(x2 * scale_x), int(y2 * scale_y)
            
            cls_name = results.names[int(box.cls[0])]
            conf_val = float(box.conf[0])

            # Expand crop by 10% on each side for better re-ID embedding
            pad_x = int((x2 - x1) * 0.10)
            pad_y = int((y2 - y1) * 0.10)
            
            # Clamp to image bounds
            crop_y1 = max(0, y1 - pad_y)
            crop_y2 = min(orig_h, y2 + pad_y)
            crop_x1 = max(0, x1 - pad_x)
            crop_x2 = min(orig_w, x2 + pad_x)
            
            # Extract crop from original image
            crop = img[crop_y1:crop_y2, crop_x1:crop_x2].copy()
            
            # Skip if crop is too small or invalid
            if crop.size == 0 or crop.shape[0] < 10 or crop.shape[1] < 10:
                continue

            detections.append(Detection(
                frame_id=image_path,
                class_name=cls_name,
                confidence=conf_val,
                bbox=np.array([x1, y1, x2, y2]),
                crop=crop,
                centroid_px=np.array([(x1 + x2) / 2, (y1 + y2) / 2])
            ))

        return detections