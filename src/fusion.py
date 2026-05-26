from typing import List
import numpy as np
from .detector import Detection
from .embedder import ReIDEmbedder
from .projector import Projector3D
from .metadata import DronePose


class DetectionFuser:
    def __init__(self, embedder: ReIDEmbedder, projector: Projector3D):
        self.embedder  = embedder
        self.projector = projector

    def fuse(self, detections: List[Detection],
             pose: DronePose) -> List[Detection]:
        """
        For each detection:
          1. Extract re-ID embedding from crop (batched)
          2. Estimate depth from bbox size + known component height
          3. Back-project centroid pixel to world XYZ
        """
        if not detections:
            return detections

        # Batch embed all crops in one GPU pass
        crops      = [d.crop for d in detections]
        embeddings = self.embedder.embed_batch(crops)

        for det, emb in zip(detections, embeddings):
            det.embedding = emb

            depth = self.projector.estimate_depth(
                det.bbox, det.class_name
            )
            det.world_xyz = self.projector.backproject(
                det.centroid_px[0],
                det.centroid_px[1],
                depth,
                pose
            )

        return detections