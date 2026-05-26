from typing import List, Dict
import numpy as np
from scipy.spatial.distance import cdist
from .detector import Detection


class Deduplicator:
    """
    Deduplicates detections across all frames using embedding similarity.
    """
    def __init__(self, config: Dict = None):
        if config is None:
            config = {}
        self.distance_threshold = config.get("distance_threshold", 0.5)
        self.confidence_threshold = config.get("confidence_threshold", 0.5)

    def deduplicate(self, detections: List[Detection]) -> Dict[str, List[Detection]]:
        """
        Groups detections by class, then clusters within each class using
        embedding similarity to identify unique components.
        
        Returns:
            Dict mapping class_name -> list of unique detections (best per cluster)
        """
        if not detections:
            return {}
        
        # Group by class
        by_class = {}
        for det in detections:
            if det.class_name not in by_class:
                by_class[det.class_name] = []
            by_class[det.class_name].append(det)
        
        inventory = {}
        
        # Deduplicate within each class
        for class_name, class_dets in by_class.items():
            unique_dets = self._cluster_by_embedding(class_dets)
            inventory[class_name] = unique_dets
        
        return inventory
    
    def _cluster_by_embedding(self, detections: List[Detection]) -> List[Detection]:
        """
        Cluster detections by embedding similarity and return the best detection per cluster.
        """
        if not detections:
            return []
        
        if len(detections) == 1:
            return detections
        
        # Extract embeddings
        embeddings = np.array([det.embedding for det in detections])
        
        # Compute pairwise distances
        distances = cdist(embeddings, embeddings, metric="cosine")
        
        # Simple clustering: greedy assignment to clusters
        clusters = []
        assigned = set()
        
        for i in range(len(detections)):
            if i in assigned:
                continue
            
            cluster = [i]
            assigned.add(i)
            
            # Find all detections close to detection i
            for j in range(i + 1, len(detections)):
                if j not in assigned and distances[i, j] < self.distance_threshold:
                    cluster.append(j)
                    assigned.add(j)
            
            clusters.append(cluster)
        
        # For each cluster, pick the detection with highest confidence
        unique_dets = []
        for cluster in clusters:
            best_idx = max(cluster, key=lambda idx: detections[idx].confidence)
            unique_dets.append(detections[best_idx])
        
        return unique_dets