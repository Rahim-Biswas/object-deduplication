import numpy as np
from .metadata import DronePose, gps_to_enu, pose_to_rotation


class Projector3D:
    def __init__(self, camera_cfg: dict, component_sizes: dict):
        # Camera intrinsic matrix
        self.K = np.array([
            [camera_cfg["fx"],              0, camera_cfg["cx"]],
            [             0,  camera_cfg["fy"], camera_cfg["cy"]],
            [             0,              0,              1     ]
        ])
        self.component_sizes = component_sizes
        self.ref_pose = None   # set on the very first frame

    def estimate_depth(self, bbox: np.ndarray,
                       class_name: str) -> float:
        """
        Depth via similar triangles:
            depth = (real_height * fy) / bbox_height_pixels
        Uses known physical component height from config.
        """
        real_h  = self.component_sizes.get(class_name, 0.4)
        bbox_h  = max(bbox[3] - bbox[1], 1)
        depth   = (real_h * self.K[1, 1]) / bbox_h
        return float(depth)

    def backproject(self, cx_px: float, cy_px: float,
                    depth: float, pose: DronePose) -> np.ndarray:
        """
        Convert (cx_px, cy_px) pixel centroid + depth estimate
        into a world XYZ point in local ENU metres.
        """
        # Set reference origin on first call
        if self.ref_pose is None:
            self.ref_pose = pose

        # Camera position in ENU world coords
        t_world = gps_to_enu(
            pose.lat, pose.lon, pose.alt_abs,
            self.ref_pose.lat, self.ref_pose.lon, self.ref_pose.alt_abs
        )

        # Rotation from camera to world
        R = pose_to_rotation(pose)

        # Back-project pixel to camera-space unit ray
        uv1     = np.array([cx_px, cy_px, 1.0])
        ray_cam = np.linalg.inv(self.K) @ uv1
        ray_cam = ray_cam / np.linalg.norm(ray_cam)

        # Scale ray by depth estimate and rotate to world space
        ray_world = R @ ray_cam
        world_xyz = t_world + ray_world * depth
        return world_xyz