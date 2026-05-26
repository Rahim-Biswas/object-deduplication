import subprocess, json, math, shutil
from dataclasses import dataclass
import numpy as np


# Detect exiftool once at module import time and track whether we've warned
_EXIFTOOL_PATH = shutil.which("exiftool")
_exiftool_warned = False


@dataclass
class DronePose:
    lat: float
    lon: float
    alt_abs: float      # absolute altitude in metres
    alt_rel: float      # relative to takeoff point in metres
    gimbal_yaw: float   # degrees
    gimbal_pitch: float
    gimbal_roll: float
    flight_yaw: float   # drone body heading


def parse_dji_metadata(image_path: str) -> DronePose:
    """Extract pose from DJI XMP/EXIF using exiftool.
    Falls back to default values if exiftool is unavailable.
    """
    # If exiftool binary is not available, warn once and return defaults.
    global _exiftool_warned
    if _EXIFTOOL_PATH is None:
        if not _exiftool_warned:
            print("    Note: Using default pose (exiftool unavailable or not in PATH)")
            _exiftool_warned = True
        return DronePose(
            lat=0.0,
            lon=0.0,
            alt_abs=100.0,
            alt_rel=100.0,
            gimbal_yaw=0.0,
            gimbal_pitch=-90.0,
            gimbal_roll=0.0,
            flight_yaw=0.0,
        )

    try:
        cmd = [
            _EXIFTOOL_PATH, "-j",
            "-GPSLatitude", "-GPSLongitude",
            "-AbsoluteAltitude", "-RelativeAltitude",
            "-GimbalYawDegree", "-GimbalPitchDegree",
            "-GimbalRollDegree", "-FlightYawDegree",
            image_path
        ]
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=5)
        if result.returncode != 0 or not result.stdout:
            # exiftool failed or no output; use defaults
            raise RuntimeError("exiftool returned no data")

        data = json.loads(result.stdout)[0]

        return DronePose(
            lat=float(data.get("GPSLatitude", 0)),
            lon=float(data.get("GPSLongitude", 0)),
            alt_abs=float(data.get("AbsoluteAltitude", 0)),
            alt_rel=float(data.get("RelativeAltitude", 0)),
            gimbal_yaw=float(data.get("GimbalYawDegree", 0)),
            gimbal_pitch=float(data.get("GimbalPitchDegree", 0)),
            gimbal_roll=float(data.get("GimbalRollDegree", 0)),
            flight_yaw=float(data.get("FlightYawDegree", 0)),
        )
    except (subprocess.TimeoutExpired, json.JSONDecodeError, 
            RuntimeError, KeyError) as e:
        # exiftool failed during execution/parsing; warn once and return defaults
        if not _exiftool_warned:
            print(f"    Note: Using default pose (exiftool failed: {type(e).__name__})")
            _exiftool_warned = True
        return DronePose(
            lat=0.0,
            lon=0.0,
            alt_abs=100.0,  # Assume 100m altitude
            alt_rel=100.0,
            gimbal_yaw=0.0,
            gimbal_pitch=-90.0,  # Straight down
            gimbal_roll=0.0,
            flight_yaw=0.0,
        )


def gps_to_enu(lat, lon, alt, ref_lat, ref_lon, ref_alt):
    """Convert GPS coords to local ENU (East-North-Up) in metres.
    ref_* = coordinates of the first image (origin point).
    """
    R_earth = 6371000.0
    dlat = math.radians(lat - ref_lat)
    dlon = math.radians(lon - ref_lon)
    east  = dlon * R_earth * math.cos(math.radians(ref_lat))
    north = dlat * R_earth
    up    = alt - ref_alt
    return np.array([east, north, up])


def pose_to_rotation(pose: DronePose) -> np.ndarray:
    """Build 3x3 rotation matrix from gimbal yaw/pitch/roll."""
    yaw   = math.radians(pose.gimbal_yaw)
    pitch = math.radians(pose.gimbal_pitch)
    roll  = math.radians(pose.gimbal_roll)

    Rz = np.array([
        [math.cos(yaw), -math.sin(yaw), 0],
        [math.sin(yaw),  math.cos(yaw), 0],
        [0,              0,             1]
    ])
    Ry = np.array([
        [ math.cos(pitch), 0, math.sin(pitch)],
        [ 0,               1, 0              ],
        [-math.sin(pitch), 0, math.cos(pitch)]
    ])
    Rx = np.array([
        [1, 0,             0            ],
        [0, math.cos(roll), -math.sin(roll)],
        [0, math.sin(roll),  math.cos(roll)]
    ])
    return Rz @ Ry @ Rx