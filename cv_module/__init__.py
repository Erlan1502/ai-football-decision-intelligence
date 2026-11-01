"""
Scout CV Video-to-Data Module (PoC)
Includes:
- Ultralytics YOLOv11 Player & Ball Detector
- ByteTrack Multi-Object Association Tracker
- 2D Homography Pitch Projection (105x68m)
- End-to-End Tracking Pipeline & Standardized JSON Exporter
"""

from cv_module.homography import (
    compute_homography,
    project_pixel_to_pitch,
    project_bbox_to_pitch,
    get_canonical_pitch_corners,
    PITCH_LENGTH_METERS,
    PITCH_WIDTH_METERS,
)
from cv_module.detector import ScoutObjectDetector
from cv_module.tracker import ScoutByteTracker
from cv_module.pipeline import ScoutCVPipeline

__all__ = [
    "compute_homography",
    "project_pixel_to_pitch",
    "project_bbox_to_pitch",
    "get_canonical_pitch_corners",
    "PITCH_LENGTH_METERS",
    "PITCH_WIDTH_METERS",
    "ScoutObjectDetector",
    "ScoutByteTracker",
    "ScoutCVPipeline",
]

