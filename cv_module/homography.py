import numpy as np
from typing import Tuple, List, Optional

# Standard FIFA football pitch dimensions in meters
PITCH_LENGTH_METERS = 105.0
PITCH_WIDTH_METERS = 68.0


def get_canonical_pitch_corners() -> np.ndarray:
    """
    Returns standard 2D metric coordinates of the 4 pitch corners (in meters):
    (0,0) Top-Left, (105,0) Top-Right, (105,68) Bottom-Right, (0,68) Bottom-Left.
    """
    return np.array([
        [0.0, 0.0],
        [PITCH_LENGTH_METERS, 0.0],
        [PITCH_LENGTH_METERS, PITCH_WIDTH_METERS],
        [0.0, PITCH_WIDTH_METERS],
    ], dtype=np.float32)


def compute_homography_dlt(src_points: np.ndarray, dst_points: np.ndarray) -> np.ndarray:
    """
    Direct Linear Transformation (DLT) via SVD to compute the 3x3 homography matrix.
    Pure NumPy implementation ensuring zero-dependency resilience and mathematical precision.
    """
    src = np.asarray(src_points, dtype=np.float64)
    dst = np.asarray(dst_points, dtype=np.float64)

    if len(src) < 4 or len(dst) < 4:
        raise ValueError("At least 4 corresponding point pairs are required for homography.")

    A = []
    for i in range(len(src)):
        x, y = src[i, 0], src[i, 1]
        u, v = dst[i, 0], dst[i, 1]
        A.append([-x, -y, -1.0, 0.0, 0.0, 0.0, x * u, y * u, u])
        A.append([0.0, 0.0, 0.0, -x, -y, -1.0, x * v, y * v, v])

    A = np.array(A)
    # Singular Value Decomposition: Ah = 0, solution is last column of V (last row of Vh)
    _, _, Vh = np.linalg.svd(A)
    H = Vh[-1].reshape(3, 3)

    # Normalize by H[2,2]
    if abs(H[2, 2]) > 1e-9:
        H = H / H[2, 2]
    return H


def compute_homography(src_points: np.ndarray, dst_points: np.ndarray) -> np.ndarray:
    """
    Computes 3x3 homography matrix mapping source points (pixels) to destination points (pitch meters).
    Uses OpenCV cv2.findHomography if available, with pure NumPy DLT as guaranteed fallback.
    """
    try:
        import cv2
        src = np.asarray(src_points, dtype=np.float32)
        dst = np.asarray(dst_points, dtype=np.float32)
        H, _ = cv2.findHomography(src, dst, cv2.RANSAC, 5.0)
        if H is not None:
            if abs(H[2, 2]) > 1e-9:
                H = H / H[2, 2]
            return H
    except Exception:
        pass

    return compute_homography_dlt(src_points, dst_points)


def project_pixel_to_pitch(
    H: np.ndarray,
    px_x: float,
    px_y: float,
    clip: bool = True,
) -> Tuple[float, float]:
    """
    Transforms a single pixel coordinate (px_x, px_y) to pitch plane coordinates (X_m, Y_m).
    """
    pt = np.array([px_x, px_y, 1.0], dtype=np.float64)
    res = np.dot(H, pt)

    w = res[2]
    if abs(w) < 1e-9:
        w = 1e-9

    pitch_x = float(res[0] / w)
    pitch_y = float(res[1] / w)

    if clip:
        pitch_x = float(np.clip(pitch_x, 0.0, PITCH_LENGTH_METERS))
        pitch_y = float(np.clip(pitch_y, 0.0, PITCH_WIDTH_METERS))

    return round(pitch_x, 2), round(pitch_y, 2)


def project_bbox_to_pitch(
    H: np.ndarray,
    bbox: List[float],
    clip: bool = True,
) -> Tuple[float, float]:
    """
    Transforms bounding box [x1, y1, x2, y2] to pitch plane coordinates
    using the bottom center point (feet contact point on the turf).
    """
    x1, y1, x2, y2 = bbox[:4]
    feet_x = (x1 + x2) / 2.0
    feet_y = y2
    return project_pixel_to_pitch(H, feet_x, feet_y, clip=clip)

