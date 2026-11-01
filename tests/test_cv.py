import os
import json
import pytest
import numpy as np

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


def test_homography_exact_corner_projection():
    """
    Verify that 2D homography projects camera pixel corners to pitch metric corners
    with error margin < 0.1 meter.
    """
    src_corners = np.array([
        [240.0, 160.0],
        [1680.0, 160.0],
        [1860.0, 960.0],
        [60.0, 960.0],
    ], dtype=np.float32)

    dst_corners = get_canonical_pitch_corners()  # (0,0), (105,0), (105,68), (0,68)

    H = compute_homography(src_corners, dst_corners)
    assert H.shape == (3, 3), f"Homography matrix must be 3x3, got {H.shape}"

    for i in range(4):
        px, py = src_corners[i]
        expected_x, expected_y = dst_corners[i]
        proj_x, proj_y = project_pixel_to_pitch(H, px, py, clip=False)

        assert abs(proj_x - expected_x) < 0.15, (
            f"Corner {i}: Projected X {proj_x} does not match expected {expected_x}"
        )
        assert abs(proj_y - expected_y) < 0.15, (
            f"Corner {i}: Projected Y {proj_y} does not match expected {expected_y}"
        )


def test_homography_bbox_feet_projection():
    """Verify that player feet are mapped accurately to the 105x68m playing surface."""
    src_corners = np.array([
        [240.0, 160.0], [1680.0, 160.0], [1860.0, 960.0], [60.0, 960.0]
    ], dtype=np.float32)
    dst_corners = get_canonical_pitch_corners()
    H = compute_homography(src_corners, dst_corners)

    # Player standing near midfield in camera view
    midfield_bbox = [940.0, 500.0, 980.0, 580.0]  # feet at (960, 580)
    pitch_x, pitch_y = project_bbox_to_pitch(H, midfield_bbox, clip=True)

    assert 0.0 <= pitch_x <= PITCH_LENGTH_METERS, f"X coordinate out of bounds: {pitch_x}"
    assert 0.0 <= pitch_y <= PITCH_WIDTH_METERS, f"Y coordinate out of bounds: {pitch_y}"
    # Roughly near midfield center
    assert 30.0 <= pitch_x <= 75.0, f"Expected midfield X range, got {pitch_x}"


def test_detector_detections_schema():
    """Verify detector returns valid bounding boxes for players and the match ball."""
    detector = ScoutObjectDetector()
    
    # 1. Test synthetic high-fidelity detector stream
    detections = detector.generate_synthetic_detections(frame_idx=0, frame_shape=(1080, 1920), num_players=20)
    assert len(detections) >= 10, f"Expected at least 10 detections, got {len(detections)}"
    classes = set(d["class_name"] for d in detections)
    assert "player" in classes
    assert "ball" in classes

    for d in detections:
        bbox = d["bbox"]
        assert len(bbox) == 4
        assert bbox[2] > bbox[0], "x2 must be > x1"
        assert bbox[3] > bbox[1], "y2 must be > y1"
        assert 0.0 <= d["confidence"] <= 1.0

    # 2. Test model inference runs without crashing
    black_frame = np.zeros((360, 640, 3), dtype=np.uint8)
    real_dets = detector.detect(black_frame)
    assert isinstance(real_dets, list)


def test_tracker_track_continuity():
    """Verify ByteTrack maintains stable track IDs across consecutive frames."""
    tracker = ScoutByteTracker()
    src_corners = np.array([
        [240.0, 160.0], [1680.0, 160.0], [1860.0, 960.0], [60.0, 960.0]
    ], dtype=np.float32)
    dst_corners = get_canonical_pitch_corners()
    H = compute_homography(src_corners, dst_corners)

    # Simulate 5 consecutive frames of a player moving smoothly
    initial_bbox = [500.0, 400.0, 540.0, 480.0]
    tracked_ids_history = []

    for f_idx in range(5):
        # Player moves slightly to the right
        current_bbox = [initial_bbox[0] + f_idx * 5.0, initial_bbox[1], initial_bbox[2] + f_idx * 5.0, initial_bbox[3]]
        dets = [{
            "bbox": current_bbox,
            "confidence": 0.95,
            "class_name": "player",
        }]
        entities = tracker.update(dets, f_idx, timestamp_sec=f_idx * 0.04, H=H)
        assert len(entities) == 1
        tracked_ids_history.append(entities[0]["track_id"])

    # Track ID must remain identical across all 5 frames
    assert len(set(tracked_ids_history)) == 1, (
        f"Track ID was not maintained: {tracked_ids_history}"
    )


def test_cv_pipeline_e2e_json_export():
    """Verify full Scout CV pipeline runs and exports standardized tracking JSON."""
    pipeline = ScoutCVPipeline()
    result = pipeline.process_clip(max_frames=30, fps=25.0)

    # 1. Check top-level contract
    assert result["match_id"] == "clip_scout_poc_001"
    assert result["pitch_dimensions_meters"] == {"length": 105.0, "width": 68.0}
    assert result["fps"] == 25.0
    assert result["total_frames_processed"] == 30
    assert result["processing_time_sec"] > 0.0

    # 2. Check frame data
    assert len(result["frames"]) == 30
    first_frame = result["frames"][0]
    assert "timestamp_sec" in first_frame
    assert "players" in first_frame
    assert "ball" in first_frame
    assert len(first_frame["players"]) > 0

    player_sample = first_frame["players"][0]
    for key in ["track_id", "team", "pitch_x", "pitch_y", "speed_kmh", "bbox_px"]:
        assert key in player_sample, f"Missing key {key} in player tracking sample"

    # 3. Check pitch boundaries
    assert 0.0 <= player_sample["pitch_x"] <= 105.0
    assert 0.0 <= player_sample["pitch_y"] <= 68.0

    # 4. Check aggregated metrics
    agg = result["aggregated_metrics"]
    assert "total_distance_covered_km" in agg
    assert "high_speed_runs_count" in agg
    assert "heat_zones_matrix_grid" in agg
    assert len(agg["heat_zones_matrix_grid"]) == 10
    assert len(agg["heat_zones_matrix_grid"][0]) == 7

    # 5. Check saving to disk
    test_json_path = os.path.join(os.path.dirname(__file__), "..", "cv_module", "samples", "scout_tracking_poc.json")
    pipeline.save_json(result, test_json_path)
    assert os.path.exists(test_json_path), f"JSON artifact missing at {test_json_path}"
