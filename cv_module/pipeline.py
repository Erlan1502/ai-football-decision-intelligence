import os
import json
import time
from typing import Dict, Any, List, Optional, Tuple
import numpy as np

from cv_module.homography import compute_homography, get_canonical_pitch_corners, PITCH_LENGTH_METERS, PITCH_WIDTH_METERS
from cv_module.detector import ScoutObjectDetector
from cv_module.tracker import ScoutByteTracker

SAMPLE_OUTPUT_JSON = os.path.join(os.path.dirname(__file__), "samples", "scout_tracking_poc.json")


class ScoutCVPipeline:
    """
    End-to-End Scout Video-to-Data Pipeline.
    YOLOv11 Detection -> ByteTrack Association -> 2D Pitch Homography -> Standardized JSON Export.
    """

    def __init__(
        self,
        model_name: str = "yolo11n.pt",
        src_corners: Optional[np.ndarray] = None,
        dst_corners: Optional[np.ndarray] = None,
    ):
        self.detector = ScoutObjectDetector(model_name=model_name)
        self.tracker = ScoutByteTracker()

        # Calibration setup (Default tactical camera viewing 1920x1080)
        if src_corners is None:
            # 4 corners of the pitch in broadcast view pixels
            self.src_corners = np.array([
                [240.0, 160.0],    # Top-Left corner (0m, 0m)
                [1680.0, 160.0],   # Top-Right corner (105m, 0m)
                [1860.0, 960.0],   # Bottom-Right corner (105m, 68m)
                [60.0, 960.0],     # Bottom-Left corner (0m, 68m)
            ], dtype=np.float32)
        else:
            self.src_corners = np.asarray(src_corners, dtype=np.float32)

        if dst_corners is None:
            self.dst_corners = get_canonical_pitch_corners()
        else:
            self.dst_corners = np.asarray(dst_corners, dtype=np.float32)

        self.H = compute_homography(self.src_corners, self.dst_corners)

    def process_clip(
        self,
        video_path: Optional[str] = None,
        max_frames: int = 75,  # 3 seconds @ 25fps for fast PoC test
        fps: float = 25.0,
    ) -> Dict[str, Any]:
        """
        Processes a video clip or synthetic tactical camera stream.
        Outputs a standardized tracking JSON artifact.
        """
        t0 = time.time()
        frames_data = []

        # Heat zone grid: 10 zones along length (10.5m each), 7 zones along width (~9.7m each)
        heat_grid = np.zeros((10, 7), dtype=int)
        total_distance_m = 0.0
        high_speed_runs = 0

        has_cv2_video = False
        cap = None
        if video_path and os.path.exists(video_path):
            try:
                import cv2
                cap = cv2.VideoCapture(video_path)
                if cap.isOpened():
                    has_cv2_video = True
            except Exception:
                has_cv2_video = False

        for f_idx in range(max_frames):
            timestamp_sec = round(f_idx / fps, 2)

            if has_cv2_video:
                ret, frame = cap.read()
                if not ret:
                    break
                detections = self.detector.detect(frame)
            else:
                # High-fidelity tactical frame generator
                detections = self.detector.generate_synthetic_detections(
                    frame_idx=f_idx, frame_shape=(1080, 1920), num_players=20
                )

            # ByteTrack updates and transforms coordinates via Homography
            tracked_entities = self.tracker.update(detections, f_idx, timestamp_sec, self.H)

            players_frame = []
            ball_frame = None

            for ent in tracked_entities:
                px = ent["pitch_x"]
                py = ent["pitch_y"]
                speed = ent["speed_kmh"]

                # Accumulate heat zones
                grid_x = min(9, int(px / (PITCH_LENGTH_METERS / 10.0)))
                grid_y = min(6, int(py / (PITCH_WIDTH_METERS / 7.0)))
                heat_grid[grid_x, grid_y] += 1

                if speed >= 21.0:
                    high_speed_runs += 1

                if ent["class_name"] == "ball":
                    ball_frame = {
                        "pitch_x": px,
                        "pitch_y": py,
                        "speed_kmh": speed,
                    }
                else:
                    players_frame.append({
                        "track_id": ent["track_id"],
                        "team": ent["team"],
                        "pitch_x": px,
                        "pitch_y": py,
                        "speed_kmh": speed,
                        "bbox_px": ent["bbox_px"],
                    })

            frames_data.append({
                "frame_idx": f_idx,
                "timestamp_sec": timestamp_sec,
                "players_count": len(players_frame),
                "players": players_frame,
                "ball": ball_frame or {"pitch_x": 52.5, "pitch_y": 34.0, "speed_kmh": 0.0},
            })

        if cap is not None:
            cap.release()

        # Compute total distance covered by players across the sequence
        for track in self.tracker.tracks.values():
            if len(track.pitch_positions) >= 2:
                pts = np.array([(p[0], p[1]) for p in track.pitch_positions])
                diffs = np.diff(pts, axis=0)
                dists = np.hypot(diffs[:, 0], diffs[:, 1])
                total_distance_m += float(np.sum(dists))

        # Normalize heat matrix to percentages
        total_zone_pts = max(1, np.sum(heat_grid))
        norm_heat_grid = (heat_grid / total_zone_pts * 100.0).round(1).tolist()

        result = {
            "match_id": "clip_scout_poc_001",
            "pitch_dimensions_meters": {
                "length": PITCH_LENGTH_METERS,
                "width": PITCH_WIDTH_METERS,
            },
            "fps": fps,
            "total_frames_processed": len(frames_data),
            "processing_time_sec": round(time.time() - t0, 3),
            "frames": frames_data,
            "aggregated_metrics": {
                "total_distance_covered_km": round(total_distance_m / 1000.0, 3),
                "high_speed_runs_count": high_speed_runs,
                "heat_zones_matrix_grid": norm_heat_grid,
            },
        }

        return result

    @staticmethod
    def save_json(data: Dict[str, Any], filepath: Optional[str] = None):
        target_path = filepath or SAMPLE_OUTPUT_JSON
        os.makedirs(os.path.dirname(target_path), exist_ok=True)
        with open(target_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
        print(f"[CV Pipeline] Tracking event JSON saved to {target_path}")


if __name__ == "__main__":
    pipeline = ScoutCVPipeline()
    result = pipeline.process_clip(max_frames=75)
    pipeline.save_json(result)
