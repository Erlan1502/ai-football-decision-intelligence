import math
from typing import List, Dict, Any, Optional, Tuple
import numpy as np

def calculate_iou(boxA: List[float], boxB: List[float]) -> float:
    """Calculates Intersection over Union (IoU) between two bounding boxes [x1, y1, x2, y2]."""
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    interArea = max(0.0, xB - xA) * max(0.0, yB - yA)
    boxAArea = max(1.0, (boxA[2] - boxA[0]) * (boxA[3] - boxA[1]))
    boxBArea = max(1.0, (boxB[2] - boxB[0]) * (boxB[3] - boxB[1]))

    return interArea / float(boxAArea + boxBArea - interArea)


class TrackedEntity:
    def __init__(self, track_id: int, bbox: List[float], class_name: str, team: str, start_frame: int):
        self.track_id = track_id
        self.bbox = bbox
        self.class_name = class_name
        self.team = team
        self.last_frame = start_frame
        self.pitch_positions: List[Tuple[float, float, float]] = []  # (x_m, y_m, timestamp)
        self.speed_kmh: float = 0.0

    def update(self, bbox: List[float], frame_idx: int, pitch_coord: Tuple[float, float], timestamp: float):
        self.bbox = bbox
        self.last_frame = frame_idx
        self.pitch_positions.append((pitch_coord[0], pitch_coord[1], timestamp))

        # Calculate smoothed velocity over recent frames
        if len(self.pitch_positions) >= 3:
            prev_x, prev_y, prev_t = self.pitch_positions[-3]
            dt = timestamp - prev_t
            if dt > 0.01:
                dist_meters = math.hypot(pitch_coord[0] - prev_x, pitch_coord[1] - prev_y)
                raw_speed = (dist_meters / dt) * 3.6  # m/s to km/h
                # Smooth and bound within realistic human sprinting limit (up to 36 km/h)
                self.speed_kmh = round(float(np.clip(raw_speed, 0.0, 36.0)), 1)


class ScoutByteTracker:
    """
    ByteTrack-inspired multi-object tracker for football scouting.
    Maintains continuous track IDs, handles missed frames, and computes pitch velocities.
    """

    def __init__(self, max_lost_frames: int = 15, iou_threshold: float = 0.25):
        self.max_lost_frames = max_lost_frames
        self.iou_threshold = iou_threshold
        self.next_track_id = 1
        self.tracks: Dict[int, TrackedEntity] = {}

    def update(
        self,
        detections: List[Dict[str, Any]],
        frame_idx: int,
        timestamp_sec: float,
        H: np.ndarray,
    ) -> List[Dict[str, Any]]:
        """
        Updates tracks with detections from current frame and calculates pitch coordinates.
        """
        from cv_module.homography import project_bbox_to_pitch

        # Match detections to existing tracks
        active_track_ids = list(self.tracks.keys())
        unmatched_dets = []
        matched_tracks = set()

        # Simple greedy IoU association matching ByteTrack tier 1
        for det in detections:
            bbox = det["bbox"]
            best_iou = 0.0
            best_tid = None

            for tid in active_track_ids:
                if tid in matched_tracks:
                    continue
                iou = calculate_iou(bbox, self.tracks[tid].bbox)
                if iou > best_iou:
                    best_iou = iou
                    best_tid = tid

            if best_tid is not None and best_iou >= self.iou_threshold:
                matched_tracks.add(best_tid)
                pitch_pt = project_bbox_to_pitch(H, bbox)
                self.tracks[best_tid].update(bbox, frame_idx, pitch_pt, timestamp_sec)
            else:
                unmatched_dets.append(det)

        # Create new tracks for unmatched high-confidence detections
        for det in unmatched_dets:
            bbox = det["bbox"]
            cls_name = det.get("class_name", "player")
            pitch_pt = project_bbox_to_pitch(H, bbox)

            # Assign team based on initial field half
            team = "ball" if cls_name == "ball" else ("team_a" if pitch_pt[0] < 52.5 else "team_b")
            entity = TrackedEntity(self.next_track_id, bbox, cls_name, team, frame_idx)
            entity.update(bbox, frame_idx, pitch_pt, timestamp_sec)
            self.tracks[self.next_track_id] = entity
            self.next_track_id += 1

        # Prune old tracks exceeding max_lost_frames
        dead_tracks = [
            tid for tid, track in self.tracks.items()
            if (frame_idx - track.last_frame) > self.max_lost_frames
        ]
        for tid in dead_tracks:
            del self.tracks[tid]

        # Prepare frame output
        output_entities = []
        for tid, track in self.tracks.items():
            if track.last_frame == frame_idx:
                last_pitch = track.pitch_positions[-1]
                output_entities.append({
                    "track_id": track.track_id,
                    "class_name": track.class_name,
                    "team": track.team,
                    "bbox_px": track.bbox,
                    "pitch_x": last_pitch[0],
                    "pitch_y": last_pitch[1],
                    "speed_kmh": track.speed_kmh,
                })

        return output_entities

