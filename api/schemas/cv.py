from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field


class PitchDimensions(BaseModel):
    length: float = 105.0
    width: float = 68.0


class PlayerTrackItem(BaseModel):
    track_id: int
    team: str
    pitch_x: float
    pitch_y: float
    speed_kmh: float
    bbox_px: List[float]


class BallTrackItem(BaseModel):
    pitch_x: float
    pitch_y: float
    speed_kmh: float


class FrameTrackingData(BaseModel):
    frame_idx: int
    timestamp_sec: float
    players_count: int
    players: List[PlayerTrackItem]
    ball: BallTrackItem


class AggregatedCVMetrics(BaseModel):
    total_distance_covered_km: float
    high_speed_runs_count: int
    heat_zones_matrix_grid: List[List[float]]


class TrackingEventResponse(BaseModel):
    match_id: str
    pitch_dimensions_meters: PitchDimensions
    fps: float
    total_frames_processed: int
    processing_time_sec: float
    frames: List[FrameTrackingData]
    aggregated_metrics: AggregatedCVMetrics


class CVProcessRequest(BaseModel):
    video_path: Optional[str] = None
    max_frames: int = Field(default=50, ge=10, le=300)
    fps: float = Field(default=25.0, ge=10.0, le=60.0)
