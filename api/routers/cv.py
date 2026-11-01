import os
import json
from fastapi import APIRouter, HTTPException
from api.schemas.cv import CVProcessRequest, TrackingEventResponse
from cv_module.pipeline import ScoutCVPipeline, SAMPLE_OUTPUT_JSON

router = APIRouter(prefix="/cv", tags=["Computer Vision & Video-to-Data"])

_pipeline = None


def get_cv_pipeline() -> ScoutCVPipeline:
    global _pipeline
    if _pipeline is None:
        _pipeline = ScoutCVPipeline()
    return _pipeline


@router.post("/process-clip", response_model=TrackingEventResponse, summary="Process Scouting Video Clip")
def process_scout_clip(req: CVProcessRequest):
    """
    Executes the Scout Video-to-Data pipeline:
    YOLOv11 Detection -> ByteTrack Association -> 2D Pitch Homography -> Standardized Tracking JSON.
    """
    pipeline = get_cv_pipeline()
    try:
        result = pipeline.process_clip(
            video_path=req.video_path,
            max_frames=req.max_frames,
            fps=req.fps,
        )
        return TrackingEventResponse(**result)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"CV pipeline processing error: {str(e)}")


@router.get("/sample-tracking", response_model=TrackingEventResponse, summary="Get Pre-computed Sample Tracking Data")
def get_sample_tracking():
    """Retrieves the calibrated 105x68m sample tracking dataset for UI visualization."""
    if not os.path.exists(SAMPLE_OUTPUT_JSON):
        pipeline = get_cv_pipeline()
        res = pipeline.process_clip(max_frames=50)
        pipeline.save_json(res, SAMPLE_OUTPUT_JSON)
        return TrackingEventResponse(**res)

    with open(SAMPLE_OUTPUT_JSON, "r", encoding="utf-8") as f:
        data = json.load(f)
    return TrackingEventResponse(**data)
