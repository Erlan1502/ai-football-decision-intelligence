from typing import List, Optional
from pydantic import BaseModel, Field


class TacticalAnalogItem(BaseModel):
    id: int
    name: str
    club: str
    league: str
    primary_position: str
    age: int
    market_value_eur: float
    predicted_value_eur: float
    undervaluation_delta_eur: float
    similarity_score_pct: float = Field(..., ge=0.0, le=100.0)
    cosine_distance: float


class TacticalAnalogsResponse(BaseModel):
    target_player_id: int
    target_player_name: str
    target_position: str
    target_market_value_eur: float
    total_analogs_found: int
    analogs: List[TacticalAnalogItem]

