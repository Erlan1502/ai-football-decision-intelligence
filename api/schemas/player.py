from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field


class PlayerStatsSchema(BaseModel):
    season: str = "2025/2026"
    goals_p90: float = 0.0
    assists_p90: float = 0.0
    npxg_p90: float = 0.0
    shots_total_p90: float = 0.0
    shots_on_target_pct: float = 0.0
    xag_p90: float = 0.0
    shot_creating_actions_p90: float = 0.0
    goal_creating_actions_p90: float = 0.0
    key_passes_p90: float = 0.0
    pass_completion_pct: float = 0.0
    passes_attempted_p90: float = 0.0
    prgp_p90: float = 0.0
    passes_into_final_third_p90: float = 0.0
    passes_into_penalty_area_p90: float = 0.0
    prgc_p90: float = 0.0
    dribbles_completed_p90: float = 0.0
    dribble_success_pct: float = 0.0
    carries_into_final_third_p90: float = 0.0
    dispossessed_p90: float = 0.0
    tackles_p90: float = 0.0
    tackles_won_p90: float = 0.0
    interceptions_p90: float = 0.0
    blocks_p90: float = 0.0
    clearances_p90: float = 0.0
    recoveries_p90: float = 0.0
    tackles_def_3rd_p90: float = 0.0
    tackles_mid_3rd_p90: float = 0.0
    tackles_att_3rd_p90: float = 0.0
    aerial_duels_won_pct: float = 0.0
    aerial_duels_total_p90: float = 0.0


class ShapDriverItem(BaseModel):
    feature: str
    label: str
    impact_eur: float
    direction: str  # "positive" | "negative"


class PlayerValuationSchema(BaseModel):
    predicted_value_eur: float
    actual_value_eur: float
    undervaluation_delta_eur: float
    undervaluation_ratio: float
    confidence_score: float = 0.85
    shap_drivers: Optional[List[ShapDriverItem]] = None


class PlayerResponse(BaseModel):
    id: int
    name: str
    age: int
    nationality: str
    club: str
    league: str
    primary_position: str
    secondary_position: Optional[str] = None
    market_value_eur: float
    contract_expires_year: int
    minutes_played: int
    matches_played: int
    performance_score: float = 65.0
    valuation: Optional[PlayerValuationSchema] = None
    stats: Optional[PlayerStatsSchema] = None


class PlayerFilterParams(BaseModel):
    position: Optional[str] = None
    league: Optional[str] = None
    max_market_value: Optional[float] = None
    min_delta: Optional[float] = None
    only_undervalued: bool = False
    limit: int = Field(default=50, ge=1, le=500)
    offset: int = Field(default=0, ge=0)
