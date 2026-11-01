from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field


class SquadOptimizationRequest(BaseModel):
    budget_cap_eur: float = Field(default=120_000_000.0, ge=15_000_000.0, le=800_000_000.0)
    formation: str = Field(default="4-3-3", description="4-3-3 | 4-2-3-1 | 3-5-2")
    max_per_club: int = Field(default=3, ge=1, le=11)
    moneyball_weight: float = Field(default=0.35, ge=0.0, le=1.0)
    min_delta_filter: Optional[float] = None


class SelectedSquadPlayer(BaseModel):
    id: int
    name: str
    club: str
    league: str
    primary_position: str
    age: int
    market_value_eur: float
    performance_score: float
    predicted_value_eur: float
    undervaluation_delta_eur: float


class SquadOptimizationResponse(BaseModel):
    success: bool
    formation: str
    budget_cap_eur: float
    total_spent_eur: float
    budget_remaining_eur: float
    total_undervaluation_delta_eur: float
    avg_performance_score: float
    total_players: int
    solve_time_sec: float
    squad: List[SelectedSquadPlayer]
