from fastapi import APIRouter, HTTPException
from api.schemas.squad import SquadOptimizationRequest, SquadOptimizationResponse, SelectedSquadPlayer
from core_ml.data_loader import load_raw_dataset
from core_ml.squad_optimizer import SquadOptimizer

router = APIRouter(prefix="/squad", tags=["Squad Builder & Linear Programming"])

_optimizer = None


def get_optimizer() -> SquadOptimizer:
    global _optimizer
    if _optimizer is None:
        players = load_raw_dataset()
        _optimizer = SquadOptimizer(players)
    return _optimizer


@router.post("/optimize", response_model=SquadOptimizationResponse, summary="Optimize 11-Player Squad (Scipy MILP)")
def optimize_squad(req: SquadOptimizationRequest):
    """
    Executes Integer Linear Programming (0-1 Knapsack) to build the optimal 11-player lineup
    maximizing performance score and Moneyball undervaluation delta under the specified budget ceiling.
    """
    opt = get_optimizer()
    try:
        res = opt.optimize_squad(
            budget_cap_eur=req.budget_cap_eur,
            formation=req.formation,
            max_per_club=req.max_per_club,
            moneyball_weight=req.moneyball_weight,
            min_delta_filter=req.min_delta_filter,
        )
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"MILP solver error: {str(e)}")

    if not res.get("success", False):
        raise HTTPException(
            status_code=422,
            detail=res.get("message", "Could not satisfy budget and formation constraints with current player pool."),
        )

    squad_players = [
        SelectedSquadPlayer(
            id=p["id"],
            name=p["name"],
            club=p["club"],
            league=p["league"],
            primary_position=p["primary_position"],
            age=p["age"],
            market_value_eur=p["market_value_eur"],
            performance_score=p["performance_score"],
            predicted_value_eur=p["predicted_value_eur"],
            undervaluation_delta_eur=p["undervaluation_delta_eur"],
        )
        for p in res["squad"]
    ]

    return SquadOptimizationResponse(
        success=True,
        formation=res["formation"],
        budget_cap_eur=res["budget_cap_eur"],
        total_spent_eur=res["total_spent_eur"],
        budget_remaining_eur=res["budget_remaining_eur"],
        total_undervaluation_delta_eur=res["total_undervaluation_delta_eur"],
        avg_performance_score=res["avg_performance_score"],
        total_players=res["total_players"],
        solve_time_sec=res["solve_time_sec"],
        squad=squad_players,
    )
