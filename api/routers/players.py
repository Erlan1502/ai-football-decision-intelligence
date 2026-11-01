from typing import List, Optional
from fastapi import APIRouter, HTTPException, Query
from api.schemas.player import PlayerResponse, PlayerStatsSchema, PlayerValuationSchema, ShapDriverItem
from api.services.vector_search import vector_search_service

router = APIRouter(prefix="/players", tags=["Players"])


def _format_player_response(p: dict) -> PlayerResponse:
    stats = p.get("stats")
    stats_schema = PlayerStatsSchema(**stats) if stats else None

    val = p.get("valuation")
    val_schema = None
    if val:
        shap_items = None
        if "shap_drivers" in val and val["shap_drivers"]:
            shap_items = [
                ShapDriverItem(
                    feature=d.get("feature", "stat"),
                    label=d.get("label", d.get("feature", "stat")),
                    impact_eur=float(d.get("impact_eur", 0.0)),
                    direction=d.get("direction", "positive"),
                )
                for d in val["shap_drivers"]
            ]
        val_schema = PlayerValuationSchema(
            predicted_value_eur=float(val.get("predicted_value_eur", p["market_value_eur"])),
            actual_value_eur=float(val.get("actual_value_eur", p["market_value_eur"])),
            undervaluation_delta_eur=float(val.get("undervaluation_delta_eur", 0.0)),
            undervaluation_ratio=float(val.get("undervaluation_ratio", 1.0)),
            confidence_score=float(val.get("confidence_score", 0.85)),
            shap_drivers=shap_items,
        )

    return PlayerResponse(
        id=p["id"],
        name=p["name"],
        age=p["age"],
        nationality=p["nationality"],
        club=p["club"],
        league=p["league"],
        primary_position=p["primary_position"],
        secondary_position=p.get("secondary_position"),
        market_value_eur=p["market_value_eur"],
        contract_expires_year=p.get("contract_expires_year", 2028),
        minutes_played=p.get("minutes_played", 0),
        matches_played=p.get("matches_played", 0),
        performance_score=p.get("performance_score", 65.0),
        stats=stats_schema,
        valuation=val_schema,
    )


@router.get("", response_model=List[PlayerResponse], summary="List & Filter Players")
def get_players(
    position: Optional[str] = Query(None, description="CB, FB, CM, CAM, W, ST"),
    league: Optional[str] = Query(None, description="Filter by league name"),
    max_market_value: Optional[float] = Query(None, description="Max market value in EUR"),
    min_delta: Optional[float] = Query(None, description="Minimum Moneyball undervaluation delta in EUR"),
    only_undervalued: bool = Query(False, description="Filter only players where predicted > actual value"),
    limit: int = Query(50, ge=1, le=500),
    offset: int = Query(0, ge=0),
):
    """Retrieves player profiles with p90 statistics and ML valuations."""
    players = vector_search_service.search_players(
        position=position,
        league=league,
        max_market_value=max_market_value,
        min_delta=min_delta,
        only_undervalued=only_undervalued,
        limit=limit,
        offset=offset,
    )
    return [_format_player_response(p) for p in players]


@router.get("/{player_id}", response_model=PlayerResponse, summary="Get Player Profile")
def get_player_by_id(player_id: int):
    """Retrieves a single player's complete profile by ID."""
    player = vector_search_service.get_player(player_id)
    if not player:
        raise HTTPException(status_code=404, detail=f"Player with ID {player_id} not found.")
    return _format_player_response(player)

