from typing import Optional
from fastapi import APIRouter, HTTPException, Query
from api.schemas.analog import TacticalAnalogsResponse
from api.schemas.report import TransferDossierRequest, TransferDossierSchema
from api.services.vector_search import vector_search_service
from api.services.scout_agent import scout_agent
from core_ml.explainability import MarketValueExplainer
from core_ml.data_loader import prepare_tabular_data

router = APIRouter(prefix="/scouting", tags=["Scouting Intelligence"])

# Lazy-loaded SHAP explainer instance
_explainer = None


def get_explainer() -> MarketValueExplainer:
    global _explainer
    if _explainer is None:
        try:
            _explainer = MarketValueExplainer()
        except Exception:
            _explainer = None
    return _explainer


@router.get("/analogs/{target_player_id}", response_model=TacticalAnalogsResponse, summary="Find Tactical Analogs")
def get_tactical_analogs(
    target_player_id: int,
    top_k: int = Query(5, ge=1, le=20),
    same_position: bool = Query(True, description="Limit search strictly to target's position"),
    max_market_value: Optional[float] = Query(None, description="Max fee ceiling for analog search"),
    only_undervalued: bool = Query(False, description="Filter only players with positive Moneyball delta"),
):
    """
    Performs cosine vector search (matching pgvector HNSW index) to discover tactical clones.
    """
    try:
        return vector_search_service.find_tactical_analogs(
            target_player_id=target_player_id,
            top_k=top_k,
            same_position=same_position,
            max_market_value=max_market_value,
            only_undervalued=only_undervalued,
        )
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))


@router.post("/generate-dossier", response_model=TransferDossierSchema, summary="Generate AI Transfer Dossier")
def generate_transfer_dossier(req: TransferDossierRequest):
    """
    Generates an executive scouting transfer audit using Gemini 3.8 Flash (google-genai SDK).
    Includes SHAP XAI feature attributions and financial arbitrage analysis.
    """
    target = vector_search_service.get_player(req.target_player_id)
    if not target:
        raise HTTPException(status_code=404, detail=f"Target player ID {req.target_player_id} not found.")

    analog = vector_search_service.get_player(req.analog_player_id)
    if not analog:
        raise HTTPException(status_code=404, detail=f"Analog player ID {req.analog_player_id} not found.")

    # Calculate tactical similarity score
    analogs_res = vector_search_service.find_tactical_analogs(
        target_player_id=req.target_player_id,
        top_k=50,
        same_position=False,
    )
    sim_match = next((a for a in analogs_res.analogs if a.id == req.analog_player_id), None)
    similarity_score = sim_match.similarity_score_pct if sim_match else 85.0

    # Extract SHAP drivers for analog player
    shap_data = None
    explainer = get_explainer()
    if explainer is not None:
        try:
            X, _, _, df_full = prepare_tabular_data()
            matched_row = X[df_full["id"] == req.analog_player_id]
            if len(matched_row) > 0:
                shap_data = explainer.explain_player(matched_row.iloc[0], top_k=4)
        except Exception:
            shap_data = None

    # Generate dossier via Gemini 3.8 Flash (or authoritative fallback mock)
    dossier = scout_agent.generate_dossier(
        target_player=target,
        analog_player=analog,
        similarity_score_pct=similarity_score,
        shap_drivers=shap_data,
        custom_instructions=getattr(req, "custom_tactical_instructions", None),
    )

    return dossier
