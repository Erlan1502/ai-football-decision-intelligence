import pytest
from fastapi.testclient import TestClient
from api.main import app
from api.schemas.report import TransferDossierSchema
from api.schemas.squad import SquadOptimizationResponse
from api.schemas.cv import TrackingEventResponse


@pytest.fixture(scope="session")
def client():
    with TestClient(app) as c:
        yield c


def test_healthcheck_endpoint(client):
    """Verify system health endpoint returns 200 and model metadata."""
    resp = client.get("/health")
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "ok"
    assert data["version"] == "2.0.0"
    assert "gemini-3.8-flash" in data["llm_model"]


def test_players_listing_and_filtering(client):
    """Verify player filtering by position and Moneyball undervaluation flag."""
    resp = client.get("/api/v1/players?limit=10")
    assert resp.status_code == 200
    players = resp.json()
    assert len(players) == 10
    sample = players[0]
    assert "name" in sample
    assert "market_value_eur" in sample
    assert "stats" in sample

    # Test filtering by CM position and only undervalued
    filtered_resp = client.get("/api/v1/players?position=CM&only_undervalued=true&limit=5")
    assert filtered_resp.status_code == 200
    filtered = filtered_resp.json()
    assert len(filtered) > 0
    for p in filtered:
        assert p["primary_position"] == "CM"
        assert p["valuation"]["undervaluation_delta_eur"] > 0


def test_get_single_player_and_404(client):
    """Verify retrieving existing player and 404 for invalid ID."""
    resp = client.get("/api/v1/players/1")
    assert resp.status_code == 200
    p = resp.json()
    assert p["id"] == 1
    assert p["name"] == "Rodri"

    resp_404 = client.get("/api/v1/players/999999")
    assert resp_404.status_code == 404


def test_tactical_analogs_search_endpoint(client):
    """Verify vector analog endpoint discovers tactical twins for Rodri (CM)."""
    resp = client.get("/api/v1/scouting/analogs/1?top_k=5&same_position=true")
    assert resp.status_code == 200
    data = resp.json()
    assert data["target_player_id"] == 1
    assert data["target_player_name"] == "Rodri"
    assert len(data["analogs"]) == 5

    analog_names = [a["name"] for a in data["analogs"]]
    assert any(name in ["Marten Hjulmand", "Alan Varela"] for name in analog_names)
    assert data["analogs"][0]["similarity_score_pct"] >= 80.0


def test_scout_agent_transfer_dossier_generation(client):
    """
    Verify Gemini 3.8 Flash agent generates structured transfer audit
    with financial arbitrage, SHAP strengths and risk mitigation.
    """
    # Compare Rodri (ID 1) with Marten Hjulmand (find Hjulmand ID)
    players_resp = client.get("/api/v1/players?limit=50")
    hjulmand = next((p for p in players_resp.json() if p["name"] == "Marten Hjulmand"), None)
    assert hjulmand is not None, "Marten Hjulmand not found in seed dataset"

    payload = {
        "target_player_id": 1,  # Rodri (€110M)
        "analog_player_id": hjulmand["id"],  # Hjulmand (€35M)
        "custom_tactical_instructions": "Prioritize transition defense and ball retention in UEFA Champions League.",
    }

    resp = client.post("/api/v1/scouting/generate-dossier", json=payload)
    assert resp.status_code == 200

    dossier = TransferDossierSchema.model_validate(resp.json())
    assert dossier.target_player_name == "Rodri"
    assert dossier.recommended_analog_name == "Marten Hjulmand"
    assert dossier.tactical_similarity_score >= 80.0

    # Financial Arbitrage verification
    arb = dossier.financial_arbitrage
    assert arb.expected_savings_eur > 40_000_000.0  # €110M - €35M = €75M savings!
    assert arb.projected_resale_profit_eur >= 0.0

    # Key strengths with SHAP justification
    assert len(dossier.key_strengths) >= 2
    assert "Progressive" in dossier.key_strengths[0].metric or "Passing" in dossier.key_strengths[0].metric or "Recoveries" in dossier.key_strengths[0].metric

    # Risks and verdict
    assert len(dossier.risks_and_mitigations) >= 2
    assert dossier.verdict in ["STRONG_BUY", "BUY"]


def test_squad_optimization_endpoint(client):
    """Verify Scipy MILP squad optimization endpoint."""
    payload = {
        "budget_cap_eur": 120_000_000.0,
        "formation": "4-3-3",
        "max_per_club": 3,
        "moneyball_weight": 0.40,
    }
    resp = client.post("/api/v1/squad/optimize", json=payload)
    assert resp.status_code == 200

    squad_res = SquadOptimizationResponse.model_validate(resp.json())
    assert squad_res.success is True
    assert squad_res.total_players == 11
    assert squad_res.total_spent_eur <= 120_000_000.0
    assert squad_res.solve_time_sec < 1.0


def test_cv_sample_tracking_endpoint(client):
    """Verify CV video-to-data sample tracking JSON retrieval."""
    resp = client.get("/api/v1/cv/sample-tracking")
    assert resp.status_code == 200

    cv_data = TrackingEventResponse.model_validate(resp.json())
    assert cv_data.pitch_dimensions_meters.length == 105.0
    assert cv_data.pitch_dimensions_meters.width == 68.0
    assert len(cv_data.frames) > 0
    assert len(cv_data.aggregated_metrics.heat_zones_matrix_grid) == 10

