import os
import json
import pytest
import numpy as np
from fastapi.testclient import TestClient

from api.main import app
from core_ml.data_loader import load_raw_dataset
from core_ml.explainability import MarketValueExplainer
from core_ml.squad_optimizer import SquadOptimizer
from cv_module.homography import compute_homography, project_pixel_to_pitch
from cv_module.pipeline import ScoutCVPipeline
from config.settings import settings


client = TestClient(app)


def test_e2e_full_system_lifecycle():
    """
    End-to-End System Integration Test (Football Moneyball 2.0).
    Validates complete pipeline from raw data to ML, CV, Agent Dossier, and Squad Optimization.
    """
    # 1. Dataset Integrity
    players = load_raw_dataset()
    assert len(players) >= 500, f"Expected 500+ players, got {len(players)}"
    
    # Verify player structure and 64-d tactical vector
    rodri = next((p for p in players if p["name"] == "Rodri"), None)
    assert rodri is not None, "Benchmark star 'Rodri' must exist in dataset"
    emb = rodri.get("embedding") or rodri.get("tactical_embedding")
    assert emb is not None, "Embedding vector missing for benchmark star"
    assert len(emb) == 64
    assert np.isclose(np.linalg.norm(emb), 1.0, atol=1e-3)

    # 2. Tactical Analog Search via API
    resp_analogs = client.get(
        f"/api/v1/scouting/analogs/{rodri['id']}",
        params={"top_k": 5, "max_market_value": 50_000_000.0}
    )
    assert resp_analogs.status_code == 200, resp_analogs.text
    analogs_data = resp_analogs.json()
    assert analogs_data["target_player_name"] == "Rodri"
    assert len(analogs_data["analogs"]) > 0

    top_clone = analogs_data["analogs"][0]
    assert top_clone["similarity_score_pct"] >= 75.0, "Tactical clone similarity must be >= 75%"
    assert top_clone["market_value_eur"] <= 50_000_000.0, "Clone must respect budget ceiling"
    savings = rodri["market_value_eur"] - top_clone["market_value_eur"]
    assert savings >= 40_000_000.0, f"Expected €40M+ arbitrage savings, got €{savings/1e6:.1f}M"

    # 3. SHAP Feature Attribution
    clone_full = next(p for p in players if p["id"] == top_clone["id"])
    explainer = MarketValueExplainer()
    shap_res = explainer.explain_player(clone_full)
    assert "top_positive_drivers" in shap_res
    assert len(shap_res["top_positive_drivers"]) > 0
    assert "impact_eur" in shap_res["top_positive_drivers"][0]

    # 4. Gemini 3.8 Flash Transfer Dossier Generation
    resp_report = client.post(
        "/api/v1/scouting/generate-dossier",
        json={
            "target_player_id": rodri["id"],
            "analog_player_id": top_clone["id"],
            "custom_tactical_instructions": "Focus on high press counter-pressing and progressive passing volume."
        }
    )
    assert resp_report.status_code == 200, resp_report.text
    dossier = resp_report.json()
    assert dossier["target_player_name"] == rodri["name"]
    assert dossier["recommended_analog_name"] == top_clone["name"]
    assert dossier["tactical_similarity_score"] >= 70.0
    assert dossier["verdict"] in ["STRONG_BUY", "BUY", "HOLD", "AVOID"]
    assert dossier["financial_arbitrage"]["expected_savings_eur"] > 0
    assert len(dossier["key_strengths"]) >= 1
    assert len(dossier["risks_and_mitigations"]) >= 1

    # 5. SciPy MILP Squad Optimization
    optimizer = SquadOptimizer(players)
    squad_res = optimizer.optimize_squad(
        budget_cap_eur=250_000_000.0,
        formation="4-3-3",
        max_per_club=3,
    )
    assert squad_res["success"] is True
    assert len(squad_res["selected_players"]) == 11
    assert squad_res["total_cost_eur"] <= 250_000_000.0
    assert squad_res["total_performance_score"] > 700.0

    # 6. CV Video-to-Data Homography & Tracking PoC
    sample_cv_path = os.path.join(settings.BASE_DIR, "cv_module", "samples", "scout_tracking_poc.json")
    assert os.path.exists(sample_cv_path)
    with open(sample_cv_path, "r") as f:
        tracking_poc = json.load(f)

    assert "frames" in tracking_poc
    assert len(tracking_poc["frames"]) >= 10
    first_frame = tracking_poc["frames"][0]
    assert len(first_frame["players"]) > 0
    first_player = first_frame["players"][0]
    px = first_player.get("pitch_x", first_player.get("pitch_pos_meters", [0, 0])[0])
    py = first_player.get("pitch_y", first_player.get("pitch_pos_meters", [0, 0])[1])
    assert 0.0 <= px <= 105.0, f"X coord {px} out of pitch bounds"
    assert 0.0 <= py <= 68.0, f"Y coord {py} out of pitch bounds"
    assert "speed_kmh" in first_player
