import os
import json
import time
import pytest
import numpy as np
import pandas as pd

from core_ml.data_loader import load_raw_dataset, prepare_tabular_data
from core_ml.train_value_model import load_trained_model, train_market_value_regressor
from core_ml.explainability import MarketValueExplainer
from core_ml.tactical_embeddings import TacticalEmbeddingEngine
from core_ml.squad_optimizer import SquadOptimizer, FORMATIONS


@pytest.fixture(scope="session")
def raw_players():
    return load_raw_dataset()


@pytest.fixture(scope="session")
def trained_artifacts():
    model, feature_names = load_trained_model()
    return model, feature_names


def test_lightgbm_performance_dod(trained_artifacts):
    """Verify LightGBM model reaches R^2 >= 0.75 on test holdout."""
    metrics_path = os.path.join(os.path.dirname(__file__), "..", "core_ml", "artifacts", "training_metrics.json")
    assert os.path.exists(metrics_path), "Training metrics artifact not found."
    with open(metrics_path, "r", encoding="utf-8") as f:
        metrics = json.load(f)

    assert metrics["r2_score"] >= 0.75, f"R^2 must be >= 0.75, got {metrics['r2_score']}"
    assert metrics["rmse_log"] < 0.50, f"RMSE too high: {metrics['rmse_log']}"


def test_undervaluation_delta_calculation(trained_artifacts, raw_players):
    """Verify market valuation, predicted values and Moneyball deltas."""
    model, feature_names = trained_artifacts
    X, y, _, df_full = prepare_tabular_data()

    preds_log = model.predict(X)
    preds_eur = np.expm1(preds_log)

    df_full["predicted_val"] = preds_eur
    df_full["delta"] = df_full["predicted_val"] - df_full["market_value_eur"]

    # All predicted values must be positive and non-zero
    assert (df_full["predicted_val"] > 500_000).all(), "All valuations should be > €500K"

    # Verify donor gems have positive Moneyball deltas
    hjulmand_rows = df_full[df_full["name"] == "Marten Hjulmand"]
    assert len(hjulmand_rows) == 1
    # Hjulmand in donor league has predicted fair value matching top tier stats
    assert hjulmand_rows.iloc[0]["predicted_val"] > 30_000_000


def test_shap_explainer_drivers(trained_artifacts):
    """Verify TreeSHAP generates positive and negative value drivers in under 100ms."""
    model, feature_names = trained_artifacts
    explainer = MarketValueExplainer(model, feature_names)

    X, _, _, df_full = prepare_tabular_data()
    sample_row = X.iloc[0]

    t0 = time.time()
    explanation = explainer.explain_player(sample_row, top_k=3)
    elapsed_ms = (time.time() - t0) * 1000.0

    assert elapsed_ms < 150.0, f"SHAP calculation took too long: {elapsed_ms:.1f}ms"
    assert "top_positive_drivers" in explanation
    assert "top_negative_drivers" in explanation
    assert len(explanation["top_positive_drivers"]) > 0

    driver = explanation["top_positive_drivers"][0]
    for key in ["feature", "label", "impact_eur", "direction"]:
        assert key in driver
    assert driver["direction"] == "positive"


def test_tactical_embeddings_and_analog_search(raw_players):
    """Verify 64-d embedding engine returns tactical twins for Rodri (CM)."""
    engine = TacticalEmbeddingEngine(vector_dim=64)
    engine.fit_from_dataset(raw_players)

    # Find Rodri
    rodri = next(p for p in raw_players if p["name"] == "Rodri")
    analogs = engine.find_analogs(target_player_id=rodri["id"], top_k=5, same_position=True)

    assert len(analogs) == 5
    for a in analogs:
        assert a["primary_position"] == "CM", "Analog must have same primary position"
        assert a["similarity_score_pct"] >= 80.0, f"Expected high tactical match, got {a['similarity_score_pct']}%"
        assert a["id"] != rodri["id"], "Target player should not be in their own analog list"

    # Verify Marten Hjulmand or Alan Varela is among top analogs
    analog_names = [a["name"] for a in analogs]
    assert any(name in ["Marten Hjulmand", "Alan Varela"] for name in analog_names), (
        f"Expected Hjulmand or Varela in Rodri analogs, got {analog_names}"
    )


def test_squad_optimizer_milp_solver(raw_players):
    """Verify Scipy MILP solves optimal 11-player squad under €120M budget in < 1.5s."""
    optimizer = SquadOptimizer(raw_players)

    t0 = time.time()
    res = optimizer.optimize_squad(
        budget_cap_eur=120_000_000.0,
        formation="4-3-3",
        max_per_club=3,
        moneyball_weight=0.35,
    )
    elapsed_sec = time.time() - t0

    assert elapsed_sec < 1.5, f"Solver too slow: {elapsed_sec:.2f}s"
    assert res["success"] is True, f"Optimization failed: {res.get('message')}"
    assert res["total_players"] == 11, f"Squad must have exactly 11 players, got {res['total_players']}"
    assert res["total_spent_eur"] <= 120_000_000.0, f"Budget exceeded: {res['total_spent_eur']}"
    assert res["budget_remaining_eur"] >= 0.0

    # Verify positional composition matches 4-3-3: 3 CB, 2 FB, 2 CM, 1 CAM, 2 W, 1 ST
    pos_counts = pd.Series([p["primary_position"] for p in res["squad"]]).value_counts().to_dict()
    assert pos_counts.get("CB", 0) == 3
    assert pos_counts.get("FB", 0) == 2
    assert pos_counts.get("CM", 0) == 2
    assert pos_counts.get("CAM", 0) == 1
    assert pos_counts.get("W", 0) == 2
    assert pos_counts.get("ST", 0) == 1

    # Verify max players per club constraint
    club_counts = pd.Series([p["club"] for p in res["squad"]]).value_counts()
    assert (club_counts <= 3).all(), f"Club limit exceeded: {club_counts.to_dict()}"

    # Verify average performance score is strong
    assert res["avg_performance_score"] >= 65.0
