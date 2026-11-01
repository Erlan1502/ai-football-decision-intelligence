import os
import json
import pytest
import numpy as np

MOCK_DATA_PATH = os.path.join(os.path.dirname(__file__), "..", "db", "seed", "mock_players.json")


@pytest.fixture(scope="session")
def dataset():
    assert os.path.exists(MOCK_DATA_PATH), f"Mock dataset missing at {MOCK_DATA_PATH}"
    with open(MOCK_DATA_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data


def test_dataset_size_and_schema(dataset):
    """Verify minimum 500+ players and required top-level attributes."""
    assert len(dataset) >= 500, f"Expected at least 500 players, got {len(dataset)}"
    sample = dataset[0]
    required_keys = {
        "id", "name", "age", "nationality", "club", "league",
        "primary_position", "market_value_eur", "minutes_played",
        "performance_score", "stats", "embedding", "valuation"
    }
    for k in required_keys:
        assert k in sample, f"Missing key '{k}' in player record"


def test_tactical_role_distribution(dataset):
    """Verify all 6 core positions are well-represented."""
    positions = set(p["primary_position"] for p in dataset)
    expected_positions = {"CB", "FB", "CM", "CAM", "W", "ST"}
    assert positions == expected_positions, f"Expected {expected_positions}, got {positions}"

    counts = {pos: sum(1 for p in dataset if p["primary_position"] == pos) for pos in expected_positions}
    for pos, count in counts.items():
        assert count >= 50, f"Position {pos} has too few samples ({count})"


def test_tactical_embedding_l2_normalization(dataset):
    """Verify that all tactical embeddings are exactly 64-dimensional and unit normalized."""
    for p in dataset[:100]:
        emb = np.array(p["embedding"], dtype=np.float32)
        assert len(emb) == 64, f"Embedding must be 64-dim, got {len(emb)}"
        norm = np.linalg.norm(emb)
        assert abs(norm - 1.0) < 1e-3, f"Embedding vector must be L2-normalized, norm={norm}"


def test_domain_metric_correlations(dataset):
    """Verify football domain logic: Strikers lead in npxG, CBs lead in clearances."""
    cbs = [p for p in dataset if p["primary_position"] == "CB"]
    sts = [p for p in dataset if p["primary_position"] == "ST"]
    cms = [p for p in dataset if p["primary_position"] == "CM"]
    ws = [p for p in dataset if p["primary_position"] == "W"]

    avg_st_npxg = np.mean([p["stats"]["npxg_p90"] for p in sts])
    avg_cb_npxg = np.mean([p["stats"]["npxg_p90"] for p in cbs])
    assert avg_st_npxg > 0.40, f"Strikers average npxG should be > 0.40, got {avg_st_npxg}"
    assert avg_cb_npxg < 0.10, f"Center backs average npxG should be < 0.10, got {avg_cb_npxg}"
    assert avg_st_npxg > avg_cb_npxg * 4, "Strikers npxG must significantly exceed CBs"

    avg_cb_clearances = np.mean([p["stats"]["clearances_p90"] for p in cbs])
    avg_st_clearances = np.mean([p["stats"]["clearances_p90"] for p in sts])
    assert avg_cb_clearances > 3.5, f"CB average clearances should be > 3.5, got {avg_cb_clearances}"
    assert avg_cb_clearances > avg_st_clearances * 4

    avg_cm_prgp = np.mean([p["stats"]["prgp_p90"] for p in cms])
    avg_st_prgp = np.mean([p["stats"]["prgp_p90"] for p in sts])
    assert avg_cm_prgp > avg_st_prgp * 2, "Midfielders must lead progressive passes"

    avg_w_prgc = np.mean([p["stats"]["prgc_p90"] for p in ws])
    avg_cb_prgc = np.mean([p["stats"]["prgc_p90"] for p in cbs])
    assert avg_w_prgc > avg_cb_prgc * 3, "Wingers must lead progressive carries"


def test_anchor_stars_present(dataset):
    """Verify elite anchor stars and donor gems are seeded with correct metrics."""
    names = {p["name"]: p for p in dataset}
    assert "Rodri" in names
    rodri = names["Rodri"]
    assert rodri["primary_position"] == "CM"
    assert rodri["market_value_eur"] == 110_000_000
    assert rodri["stats"]["pass_completion_pct"] >= 92.0

    assert "Vinicius Jr" in names
    vini = names["Vinicius Jr"]
    assert vini["primary_position"] == "W"
    assert vini["market_value_eur"] == 160_000_000

    assert "Marten Hjulmand" in names
    hjulmand = names["Marten Hjulmand"]
    assert hjulmand["league"] == "Liga Portugal"
    # Moneyball Arbitrage: Hjulmand's market value (€35M) is less than half of Rodri's (€110M)
    assert hjulmand["market_value_eur"] < rodri["market_value_eur"] * 0.5
    assert hjulmand["valuation"]["undervaluation_delta_eur"] > 0

