import os
import json
from typing import Tuple, List, Dict, Any, Optional
import pandas as pd
import numpy as np

MOCK_PLAYERS_PATH = os.path.join(os.path.dirname(__file__), "..", "db", "seed", "mock_players.json")

# Standardized feature schema
NUMERICAL_STATS = [
    "age",
    "minutes_played",
    "matches_played",
    "goals_p90",
    "assists_p90",
    "npxg_p90",
    "shots_total_p90",
    "shots_on_target_pct",
    "xag_p90",
    "shot_creating_actions_p90",
    "goal_creating_actions_p90",
    "key_passes_p90",
    "pass_completion_pct",
    "passes_attempted_p90",
    "prgp_p90",
    "passes_into_final_third_p90",
    "passes_into_penalty_area_p90",
    "prgc_p90",
    "dribbles_completed_p90",
    "dribble_success_pct",
    "carries_into_final_third_p90",
    "dispossessed_p90",
    "tackles_p90",
    "tackles_won_p90",
    "interceptions_p90",
    "blocks_p90",
    "clearances_p90",
    "recoveries_p90",
    "tackles_def_3rd_p90",
    "tackles_mid_3rd_p90",
    "tackles_att_3rd_p90",
    "aerial_duels_won_pct",
    "aerial_duels_total_p90",
]

CATEGORICAL_COLS = ["primary_position", "league"]


def load_raw_dataset(filepath: Optional[str] = None) -> List[Dict[str, Any]]:
    """Loads raw player list from JSON file or database."""
    target_path = filepath or MOCK_PLAYERS_PATH
    if not os.path.exists(target_path):
        from db.seed.seed_data import generate_full_dataset, save_dataset_json
        data = generate_full_dataset(550)
        save_dataset_json(data, target_path)
        return data

    with open(target_path, "r", encoding="utf-8") as f:
        return json.load(f)


def prepare_tabular_data(
    filepath: Optional[str] = None,
    min_minutes: int = 450,
) -> Tuple[pd.DataFrame, pd.Series, List[str], pd.DataFrame]:
    """
    Transforms nested player records into flat feature matrix X and target y = log1p(market_value_eur).
    
    Returns:
        X: pd.DataFrame with one-hot encoded categoricals + scaled numericals
        y: pd.Series with log1p(market_value_eur)
        feature_names: List of column names in X
        df_full: Full DataFrame containing metadata (id, name, club, etc.)
    """
    raw_data = load_raw_dataset(filepath)

    rows = []
    for p in raw_data:
        flat = {
            "id": p["id"],
            "name": p["name"],
            "club": p["club"],
            "league": p["league"],
            "primary_position": p["primary_position"],
            "market_value_eur": p["market_value_eur"],
            "performance_score": p.get("performance_score", 65.0),
            "contract_expires_year": p.get("contract_expires_year", 2028),
            "age": p["age"],
            "minutes_played": p["minutes_played"],
            "matches_played": p["matches_played"],
        }
        flat.update(p["stats"])
        rows.append(flat)

    df_full = pd.DataFrame(rows)

    # Filter by minimum minutes played
    if min_minutes > 0:
        df_full = df_full[df_full["minutes_played"] >= min_minutes].reset_index(drop=True)

    # Target variable: log-transformed market value
    y = np.log1p(df_full["market_value_eur"])

    # Features preparation
    df_num = df_full[NUMERICAL_STATS].copy()

    # One-hot encode categoricals
    df_cat = pd.get_dummies(df_full[CATEGORICAL_COLS], prefix=CATEGORICAL_COLS, drop_first=False, dtype=float)

    X = pd.concat([df_num, df_cat], axis=1)
    feature_names = list(X.columns)

    return X, y, feature_names, df_full

