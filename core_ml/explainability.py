import os
from typing import List, Dict, Any, Optional, Tuple
import numpy as np
import pandas as pd
import shap
import lightgbm as lgb

from core_ml.data_loader import prepare_tabular_data
from core_ml.train_value_model import load_trained_model

# Human-readable labels for scouting dossiers
FEATURE_LABELS = {
    "prgp_p90": "Progressive Passes (p90)",
    "prgc_p90": "Progressive Carries (p90)",
    "xag_p90": "Expected Assisted Goals (xAG/90)",
    "npxg_p90": "Non-Penalty Expected Goals (npxG/90)",
    "shot_creating_actions_p90": "Shot-Creating Actions (p90)",
    "goal_creating_actions_p90": "Goal-Creating Actions (p90)",
    "key_passes_p90": "Key Passes (p90)",
    "pass_completion_pct": "Pass Accuracy (%)",
    "passes_attempted_p90": "Pass Volume (p90)",
    "passes_into_penalty_area_p90": "Passes into Penalty Area",
    "passes_into_final_third_p90": "Passes into Final Third",
    "dribbles_completed_p90": "Successful Take-ons (p90)",
    "dribble_success_pct": "Dribble Success Rate (%)",
    "carries_into_final_third_p90": "Carries into Final Third",
    "tackles_p90": "Total Tackles (p90)",
    "tackles_won_p90": "Tackles Won (p90)",
    "interceptions_p90": "Interceptions (p90)",
    "clearances_p90": "Clearances (p90)",
    "recoveries_p90": "Ball Recoveries (p90)",
    "tackles_def_3rd_p90": "Defensive 3rd Tackles",
    "tackles_mid_3rd_p90": "Middle 3rd Tackles",
    "tackles_att_3rd_p90": "Attacking 3rd Tackles",
    "aerial_duels_won_pct": "Aerial Duel Win Rate (%)",
    "aerial_duels_total_p90": "Aerial Duels Contested",
    "shots_total_p90": "Shots per 90",
    "shots_on_target_pct": "Shot Accuracy (%)",
    "goals_p90": "Goals per 90",
    "assists_p90": "Assists per 90",
    "age": "Player Age (Curve)",
    "minutes_played": "Season Minutes Volume",
    "matches_played": "Match Involvement",
}


class MarketValueExplainer:
    """
    SHAP TreeExplainer for interpreting LightGBM market value predictions.
    Computes individual feature attributions in EUR impact for scout transfer audits.
    """

    def __init__(self, model: Optional[lgb.LGBMRegressor] = None, feature_names: Optional[List[str]] = None):
        if model is None or feature_names is None:
            self.model, self.feature_names = load_trained_model()
        else:
            self.model = model
            self.feature_names = feature_names

        self.explainer = shap.TreeExplainer(self.model)
        exp_val = self.explainer.expected_value
        if isinstance(exp_val, (np.ndarray, list)):
            self.expected_value_log = float(np.ravel(exp_val)[0])
        else:
            self.expected_value_log = float(exp_val)
        self.expected_value_eur = float(np.expm1(self.expected_value_log))

    def explain_player(
        self,
        player_row: pd.Series,
        top_k: int = 4,
    ) -> Dict[str, Any]:
        """
        Computes SHAP feature attribution for a single player.
        Returns top positive (value drivers) and negative (discount factors).
        """
        X_sample = pd.DataFrame([player_row[self.feature_names]])
        raw_shap = self.explainer.shap_values(X_sample)
        shap_values = np.ravel(raw_shap)  # 1D array of length n_features
        pred_log = float(self.model.predict(X_sample)[0])
        pred_eur = float(np.expm1(pred_log))

        # Currency attribution: proportional decomposition of the difference from baseline
        total_shap_log = float(np.sum(shap_values))
        value_gap_eur = pred_eur - self.expected_value_eur

        driver_records = []
        for i, feat in enumerate(self.feature_names):
            phi = float(shap_values[i])
            if abs(phi) < 1e-4:
                continue

            # Fractional share in EUR
            if abs(total_shap_log) > 1e-4:
                impact_eur = value_gap_eur * (phi / total_shap_log)
            else:
                impact_eur = (np.exp(phi) - 1.0) * self.expected_value_eur

            clean_label = FEATURE_LABELS.get(feat, feat.replace("_", " ").title())
            feat_val = float(player_row.get(feat, 0.0))

            driver_records.append({
                "feature": feat,
                "label": clean_label,
                "value": round(feat_val, 2),
                "shap_log": round(phi, 4),
                "impact_eur": round(impact_eur),
                "direction": "positive" if phi > 0 else "negative",
            })

        pos_drivers = sorted([d for d in driver_records if d["direction"] == "positive"], key=lambda x: x["shap_log"], reverse=True)[:top_k]
        neg_drivers = sorted([d for d in driver_records if d["direction"] == "negative"], key=lambda x: x["shap_log"])[:top_k]

        return {
            "predicted_value_eur": round(pred_eur),
            "baseline_value_eur": round(self.expected_value_eur),
            "top_positive_drivers": pos_drivers,
            "top_negative_drivers": neg_drivers,
            "total_features_analyzed": len(self.feature_names),
        }
