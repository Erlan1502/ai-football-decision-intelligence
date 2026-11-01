import os
import json
from typing import Dict, Any, Tuple, Optional, List
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import lightgbm as lgb
import joblib

from core_ml.data_loader import prepare_tabular_data

ARTIFACTS_DIR = os.path.join(os.path.dirname(__file__), "artifacts")
MODEL_PATH = os.path.join(ARTIFACTS_DIR, "lgb_value_model.joblib")
FEATURE_NAMES_PATH = os.path.join(ARTIFACTS_DIR, "feature_names.json")
METRICS_PATH = os.path.join(ARTIFACTS_DIR, "training_metrics.json")


def train_market_value_regressor(
    test_size: float = 0.2,
    random_state: int = 42,
    save_artifacts: bool = True,
) -> Tuple[lgb.LGBMRegressor, Dict[str, float], pd.DataFrame]:
    """
    Trains a production-grade LightGBM Regressor to estimate fair market value
    from sports p90 performance metrics, age, league, and playing time.
    """
    os.makedirs(ARTIFACTS_DIR, exist_ok=True)

    X, y, feature_names, df_full = prepare_tabular_data()

    X_train, X_test, y_train, y_test, idx_train, idx_test = train_test_split(
        X, y, df_full.index, test_size=test_size, random_state=random_state
    )

    model = lgb.LGBMRegressor(
        n_estimators=300,
        learning_rate=0.035,
        num_leaves=31,
        max_depth=6,
        subsample=0.85,
        colsample_bytree=0.85,
        reg_alpha=0.1,
        reg_lambda=0.1,
        random_state=random_state,
        n_jobs=-1,
        verbose=-1,
    )

    model.fit(
        X_train,
        y_train,
        eval_set=[(X_test, y_test)],
        callbacks=[lgb.early_stopping(stopping_rounds=30, verbose=False)],
    )

    # Evaluate on test set
    y_pred_log = model.predict(X_test)
    r2 = float(r2_score(y_test, y_pred_log))
    rmse_log = float(np.sqrt(mean_squared_error(y_test, y_pred_log)))
    mae_log = float(mean_absolute_error(y_test, y_pred_log))

    # Evaluate on real currency scale (EUR)
    y_test_eur = np.expm1(y_test)
    y_pred_eur = np.expm1(y_pred_log)
    mae_eur = float(mean_absolute_error(y_test_eur, y_pred_eur))
    mape_eur = float(np.mean(np.abs((y_test_eur - y_pred_eur) / y_test_eur)) * 100.0)

    metrics = {
        "r2_score": round(r2, 4),
        "rmse_log": round(rmse_log, 4),
        "mae_log": round(mae_log, 4),
        "mae_eur": round(mae_eur, 2),
        "mape_pct": round(mape_eur, 2),
        "train_samples": len(X_train),
        "test_samples": len(X_test),
    }

    print("=================================================================")
    print(" LightGBM Football Market Valuation Model Training Summary")
    print("=================================================================")
    print(f" R^2 Score (Log-Scale):       {metrics['r2_score']:.4f} (DoD Target >= 0.75)")
    print(f" RMSE (Log-Scale):            {metrics['rmse_log']:.4f}")
    print(f" MAE (Currency Scale):        EUR {metrics['mae_eur']:,.0f}")
    print(f" MAPE (Mean % Error):         {metrics['mape_pct']:.2f}%")
    print("=================================================================")

    # Predict full dataset valuations and compute Moneyball Delta
    full_pred_log = model.predict(X)
    df_full["predicted_value_eur"] = np.round(np.expm1(full_pred_log), -5)
    df_full["undervaluation_delta_eur"] = df_full["predicted_value_eur"] - df_full["market_value_eur"]
    df_full["undervaluation_ratio"] = np.round(df_full["predicted_value_eur"] / (df_full["market_value_eur"] + 1), 3)

    if save_artifacts:
        joblib.dump(model, MODEL_PATH)
        with open(FEATURE_NAMES_PATH, "w", encoding="utf-8") as f:
            json.dump(feature_names, f, indent=2)
        with open(METRICS_PATH, "w", encoding="utf-8") as f:
            json.dump(metrics, f, indent=2)
        print(f"[Artifacts] Model and features saved to {ARTIFACTS_DIR}")

    return model, metrics, df_full


def load_trained_model() -> Tuple[lgb.LGBMRegressor, List[str]]:
    """Loads serialized LightGBM model and feature schema."""
    if not os.path.exists(MODEL_PATH) or not os.path.exists(FEATURE_NAMES_PATH):
        model, _, _ = train_market_value_regressor(save_artifacts=True)
        with open(FEATURE_NAMES_PATH, "r", encoding="utf-8") as f:
            feature_names = json.load(f)
        return model, feature_names

    model = joblib.load(MODEL_PATH)
    with open(FEATURE_NAMES_PATH, "r", encoding="utf-8") as f:
        feature_names = json.load(f)
    return model, feature_names


if __name__ == "__main__":
    train_market_value_regressor()
