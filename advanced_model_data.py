"""Synthetic data helpers for the Advanced Model Deep Dive section."""
from __future__ import annotations

import numpy as np
import pandas as pd


def generate_rsf_predictions(base_predictions_df: pd.DataFrame) -> pd.DataFrame:
    """Augment simple predictions with simulated RSF outputs.

    The function introduces realistic adjustments that mimic how a Random
    Survival Forest might respond to complex feature interactions.
    """

    rsf_predictions = base_predictions_df.copy()
    rsf_predictions["rsf_failure_probability"] = rsf_predictions["failure_probability"]

    rng = np.random.default_rng(7)
    for idx, row in rsf_predictions.iterrows():
        simple_prob = row["failure_probability"]
        adjustment = rng.normal(0, 0.02)

        if row.get("age_years", 0) > 45 and row.get("maintenance_count", 0) > 5:
            adjustment += 0.15
        elif row.get("age_years", 0) < 25 and row.get("condition_score", 5) <= 2:
            adjustment += 0.2
        elif row.get("material_standardized") == "wood" and row.get("traffic_volume_category") == "High":
            adjustment += 0.12
        elif row.get("condition_score", 0) >= 5 and row.get("age_years", 0) < 15:
            adjustment -= 0.1

        rsf_prob = np.clip(simple_prob + adjustment, 0.01, 0.98)
        rsf_predictions.at[idx, "rsf_failure_probability"] = rsf_prob

    return rsf_predictions


def create_model_comparison_data() -> pd.DataFrame:
    """Create a DataFrame comparing simple vs. RSF predictions."""

    simple_preds = pd.read_csv("data/prediction_results.csv")
    clean_data = pd.read_csv("data/clean_bridge_data.csv")
    raw_data = pd.read_csv("data/raw_bridge_data.csv")

    merged = simple_preds.merge(clean_data, on="bridge_id")
    merged = merged.merge(raw_data[["bridge_id", "bridge_name"]], on="bridge_id", how="left")

    rsf_enhanced = generate_rsf_predictions(merged)
    rsf_enhanced["prediction_diff"] = (
        rsf_enhanced["rsf_failure_probability"] - rsf_enhanced["failure_probability"]
    ).abs()

    rng = np.random.default_rng(42)
    outcomes = rng.choice(
        ["Failed within 12 months", "No failure within 12 months"],
        size=len(rsf_enhanced),
        p=[0.45, 0.55],
    )
    rsf_enhanced["historical_outcome"] = outcomes
    actual_numeric = (outcomes == "Failed within 12 months").astype(float)

    simple_error = (rsf_enhanced["failure_probability"] - actual_numeric).abs()
    rsf_error = (rsf_enhanced["rsf_failure_probability"] - actual_numeric).abs()
    rsf_enhanced["best_match"] = np.where(
        rsf_error <= simple_error,
        "Random Survival Forest",
        "Simple Model",
    )

    divergent_cases = rsf_enhanced.nlargest(20, "prediction_diff").copy()
    divergent_cases.sort_values("prediction_diff", ascending=False, inplace=True)

    return divergent_cases


def calculate_permutation_importance() -> pd.DataFrame:
    """Simulate permutation importance scores."""

    features = [
        "Age",
        "Material",
        "Condition Score",
        "Traffic Volume",
        "Days Since Inspection",
        "Maintenance Count",
    ]
    importance = [0.42, 0.28, 0.31, 0.23, 0.12, 0.18]

    return pd.DataFrame({"Feature": features, "Importance": importance})


def get_detailed_importance() -> pd.DataFrame:
    """Return richer feature importance values for the RSF model."""

    return pd.DataFrame(
        {
            "Feature": [
                "Age",
                "Condition Score",
                "Maintenance Count",
                "Traffic Volume",
                "Material",
                "Days Since Inspection",
            ],
            "Importance": [0.34, 0.26, 0.18, 0.11, 0.07, 0.04],
        }
    )
