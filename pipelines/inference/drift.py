"""
Lightweight feature drift detection.

Strategy: z-score of the mean deviation.
  - Reference stats (mean, std) are computed from the full historical feature
    table (all windows), representing the distribution the model was trained on.
  - Current stats come from the most recent inference batch (latest window per pair).
  - A feature is flagged as drifted when |current_mean - ref_mean| / ref_std
    exceeds DRIFT_THRESHOLD.

This is intentionally simple — suitable for a first monitoring layer before
adding a dedicated monitoring framework (e.g. Evidently, Nannyml).
"""
import pandas as pd

NUMERIC_FEATURES = [
    "previous_count",
    "rolling_avg_3",
    "rolling_avg_5",
    "growth_rate",
]

DRIFT_THRESHOLD = 1.0  # z-score units; tune as more data accumulates


def compute_reference_stats(features: pd.DataFrame) -> dict[str, dict]:
    """
    Compute mean and std for each numeric feature across all historical windows.

    Returns a dict keyed by feature name:
      {"mean": float, "std": float}
    """
    stats: dict[str, dict] = {}
    for col in NUMERIC_FEATURES:
        if col not in features.columns:
            continue
        series = features[col].dropna()
        stats[col] = {
            "mean": float(series.mean()),
            "std": float(series.std()) if len(series) > 1 else 1.0,
        }
    return stats


def detect_drift(
    current: pd.DataFrame,
    reference_stats: dict[str, dict],
    threshold: float = DRIFT_THRESHOLD,
) -> dict:
    """
    Compare current batch feature means against reference stats.

    Returns:
      {
        "drift_detected": bool,
        "threshold": float,
        "features": {
          "<feature>": {
            "reference_mean": float,
            "current_mean": float,
            "z_score": float,
            "drifted": bool,
          },
          ...
        }
      }
    """
    feature_reports: dict[str, dict] = {}
    any_drift = False

    for col, ref in reference_stats.items():
        if col not in current.columns:
            continue

        current_mean = float(current[col].dropna().mean())
        ref_std = ref["std"] if ref["std"] > 0 else 1.0
        z_score = abs(current_mean - ref["mean"]) / ref_std
        drifted = z_score > threshold

        if drifted:
            any_drift = True

        feature_reports[col] = {
            "reference_mean": round(ref["mean"], 4),
            "current_mean": round(current_mean, 4),
            "z_score": round(z_score, 4),
            "drifted": drifted,
        }

    return {
        "drift_detected": any_drift,
        "threshold": threshold,
        "features": feature_reports,
    }
