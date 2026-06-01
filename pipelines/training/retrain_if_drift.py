"""
Drift-gated retraining entry point.

Reads the latest complete features from Hopsworks, computes drift against the
full historical distribution, and retrains only if drift exceeds the threshold.

Usage:
    python -m pipelines.training.retrain_if_drift
"""
import pandas as pd

from pipelines.feature.aggregate import WINDOW_DAYS
from pipelines.feature.hopsworks_reader import read_feature_group
from pipelines.inference.drift import DRIFT_THRESHOLD, compute_reference_stats, detect_drift
from pipelines.inference.prepare import build_inference_features


def main() -> None:
    features = read_feature_group()
    today = pd.Timestamp.now().normalize()

    complete_features = features[
        features["window_start"] + pd.Timedelta(days=WINDOW_DAYS) <= today
    ]

    reference_stats = compute_reference_stats(complete_features)
    current = build_inference_features(complete_features)
    report = detect_drift(current, reference_stats)

    print(f"Drift detected: {report['drift_detected']} (threshold: {DRIFT_THRESHOLD})")
    for feature, result in report["features"].items():
        print(f"  {feature}: z={result['z_score']:.3f} {'DRIFTED' if result['drifted'] else 'ok'}")

    if report["drift_detected"]:
        print("Drift threshold exceeded — retraining model.")
        from pipelines.training.run import run
        run(from_hopsworks=True)
    else:
        print("No significant drift — skipping retraining.")


if __name__ == "__main__":
    main()
