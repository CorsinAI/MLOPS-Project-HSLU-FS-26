"""
Entry point for the training pipeline.

Usage:
    python -m pipelines.training.run
    python -m pipelines.training.run --local-input data/postings.jsonl
"""
import argparse
from pathlib import Path

from pipelines.training.prepare import (
    HOLDOUT_WINDOWS,
    build_dataset,
    encode_categoricals,
    time_split,
    trim_for_training,
)


def run(local_input: Path | None = None) -> None:
    if local_input is not None:
        from pipelines.feature.aggregate import load_postings, assign_windows, aggregate_counts
        from pipelines.feature.features import compute_features
        print(f"Building features from {local_input} ...")
        df = load_postings(local_input)
        df = assign_windows(df)
        counts = aggregate_counts(df)
        features = compute_features(counts)
    else:
        from pipelines.feature.hopsworks_reader import read_feature_group
        print("Reading features from Hopsworks ...")
        features = read_feature_group()

    print(f"  {len(features)} feature rows across {features['window_start'].nunique()} windows")

    # --- Hold out the most recent complete windows ---
    features = trim_for_training(features)
    print(f"  After holdout: {features['window_start'].nunique()} windows "
          f"(latest {HOLDOUT_WINDOWS} complete windows reserved for inference)")

    # --- Prepare training data ---
    print("Preparing training dataset ...")
    dataset = build_dataset(features)
    dataset = encode_categoricals(dataset)

    train_df, val_df, test_df = time_split(dataset)

    print(f"  Train rows: {len(train_df)}  ({train_df['window_start'].min().date()} → {train_df['window_start'].max().date()})")
    print(f"  Val rows:   {len(val_df)}  ({val_df['window_start'].min().date()} → {val_df['window_start'].max().date()})")
    print(f"  Test rows:  {len(test_df)}  ({test_df['window_start'].min().date()} → {test_df['window_start'].max().date()})")

    # --- Train ---
    from pipelines.training.train import train_model
    print("Training LightGBM ...")
    train_model(train_df, val_df, test_df)

    print("\nDone. MLflow run logged.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--local-input", type=Path, default=None,
        help="Path to a local JSONL file; skips Hopsworks feature read",
    )
    args = parser.parse_args()
    run(local_input=args.local_input)
