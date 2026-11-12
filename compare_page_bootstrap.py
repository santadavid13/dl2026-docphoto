"""Paired page-bootstrap comparison for the Lab 4 CNN experiments.

Run from the project root inside Docker:
  docker compose run --rm dev python compare_page_bootstrap.py

Expected validation prediction files:
  submissions/val_8_none.csv
  submissions/val_12_safe.csv
  submissions/val_16_safe.csv
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from src import evaluate


def read_predictions(path: Path, all_validation_ids: pd.Series, scored_ids: pd.Series) -> np.ndarray:
    if not path.is_file():
        raise SystemExit(f"Missing predictions file: {path}")
    frame = pd.read_csv(path)
    required = {"image_id", "usability"}
    if not required.issubset(frame.columns):
        raise SystemExit(f"{path} must contain columns {sorted(required)}")
    if frame["image_id"].isna().any() or not frame["image_id"].is_unique:
        raise SystemExit(f"{path} contains missing or duplicate image_id values")
    indexed = frame.set_index("image_id")["usability"]
    missing = all_validation_ids[~all_validation_ids.isin(indexed.index)]
    extra = indexed.index[~indexed.index.isin(all_validation_ids)]
    if len(missing) or len(extra):
        raise SystemExit(
            f"{path} IDs do not match validation sample: "
            f"{len(missing)} missing, {len(extra)} extra"
        )
    values = indexed.reindex(scored_ids).to_numpy(dtype=float)
    if not np.isfinite(values).all() or ((values < 0) | (values > 1)).any():
        raise SystemExit(f"{path} contains non-finite or out-of-range predictions")
    return values


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-csv", type=Path, default=Path("data/raw/train.csv"))
    parser.add_argument("--sample", type=Path, default=Path("data/work/val_sample.csv"))
    parser.add_argument("--pred-8", type=Path, default=Path("submissions/val_8_none.csv"))
    parser.add_argument("--pred-12", type=Path, default=Path("submissions/val_12_safe.csv"))
    parser.add_argument("--pred-16", type=Path, default=Path("submissions/val_16_safe.csv"))
    parser.add_argument("--bootstrap", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    train = pd.read_csv(args.train_csv)
    sample = pd.read_csv(args.sample)
    if "image_id" not in sample or not sample["image_id"].is_unique:
        raise SystemExit("Validation sample must contain unique image_id values")
    needed = {"image_id", "usability", "eval_ok", "page_id"}
    if not needed.issubset(train.columns):
        raise SystemExit(f"Training table must contain columns {sorted(needed)}")

    validation = sample[["image_id"]].merge(
        train[list(needed)], on="image_id", how="left", validate="one_to_one", sort=False
    )
    if validation["usability"].isna().any() or validation["page_id"].isna().any():
        raise SystemExit("Some validation IDs are missing labels or page_id in train.csv")

    scored = validation[validation["eval_ok"] == 1].copy().reset_index(drop=True)
    if scored.empty:
        raise SystemExit("No scoreable validation rows (eval_ok == 1)")

    all_ids = validation["image_id"]
    ids = scored["image_id"]
    y = scored["usability"].to_numpy(dtype=float)
    pages = scored["page_id"].to_numpy()
    predictions = {
        "8 epochs / no augmentation": read_predictions(args.pred_8, all_ids, ids),
        "12 epochs / safe": read_predictions(args.pred_12, all_ids, ids),
        "16 epochs / safe": read_predictions(args.pred_16, all_ids, ids),
    }

    print(f"Scoreable validation images: {len(scored)}")
    print(f"Distinct validation pages: {pd.Series(pages).nunique()}")
    print(f"Bootstrap repetitions: {args.bootstrap}")
    print("\nValidation RMSE:")
    for name, pred in predictions.items():
        print(f"  {name:30s} {evaluate.rmse(y, pred):.6f}")

    comparisons = [
        ("8 epochs / no augmentation", "12 epochs / safe"),
        ("8 epochs / no augmentation", "16 epochs / safe"),
        ("12 epochs / safe", "16 epochs / safe"),
    ]
    print("\nPaired page-bootstrap comparisons (difference = B RMSE - A RMSE):")
    print("Positive difference means B has higher/worse RMSE.")
    for index, (name_a, name_b) in enumerate(comparisons):
        result = evaluate.page_bootstrap(
            y,
            predictions[name_a],
            predictions[name_b],
            pages,
            n=args.bootstrap,
            seed=args.seed + index,
        )
        print(f"\n  A: {name_a}\n  B: {name_b}")
        print(f"  RMSE A:             {result['rmse_a']:.6f}")
        print(f"  RMSE B:             {result['rmse_b']:.6f}")
        print(f"  Difference (B - A): {result['diff']:+.6f}")
        print(f"  95% bootstrap CI:   [{result['lo']:+.6f}, {result['hi']:+.6f}]")
        print(f"  Share B better:     {result['share_b_better']:.3f}")
        print(f"  Pages resampled:    {result['pages']}")

    print("\nInterpretation: a CI containing 0 does not establish a clear difference")
    print("between the two models on this validation set. The 8-epoch comparison")
    print("also changes augmentation, so 12 vs 16 safe is the clean duration comparison.")


if __name__ == "__main__":
    main()
