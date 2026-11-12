"""Create a Kaggle submission from a Lab 4 CNN checkpoint.

The model, loss, and per-image preprocessing are reconstructed through the
same helpers used by training.  The sample submission defines the IDs and row
order; each image is then processed independently.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml

from src.data import inputs
from src import losses
from src.models.build import build_model


def load_model(checkpoint: Path):
    """Rebuild the saved model and loss, then load the trained weights."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    ckpt = torch.load(checkpoint, map_location=device, weights_only=False)

    required = {"state_dict", "config", "in_shape", "loss_state", "scaler"}
    missing = required.difference(ckpt)
    if missing:
        raise KeyError(f"checkpoint is missing required keys: {sorted(missing)}")

    loss = losses.from_state(ckpt["loss_state"])
    model = build_model(ckpt["config"], ckpt["in_shape"], loss)
    model.load_state_dict(ckpt["state_dict"])
    model.to(device)
    model.eval()
    return model, loss, ckpt, device


def predict_one(model, loss, image_path: Path, cfg: dict, ckpt: dict, device: str) -> float:
    """Preprocess one original photograph and return its usability prediction."""
    # preprocess_one returns a batch of one, with the same preprocessing used
    # to construct the training table (including the cached-JPEG round trip).
    x = inputs.preprocess_one(image_path, cfg, ckpt)
    if not isinstance(x, torch.Tensor):
        x = torch.as_tensor(x, dtype=torch.float32)
    x = x.to(device=device, dtype=torch.float32)

    with torch.inference_mode():
        output = model(x)                 # model returns logits in Lab 4
        prediction = loss.predict(output)  # loss maps logits to usability [0, 1]

    prediction = prediction.detach().reshape(-1)
    if prediction.numel() != 1:
        raise ValueError(
            f"expected one prediction for {image_path}, got shape {tuple(prediction.shape)}"
        )
    return float(prediction.cpu().item())


def main() -> None:
    parser = argparse.ArgumentParser(description="Write a Kaggle submission CSV")
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--test-dir", default="data/raw/test")
    parser.add_argument("--sample", default="data/raw/sample_submission.csv")
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    with open(args.config, encoding="utf-8") as file:
        cfg = yaml.safe_load(file)

    sample = pd.read_csv(args.sample)
    if list(sample.columns) != ["image_id", "usability"]:
        raise SystemExit(f"unexpected sample columns: {list(sample.columns)}")
    if sample["image_id"].isna().any():
        raise SystemExit("sample submission contains a missing image_id")
    if not sample["image_id"].is_unique:
        raise SystemExit("sample submission contains duplicate image_id values")

    model, loss, ckpt, device = load_model(Path(args.checkpoint))
    test_dir = Path(args.test_dir)

    predictions: list[float] = []
    for raw_image_id in sample["image_id"]:
        image_id = str(raw_image_id)
        image_path = test_dir / f"{image_id}.jpg"
        if not image_path.is_file():
            raise SystemExit(f"missing test image: {image_path}")
        predictions.append(predict_one(model, loss, image_path, cfg, ckpt, device))

    submission = pd.DataFrame(
        {"image_id": sample["image_id"], "usability": predictions}
    )
    submission["usability"] = submission["usability"].clip(0.0, 1.0)

    # Validate the submission contract before writing it.
    if len(submission) != len(sample):
        raise SystemExit("row count does not match the sample submission")
    if not submission["image_id"].is_unique:
        raise SystemExit("duplicate image_id in predictions")
    if not np.isfinite(submission["usability"].to_numpy()).all():
        raise SystemExit("NaN or infinite value in predictions")
    if not submission["usability"].between(0.0, 1.0).all():
        raise SystemExit("prediction outside [0, 1]")

    output_path = Path(args.out)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    submission.to_csv(output_path, index=False)
    print(f"wrote {len(submission)} rows to {output_path}")
    print(submission["usability"].describe())
    print(
        "\nRemember: keep the checkpoint and record which training run produced it."
    )


if __name__ == "__main__":
    main()
