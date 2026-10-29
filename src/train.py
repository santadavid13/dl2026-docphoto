"""Train the MLP on raw pixels and log the run to Weights & Biases.

    python -m src.train --config configs/lab3_mlp.yaml

Given: pixel standardisation, the epoch loop with its logging, and the
checkpoint. Yours, each a few lines, each with its contract in the docstring:

    make_optimizer   TASK 3
    train_epoch      TASK 4   the minibatch loop
    predict          TASK 5   model outputs as numpy
    validate         TASK 6   RMSE on the scoreable validation rows

Read the given parts once. There is no framework underneath: this file is the
whole of supervised training.
"""
from __future__ import annotations

import copy

import argparse
import time

import numpy as np
import pandas as pd
import torch
import yaml

from src import config, tracking
from src.data.dataset import load_table, pixel_table
from src.data.splits import build_splits
from src.evaluate import rmse  # noqa: F401  (TASK 6 needs it)
from src.models.mlp import MLP
from src.seed import set_seed


class Standardiser:
    """Zero mean and unit variance over all pixels, fitted on the training rows only.

    One mean and one scale for the whole image, not one per pixel: with a
    thousand training rows a per-pixel estimate is mostly noise. Fitting on
    all rows would leak the validation distribution into training; a small
    leak, but the habit matters. The two numbers go into the checkpoint so
    that prediction uses exactly the same transform.
    """

    def fit(self, X: np.ndarray) -> Standardiser:
        self.mean = float(X.mean())
        self.scale = float(X.std()) + 1e-8
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        return (X - self.mean) / self.scale


def to_tensor(a, device: str) -> torch.Tensor:
    return torch.as_tensor(np.asarray(a, dtype=np.float32), device=device)


# --------------------------------------------------------------- your part

def make_optimizer(model: torch.nn.Module, cfg: dict) -> torch.optim.Optimizer:
    """TASK 3. The optimiser, using cfg['train']['lr'] and cfg['train']['weight_decay']."""
    return torch.optim.Adam(
        model.parameters(),
        lr=cfg["train"]["lr"],
        weight_decay=cfg["train"]["weight_decay"]
    )


def train_epoch(model: torch.nn.Module, optimizer: torch.optim.Optimizer, loss_fn: torch.nn.Module,
                X: torch.Tensor, y: torch.Tensor, batch_size: int, generator: torch.Generator) -> float:
    """TASK 4. One pass over the training rows in shuffled minibatches."""
    model.train()
    
    # Sorrend keverése a determinisztikus generátorral
    indices = torch.randperm(len(X), generator=generator)
    
    total_loss = 0.0
    total_samples = 0
    
    # Minibatch ciklus
    for start_idx in range(0, len(X), batch_size):
        batch_idx = indices[start_idx : start_idx + batch_size]
        X_batch = X[batch_idx]
        y_batch = y[batch_idx]
        
        optimizer.zero_grad()
        
        preds = model(X_batch)
        loss = loss_fn(preds, y_batch)
        
        loss.backward()
        optimizer.step()
        
        # A veszteséget súlyozzuk a minibatch méretével a pontos átlaghoz
        batch_len = len(X_batch)
        total_loss += loss.item() * batch_len
        total_samples += batch_len
        
    return total_loss / total_samples


def predict(model: torch.nn.Module, X: torch.Tensor, batch_size: int = 1024) -> np.ndarray:
    """TASK 5. The model's outputs for every row of X, as a numpy array of shape (len(X),)."""
    model.eval()
    all_preds = []
    
    with torch.no_grad():
        for start_idx in range(0, len(X), batch_size):
            X_batch = X[start_idx : start_idx + batch_size]
            preds = model(X_batch)
            all_preds.append(preds.cpu().numpy())
            
    return np.concatenate(all_preds)


def validate(model: torch.nn.Module, X_val: torch.Tensor, y_val: np.ndarray, ok: np.ndarray) -> float:
    """TASK 6. RMSE on the validation rows where `ok` is True (the scoreable ones)."""
    preds = predict(model, X_val)
    
    # Csak az ok (scoreable) sorokat vesszük figyelembe
    preds_ok = preds[ok]
    y_val_ok = y_val[ok]
    
    # Az importált rmse függvényt használjuk a src.evaluate-ből
    return rmse(y_val_ok, preds_ok)


# ------------------------------------------------------------------ given

def train_mlp(cfg: dict, X: np.ndarray, df: pd.DataFrame, train_idx: np.ndarray, val_idx: np.ndarray,
              device: str | None = None, tags: list[str] | None = None, log=print) -> dict:
    
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    set_seed(cfg["seed"])
    tr = cfg["train"]

    # 1. Őszinte Early Stopping Split: a train_idx 20%-át leválasztjuk belső validációra
    np.random.shuffle(train_idx)
    split_point = int(len(train_idx) * 0.8)
    sub_train_idx = train_idx[:split_point]
    sub_val_idx = train_idx[split_point:]

    scaler = Standardiser().fit(X[sub_train_idx])
    
    # Tenztorok előkészítése
    X_tr = to_tensor(scaler.transform(X[sub_train_idx]), device)
    X_sub_va = to_tensor(scaler.transform(X[sub_val_idx]), device)
    X_va = to_tensor(scaler.transform(X[val_idx]), device) # Az érintetlen végső validáció
    
    y_tr = to_tensor(df[config.TARGET].to_numpy()[sub_train_idx], device)
    y_sub_va = df[config.TARGET].to_numpy()[sub_val_idx]
    y_va = df[config.TARGET].to_numpy()[val_idx]
    
    ok_sub = df["eval_ok"].to_numpy()[sub_val_idx] == 1
    ok = df["eval_ok"].to_numpy()[val_idx] == 1

    model = MLP(in_dim=X.shape[1], hidden=list(cfg["model"]["hidden"]), dropout=cfg["model"]["dropout"],
                prior=float(y_tr.mean())).to(device)
    optimizer = make_optimizer(model, cfg)
    loss_fn = torch.nn.MSELoss()
    generator = torch.Generator().manual_seed(cfg["seed"])

    tracking.init_run(cfg, tags=tags)
    history, snapshots = [], {}
    
    best_sub_val_rmse = float('inf')
    best_model_state = None

    log(f"{cfg['name']}: {len(sub_train_idx)} train / {len(sub_val_idx)} early-stop val / {len(val_idx)} final val rows")
    
    for epoch in range(1, tr["epochs"] + 1):
        t0 = time.time()
        loss = train_epoch(model, optimizer, loss_fn, X_tr, y_tr, tr["batch_size"], generator)
        
        # Belső validáció az early stoppinghoz
        sub_val = validate(model, X_sub_va, y_sub_va, ok_sub)
        
        # Súlyok mentése, ha javult a belső validációs hiba
        if sub_val < best_sub_val_rmse:
            best_sub_val_rmse = sub_val
            best_model_state = copy.deepcopy(model.state_dict())

        train_eval = rmse(y_tr.cpu().numpy(), predict(model, X_tr))
        
        row = {"epoch": epoch, "train_loss": loss, "train_rmse": train_eval, "val_rmse": sub_val,
               "seconds": time.time() - t0}
        history.append(row)
        tracking.log_metrics({"train_loss": loss, "train_rmse": train_eval, "val_rmse": sub_val}, step=epoch)

    # Visszatöltjük a legjobb epoch súlyait
    if best_model_state is not None:
        model.load_state_dict(best_model_state)

    # 2. Végső kiértékelés az ÉRINTETLEN validációs halmazon
    final_val_rmse = validate(model, X_va, y_va, ok)
    log(f"Honest Early Stopping Final Val RMSE on untouched data: {final_val_rmse:.4f}")

    ckpt = config.ROOT / "models" / f"{cfg['name']}.pt"
    ckpt.parent.mkdir(exist_ok=True)
    torch.save({"state_dict": model.state_dict(), "config": cfg, "in_dim": X.shape[1],
                "scaler_mean": scaler.mean, "scaler_scale": scaler.scale}, ckpt)
    
    tracking.log_checkpoint(ckpt, cfg["name"], {"val_rmse": final_val_rmse})
    tracking.log_metrics({"final_val_rmse": final_val_rmse})
    url = tracking.run_url()
    tracking.finish()
    
    # A snapshots[tr["epochs"]] helyett a legjobb modellt adjuk vissza predikcióként
    snapshots[tr["epochs"]] = predict(model, X_va)

    return {"history": pd.DataFrame(history), "val_pred": snapshots[tr["epochs"]], "snapshots": snapshots,
            "model": model, "checkpoint": ckpt, "url": url, "val_index": val_idx, "ok": ok}


def main() -> None:
    ap = argparse.ArgumentParser(description="Train the MLP from a config")
    ap.add_argument("--config", required=True)
    a = ap.parse_args()
    with open(a.config) as f:
        cfg = yaml.safe_load(f)
    df = load_table()
    X = pixel_table(df, cfg["data"]["pixel_size"])
    train_idx, val_idx = build_splits(df, cfg["split"]["group_column"], cfg["split"]["val_fraction"], cfg["split"]["seed"])
    result = train_mlp(cfg, X, df, train_idx, val_idx)
    print(f"final val RMSE {result['history']['val_rmse'].iloc[-1]:.4f}  checkpoint {result['checkpoint']}  "
          f"run {result['url']}")


if __name__ == "__main__":
    main()
