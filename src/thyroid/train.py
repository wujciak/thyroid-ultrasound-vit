"""Trains one model on one cross-validation fold and evaluates it on that fold and on the test set.

    uv run python -m thyroid.train --model vit_b16 --fold 0
    uv run python -m thyroid.train --model vit_b16 --fold all
    uv run python -m thyroid.train --model cnn_thesis --fold 0 --smoke   # seconds, CPU

Fold k is the validation set (early stopping on image AUROC and the decision threshold);
the other four folds are training data. The test patients are never used for any choice.

Output in runs/<model>/fold<k>/: best.pt, history.csv, predictions.csv, metrics.json
"""

import argparse
import csv
import json
import math
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader

from thyroid.data.dataset import UltrasoundDataset, build_transform, load_rows
from thyroid.metrics import evaluate, thresholds
from thyroid.models import MODELS

N_FOLDS = 5
PATIENCE = 8
RUNS = Path("runs")


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def smoke_subset(rows: list[dict], n: int) -> list[dict]:
    """A few images of each class, so AUROC is defined."""
    return [r for r in rows if r["label"] == "0"][:n] + [r for r in rows if r["label"] == "1"][:n]


@torch.no_grad()
def predict(model: nn.Module, loader: DataLoader, device: torch.device) -> np.ndarray:
    model.eval()
    scores = []
    for x, _ in loader:
        with torch.autocast(device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
            logits = model(x.to(device, non_blocking=True))
        scores.append(torch.softmax(logits.float(), dim=1)[:, 1].cpu())
    return torch.cat(scores).numpy()


def run_fold(model_name: str, fold: int, args: argparse.Namespace) -> dict:
    spec = MODELS[model_name]
    epochs = args.epochs or spec.epochs
    device = torch.device("cpu" if args.smoke or not torch.cuda.is_available() else "cuda")
    seed_everything(args.seed + fold)

    folds = [f"fold{k}" for k in range(N_FOLDS)]
    rows = {
        "train": load_rows([f for f in folds if f != f"fold{fold}"]),
        "val": load_rows([f"fold{fold}"]),
        "test": load_rows(["test"]),
    }
    if args.smoke:
        rows = {k: smoke_subset(v, 8 if k == "train" else 4) for k, v in rows.items()}
        epochs = 1

    loaders = {}
    for split, split_rows in rows.items():
        ds = UltrasoundDataset(split_rows, build_transform(spec.img_size, split == "train", spec.pretrained))
        loaders[split] = DataLoader(
            ds, batch_size=args.batch_size, shuffle=split == "train", drop_last=split == "train",
            num_workers=0 if args.smoke else args.workers, pin_memory=device.type == "cuda",
            persistent_workers=not args.smoke and args.workers > 0,
        )

    model = spec.build().to(device)
    labels = np.array([int(r["label"]) for r in rows["train"]])
    class_weights = torch.tensor(len(labels) / (2 * np.bincount(labels, minlength=2)), dtype=torch.float32)
    criterion = nn.CrossEntropyLoss(weight=class_weights.to(device))
    optimizer = torch.optim.AdamW(model.parameters(), lr=spec.lr, weight_decay=spec.weight_decay)

    steps = epochs * len(loaders["train"])
    warmup = min(len(loaders["train"]), steps // 10)
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lambda s: (s + 1) / warmup if s < warmup else 0.5 * (1 + math.cos(math.pi * (s - warmup) / max(1, steps - warmup))),
    )

    out = RUNS / model_name / f"fold{fold}"
    out.mkdir(parents=True, exist_ok=True)
    y_val = np.array([int(r["label"]) for r in rows["val"]])
    history, best_auroc, stale = [], -1.0, 0

    for epoch in range(1, epochs + 1):
        model.train()
        start, total_loss = time.time(), 0.0
        for x, y in loaders["train"]:
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            with torch.autocast(device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
                loss = criterion(model(x).float(), y)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            total_loss += loss.item()

        val_auroc = float(roc_auc_score(y_val, predict(model, loaders["val"], device)))
        history.append({"epoch": epoch, "train_loss": total_loss / len(loaders["train"]),
                        "val_auroc": val_auroc, "seconds": round(time.time() - start, 1)})
        print(f"[{model_name} fold{fold}] epoch {epoch:3d}  loss {history[-1]['train_loss']:.4f}  "
              f"val AUROC {val_auroc:.4f}  ({history[-1]['seconds']}s)", flush=True)

        if val_auroc > best_auroc:
            best_auroc, stale = val_auroc, 0
            torch.save(model.state_dict(), out / "best.pt")
        else:
            stale += 1
            if stale >= PATIENCE:
                break

    model.load_state_dict(torch.load(out / "best.pt", map_location=device))
    preds = {split: predict(model, loaders[split], device) for split in ("val", "test")}

    def unpack(split):
        return [r["patient_id"] for r in rows[split]], np.array([int(r["label"]) for r in rows[split]]), preds[split]

    thr = thresholds(*unpack("val"))
    metrics = {
        "model": model_name,
        "fold": fold,
        "best_epoch": max(history, key=lambda h: h["val_auroc"])["epoch"],
        "val": evaluate(*unpack("val"), thr),
        "test": evaluate(*unpack("test"), thr),
    }

    (out / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")
    with open(out / "history.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(history[0]))
        writer.writeheader()
        writer.writerows(history)
    with open(out / "predictions.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["split", "image", "patient_id", "label", "score"])
        for split in ("val", "test"):
            for r, s in zip(rows[split], preds[split]):
                writer.writerow([split, r["image"], r["patient_id"], r["label"], f"{s:.6f}"])

    print(f"[{model_name} fold{fold}] test AUROC image {metrics['test']['image']['auroc']:.4f}  "
          f"patient {metrics['test']['patient']['auroc']:.4f}")
    return metrics


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=list(MODELS), required=True)
    ap.add_argument("--fold", default="all", help="0-4 or all")
    ap.add_argument("--epochs", type=int, help="override the model default")
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--smoke", action="store_true", help="tiny CPU run to check the pipeline")
    args = ap.parse_args()
    if args.smoke:
        args.batch_size = 4

    folds = range(N_FOLDS) if args.fold == "all" else [int(args.fold)]
    for fold in folds:
        run_fold(args.model, fold, args)


if __name__ == "__main__":
    main()
