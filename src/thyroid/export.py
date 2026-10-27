"""Packs one trained fold model into a self-contained file for serving.

    uv run python -m thyroid.export --model vit_b16            # fold with the best validation AUROC
    uv run python -m thyroid.export --model vit_b16 --fold 2

Output: models/<model>.pt holding the weights plus everything the API needs to preprocess
images and turn a score into a decision (input size, normalisation, validation threshold).
"""

import argparse
import json
from pathlib import Path

import torch

from thyroid.models import MODELS

RUNS = Path("runs")
OUT = Path("models")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=list(MODELS), required=True)
    ap.add_argument("--fold", type=int)
    args = ap.parse_args()

    metrics = {int(p.parent.name[4:]): json.loads(p.read_text())
               for p in (RUNS / args.model).glob("fold*/metrics.json")}
    if not metrics:
        raise SystemExit(f"no trained folds in {RUNS / args.model}")
    fold = args.fold if args.fold is not None else max(metrics, key=lambda k: metrics[k]["val"]["image"]["auroc"])

    spec = MODELS[args.model]
    m = metrics[fold]
    OUT.mkdir(exist_ok=True)
    torch.save({
        "model": args.model,
        "fold": fold,
        "img_size": spec.img_size,
        "pretrained": spec.pretrained,
        "threshold": m["val"]["thresholds"]["image"],
        "val_auroc": m["val"]["image"]["auroc"],
        "test_auroc": m["test"]["image"]["auroc"],
        "state_dict": torch.load(RUNS / args.model / f"fold{fold}" / "best.pt", map_location="cpu"),
    }, OUT / f"{args.model}.pt")
    print(f"wrote {OUT / f'{args.model}.pt'} (fold {fold}, val AUROC {m['val']['image']['auroc']:.3f})")


if __name__ == "__main__":
    main()
