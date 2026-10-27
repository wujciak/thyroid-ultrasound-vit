"""Aggregates runs/<model>/fold<k>/ into results/summary.csv and a markdown table.

    uv run python -m thyroid.summarize

For each model: mean ± std over the 5 fold models (validation and test), plus a 5-fold
ensemble on the test set (mean of the fold models' scores per image).
"""

import csv
import json
from pathlib import Path

import numpy as np

from thyroid.metrics import evaluate, thresholds
from thyroid.models import MODELS

RUNS = Path("runs")
RESULTS = Path("results")
LEVELS = ("image", "patient")
COLUMNS = ("auroc", "auprc", "sensitivity", "specificity", "balanced_accuracy")


def ensemble(model: str) -> dict | None:
    """Averages test scores of all fold models; the threshold comes from pooled validation scores."""
    files = sorted((RUNS / model).glob("fold*/predictions.csv"))
    if not files:
        return None
    test_scores, val = {}, {"ids": [], "y": [], "s": []}
    for f in files:
        for r in csv.DictReader(f.read_text().splitlines()):
            if r["split"] == "test":
                test_scores.setdefault(r["image"], [r["patient_id"], int(r["label"]), []])[2].append(float(r["score"]))
            else:
                val["ids"].append(r["patient_id"])
                val["y"].append(int(r["label"]))
                val["s"].append(float(r["score"]))
    thr = thresholds(val["ids"], np.array(val["y"]), np.array(val["s"]))
    ids = [v[0] for v in test_scores.values()]
    y = np.array([v[1] for v in test_scores.values()])
    s = np.array([np.mean(v[2]) for v in test_scores.values()])
    return evaluate(ids, y, s, thr)


def main() -> None:
    RESULTS.mkdir(exist_ok=True)
    rows = []
    for model in MODELS:
        folds = [json.loads(p.read_text()) for p in sorted((RUNS / model).glob("fold*/metrics.json"))]
        if not folds:
            continue
        for split in ("val", "test"):
            for level in LEVELS:
                row = {"model": model, "eval": f"{split} (fold mean, n={len(folds)})", "level": level}
                for c in COLUMNS:
                    v = np.array([f[split][level][c] for f in folds])
                    row[c] = f"{v.mean():.3f} ± {v.std(ddof=1) if len(v) > 1 else 0:.3f}"
                rows.append(row)
        ens = ensemble(model)
        for level in LEVELS:
            rows.append({"model": model, "eval": "test (5-fold ensemble)", "level": level,
                         **{c: f"{ens[level][c]:.3f}" for c in COLUMNS}})

    with open(RESULTS / "summary.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    header = "| " + " | ".join(rows[0]) + " |"
    print(header)
    print("|" + "---|" * len(rows[0]))
    for r in rows:
        print("| " + " | ".join(str(v) for v in r.values()) + " |")


if __name__ == "__main__":
    main()
