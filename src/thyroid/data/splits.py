"""Patient-level splits: a held-out test set plus 5 cross-validation folds on the rest.

    uv run python -m thyroid.data.splits

No patient appears in more than one of test / fold k. Stratification is on source and label
together, so every part keeps the batch1/batch2 and benign/malignant mix.
"""

import csv
import json
from collections import Counter
from pathlib import Path

from sklearn.model_selection import StratifiedGroupKFold

SPLITS = Path("data/splits")
N_FOLDS = 5
TEST_FRACTION_FOLDS = 5  # 1 of 5 patient groups -> ~20% test
SEED = 42


def main() -> None:
    rows = list(csv.DictReader((SPLITS / "manifest.csv").read_text().splitlines()))
    patients = {r["patient_id"]: f"{r['source']}-{r['label']}" for r in rows}
    ids = sorted(patients)
    strata = [patients[p] for p in ids]

    outer = StratifiedGroupKFold(n_splits=TEST_FRACTION_FOLDS, shuffle=True, random_state=SEED)
    cv_idx, test_idx = next(outer.split(ids, strata, groups=ids))
    assignment = {ids[i]: "test" for i in test_idx}

    cv_ids = [ids[i] for i in cv_idx]
    inner = StratifiedGroupKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    for fold, (_, val_idx) in enumerate(inner.split(cv_ids, [patients[p] for p in cv_ids], groups=cv_ids)):
        for i in val_idx:
            assignment[cv_ids[i]] = f"fold{fold}"

    with open(SPLITS / "patients.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["patient_id", "split"])
        writer.writerows(sorted(assignment.items()))

    summary = {}
    for split in ["test", *[f"fold{k}" for k in range(N_FOLDS)]]:
        members = [r for r in rows if assignment[r["patient_id"]] == split]
        summary[split] = {
            "patients": len({r["patient_id"] for r in members}),
            "images": len(members),
            "malignant_share": round(sum(r["label"] == "1" for r in members) / len(members), 3),
            "by_source": dict(Counter(r["source"] for r in members)),
        }
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
