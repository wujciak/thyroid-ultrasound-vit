"""Builds an anonymised, deduplicated image set and manifest from the raw Figshare archives.

    uv run python -m thyroid.data.prepare

Input:  data/raw/figshare/batch{1,2}_image.zip  (scripts/download_data.sh)
Output: data/processed/images/<patient_id>_<n>.png   grayscale, longer side <= MAX_SIDE
        splits/manifest.csv                          one row per kept image, no patient names
        splits/prepare_report.json                   what was dropped and why

Patient names exist only in memory: they are mapped to `b1-XXXX` / `b2-XXXX` ids built from the
dataset's own patient_index, and neither names nor original file names are written anywhere.
"""

import csv
import hashlib
import io
import json
import zipfile
import zlib
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image

RAW = Path("data/raw/figshare")
OUT_IMAGES = Path("data/processed/images")
SPLITS = Path("splits")
MAX_SIDE = 512
NEAR_DUP_BITS = 2  # dHash Hamming distance treated as the same image


@dataclass
class Item:
    patient_id: str
    source: str
    label: int
    member: str


def _read_csv(z: zipfile.ZipFile, member: str, encoding: str) -> list[dict]:
    return list(csv.DictReader(io.StringIO(z.read(member).decode(encoding))))


def batch1_items(z: zipfile.ZipFile, report: Counter) -> list[Item]:
    labels = _read_csv(z, "batch1_image_label.csv", "utf-8-sig")
    images = _read_csv(z, "batch1_image.csv", "utf-8-sig")

    # batch1 links images to labels by patient name only; a name shared by patients with
    # different diagnoses cannot be resolved, so those images are dropped.
    by_name: dict[str, list[dict]] = defaultdict(list)
    for row in labels:
        by_name[row["patient_name"]].append(row)

    items = []
    for row in images:
        records = by_name.get(row["patient_name"])
        if not records:
            report["batch1: image without label"] += 1
            continue
        if len({r["histo_label"] for r in records}) > 1:
            report["batch1: name shared by patients with different labels"] += 1
            continue
        index = min(int(r["patient_index"]) for r in records)
        items.append(Item(f"b1-{index:04d}", "batch1", int(records[0]["histo_label"]),
                          f"dataset/{row['path']}"))
    report["batch1: names shared by several patients (merged into one group)"] = sum(
        len(v) > 1 and len({r["histo_label"] for r in v}) == 1 for v in by_name.values()
    )
    return items


def batch2_items(z: zipfile.ZipFile, report: Counter) -> list[Item]:
    labels = {
        int(r["patient_index"]): int(r["histo_label"])
        for r in _read_csv(z, "batch2_image/batch2_image_label.csv", "utf-8-sig")
    }
    # The CSV lists the ultrasound frames; other files in the patient folders are report scans.
    images = _read_csv(z, "batch2_image/batch2_image.csv", "gb18030")
    report["batch2: files in patient folders not listed as ultrasound frames"] = sum(
        1 for i in z.infolist() if not i.is_dir() and "/thyroid_3_10_month/" in i.filename
        and not i.filename.endswith(".DS_Store")
    ) - len(images)

    items = []
    for row in images:
        index = int(row["path"].split("/")[1])
        if index not in labels:
            report["batch2: image without label"] += 1
            continue
        items.append(Item(f"b2-{index:04d}", "batch2", labels[index], f"batch2_image/{row['path']}"))
    return items


def dhash(img: Image.Image) -> int:
    small = np.asarray(img.resize((9, 8), Image.Resampling.LANCZOS), dtype=np.int16)
    bits = (small[:, 1:] > small[:, :-1]).flatten()
    return int("".join("1" if b else "0" for b in bits), 2)


def load(z: zipfile.ZipFile, member: str) -> Image.Image:
    img = Image.open(io.BytesIO(z.read(member)))
    img = img.convert("L")
    img.thumbnail((MAX_SIDE, MAX_SIDE), Image.Resampling.LANCZOS)
    return img


def near_duplicate_groups(hashes: list[int]) -> list[set[int]]:
    """Union-find over all pairs whose dHash differs by at most NEAR_DUP_BITS."""
    h = np.array(hashes, dtype=np.uint64)
    parent = list(range(len(h)))

    def find(i: int) -> int:
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    for i in range(len(h)):
        x = np.bitwise_xor(h[i + 1:], h[i])
        dist = np.unpackbits(x.view(np.uint8).reshape(-1, 8), axis=1).sum(axis=1)
        for j in np.nonzero(dist <= NEAR_DUP_BITS)[0]:
            parent[find(i)] = find(i + 1 + int(j))

    groups: dict[int, set[int]] = defaultdict(set)
    for i in range(len(h)):
        groups[find(i)].add(i)
    return [g for g in groups.values() if len(g) > 1]


def main() -> None:
    report: Counter = Counter()
    OUT_IMAGES.mkdir(parents=True, exist_ok=True)
    SPLITS.mkdir(exist_ok=True)

    archives = {
        "batch1": zipfile.ZipFile(RAW / "batch1_image.zip"),
        "batch2": zipfile.ZipFile(RAW / "batch2_image.zip", metadata_encoding="gb18030"),
    }
    items = batch1_items(archives["batch1"], report) + batch2_items(archives["batch2"], report)

    rows, hashes = [], []
    per_patient: Counter = Counter()
    for item in items:
        try:
            img = load(archives[item.source], item.member)
        except (KeyError, zlib.error, zipfile.BadZipFile, OSError):
            report[f"{item.source}: missing or corrupt in the published archive"] += 1
            continue
        per_patient[item.patient_id] += 1
        name = f"{item.patient_id}_{per_patient[item.patient_id]:03d}.png"
        img.save(OUT_IMAGES / name)
        rows.append({
            "image": name,
            "patient_id": item.patient_id,
            "source": item.source,
            "label": item.label,
            "sha1": hashlib.sha1(img.tobytes()).hexdigest(),
        })
        hashes.append(dhash(img))

    # Near-duplicates inside one patient are repeated captures: keep one. Across patients the
    # image cannot belong to both, so every copy is dropped.
    drop: set[int] = set()
    for group in near_duplicate_groups(hashes):
        patients = {rows[i]["patient_id"] for i in group}
        if len(patients) > 1:
            drop |= group
            report["near-duplicate across different patients (all copies dropped)"] += len(group)
        else:
            extra = sorted(group)[1:]
            drop |= set(extra)
            report["near-duplicate within one patient (extra copies dropped)"] += len(extra)

    kept = [r for i, r in enumerate(rows) if i not in drop]
    for i in drop:
        (OUT_IMAGES / rows[i]["image"]).unlink()

    with open(SPLITS / "manifest.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(kept[0]))
        writer.writeheader()
        writer.writerows(kept)

    summary = {
        "images": len(kept),
        "patients": len({r["patient_id"] for r in kept}),
        "by_source_and_label": {
            f"{s}/{'malignant' if l else 'benign'}": {
                "images": sum(r["source"] == s and r["label"] == l for r in kept),
                "patients": len({r["patient_id"] for r in kept if r["source"] == s and r["label"] == l}),
            }
            for s in ("batch1", "batch2") for l in (0, 1)
        },
        "dropped": dict(sorted(report.items())),
    }
    (SPLITS / "prepare_report.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
