import csv
from pathlib import Path

import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision.transforms import v2

IMAGES = Path("data/processed/images")
SPLITS = Path("data/splits")

# Grayscale statistics: ImageNet luma for pretrained stems, a neutral 0.5/0.25 for scratch models.
PRETRAINED_NORM = ([0.449], [0.226])
SCRATCH_NORM = ([0.5], [0.25])


def _read(path: Path) -> list[dict]:
    return list(csv.DictReader(path.read_text().splitlines()))


def load_rows(splits: list[str]) -> list[dict]:
    """Manifest rows whose patient belongs to one of `splits` (test, fold0..fold4)."""
    assignment = {r["patient_id"]: r["split"] for r in _read(SPLITS / "patients.csv")}
    return [r for r in _read(SPLITS / "manifest.csv") if assignment[r["patient_id"]] in splits]


def build_transform(img_size: int, train: bool, pretrained: bool) -> v2.Compose:
    mean, std = PRETRAINED_NORM if pretrained else SCRATCH_NORM
    ops = [v2.ToImage(), v2.ToDtype(torch.float32, scale=True)]
    if train:
        ops += [
            v2.RandomResizedCrop(img_size, scale=(0.8, 1.0), ratio=(0.9, 1.1), antialias=True),
            v2.RandomHorizontalFlip(),
            v2.RandomRotation(10),
            v2.ColorJitter(brightness=0.2, contrast=0.2),
        ]
    else:
        ops += [v2.Resize((img_size, img_size), antialias=True)]
    return v2.Compose([*ops, v2.Normalize(mean, std)])


class UltrasoundDataset(Dataset):
    def __init__(self, rows: list[dict], transform: v2.Compose):
        self.rows = rows
        self.transform = transform

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, i: int):
        row = self.rows[i]
        img = Image.open(IMAGES / row["image"]).convert("L")
        return self.transform(img), int(row["label"])
