"""Inference API for a model exported with `python -m thyroid.export`.

Environment:
    MODEL_PATH       exported model file              [models/vit_b16.pt]
    ALLOWED_ORIGINS  comma-separated CORS origins     [http://localhost:3000]

Research demo only, not a medical device.
"""

import io
import os

import torch
import uvicorn
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image, UnidentifiedImageError

from thyroid.data.dataset import build_transform
from thyroid.data.prepare import MAX_SIDE
from thyroid.models import MODELS

MAX_UPLOAD_BYTES = 10 * 1024 * 1024


def load(path: str):
    bundle = torch.load(path, map_location="cpu")
    model = MODELS[bundle["model"]].build(imagenet=False)
    model.load_state_dict(bundle["state_dict"])
    model.eval()
    transform = build_transform(bundle["img_size"], train=False, pretrained=bundle["pretrained"])
    return bundle, model, transform


bundle, model, transform = load(os.environ.get("MODEL_PATH", "models/vit_b16.pt"))

app = FastAPI(title="Thyroid ultrasound classifier")
app.add_middleware(
    CORSMiddleware,
    allow_origins=os.environ.get("ALLOWED_ORIGINS", "http://localhost:3000").split(","),
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)


@app.get("/health")
def health():
    return {
        "status": "ok",
        "model": bundle["model"],
        "fold": bundle["fold"],
        "val_auroc": round(bundle["val_auroc"], 3),
        "test_auroc": round(bundle["test_auroc"], 3),
    }


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    data = await file.read(MAX_UPLOAD_BYTES + 1)
    if len(data) > MAX_UPLOAD_BYTES:
        raise HTTPException(413, "Image larger than 10 MB")
    try:
        img = Image.open(io.BytesIO(data)).convert("L")
    except UnidentifiedImageError:
        raise HTTPException(415, "Not a readable image") from None

    img.thumbnail((MAX_SIDE, MAX_SIDE), Image.Resampling.LANCZOS)  # same as training data
    with torch.no_grad():
        probability = torch.softmax(model(transform(img).unsqueeze(0)), dim=1)[0, 1].item()

    return {
        "label": "malignant" if probability >= bundle["threshold"] else "benign",
        "probability_malignant": round(probability, 4),
        "threshold": round(bundle["threshold"], 4),
        "model": bundle["model"],
    }


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
