"""Grad-CAM for the timm models: where in the frame the malignancy score comes from."""

from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from thyroid.models import MODELS

RUNS = Path("runs")


def load_fold(model_name: str, fold: int) -> nn.Module:
    model = MODELS[model_name].build(imagenet=False)
    path = RUNS / model_name / f"fold{fold}" / "best.pt"
    model.load_state_dict(torch.load(path, map_location="cpu"))
    return model.eval()


def _target_layer(model: nn.Module) -> nn.Module:
    if hasattr(model, "blocks"):  # timm ViT: tokens entering the last attention block
        return model.blocks[-1].norm1
    return model.layer4  # timm ResNet


def gradcam(model: nn.Module, x: torch.Tensor, target: int = 1) -> np.ndarray:
    """Grad-CAM of the `target` logit (1 = malignant), upsampled to the input and scaled to [0, 1]."""
    acts, grads = {}, {}
    layer = _target_layer(model)
    h1 = layer.register_forward_hook(lambda m, i, o: acts.update(a=o))
    h2 = layer.register_full_backward_hook(lambda m, gi, go: grads.update(g=go[0]))
    try:
        model.zero_grad()
        model(x)[:, target].sum().backward()
    finally:
        h1.remove()
        h2.remove()

    a, g = acts["a"].detach(), grads["g"].detach()
    if a.dim() == 3:  # (B, 1 + N, C) tokens -> (B, C, h, w) without the class token
        side = int((a.shape[1] - 1) ** 0.5)
        a, g = (t[:, 1:].transpose(1, 2).reshape(t.shape[0], -1, side, side) for t in (a, g))
    cam = F.relu((g.mean(dim=(2, 3), keepdim=True) * a).sum(1, keepdim=True))
    cam = F.interpolate(cam, size=x.shape[-2:], mode="bilinear", align_corners=False)[:, 0]
    cam = cam / cam.flatten(1).amax(1).clamp_min(1e-8)[:, None, None]
    return cam.numpy()


@torch.no_grad()
def scores(model: nn.Module, x: torch.Tensor, batch_size: int = 32) -> np.ndarray:
    """Malignancy probability per image."""
    return torch.cat([torch.softmax(model(b), 1)[:, 1] for b in x.split(batch_size)]).numpy()
