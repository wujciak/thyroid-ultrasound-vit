"""Model registry: architecture plus the input size and training defaults it is trained with.

Thesis models are trained from scratch on 256 px grayscale input, as in the thesis.
Pretrained models use ImageNet weights at 224 px; timm folds the RGB stem into one channel.
"""

from dataclasses import dataclass
from typing import Callable

import timm
import torch.nn as nn

from thyroid.models.cnn_thesis import ThesisCNN
from thyroid.models.vit_thesis import VisionTransformers


@dataclass(frozen=True)
class Spec:
    build: Callable[[], nn.Module]
    img_size: int
    lr: float
    epochs: int
    weight_decay: float
    pretrained: bool


def _timm(name: str) -> Callable[[], nn.Module]:
    return lambda: timm.create_model(name, pretrained=True, in_chans=1, num_classes=2)


MODELS: dict[str, Spec] = {
    "cnn_thesis": Spec(lambda: ThesisCNN(img_size=256), 256, lr=1e-3, epochs=40,
                       weight_decay=1e-4, pretrained=False),
    "vit_thesis": Spec(
        lambda: VisionTransformers(img_size=256, patch_size=16, in_chans=1, n_classes=2,
                                   embed_dim=768, depth=12, n_heads=12, p=0.1, attn_p=0.1),
        256, lr=1e-4, epochs=60, weight_decay=0.05, pretrained=False,
    ),
    "resnet50": Spec(_timm("resnet50.a1_in1k"), 224, lr=1e-4, epochs=25,
                     weight_decay=1e-4, pretrained=True),
    "vit_b16": Spec(_timm("vit_base_patch16_224.augreg2_in21k_ft_in1k"), 224, lr=3e-5, epochs=25,
                    weight_decay=0.05, pretrained=True),
}
