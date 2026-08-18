"""U-Net with a pretrained ResNet34 encoder."""

from __future__ import annotations

import segmentation_models_pytorch as smp
import torch
from torch import nn


def build_model(
    encoder_name: str = "resnet34",
    encoder_weights: str | None = "imagenet",
    in_channels: int = 3,
    classes: int = 1,
) -> nn.Module:
    return smp.Unet(
        encoder_name=encoder_name,
        encoder_weights=encoder_weights,
        in_channels=in_channels,
        classes=classes,
        activation=None,
    )


def parameter_count(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


def param_groups(model: nn.Module, encoder_lr: float, decoder_lr: float) -> list[dict]:
    encoder_ids = {id(p) for p in model.encoder.parameters()}
    encoder_params = [p for p in model.parameters() if id(p) in encoder_ids]
    other_params = [p for p in model.parameters() if id(p) not in encoder_ids]
    return [
        {"params": encoder_params, "lr": encoder_lr},
        {"params": other_params, "lr": decoder_lr},
    ]


def count_finite_grads(model: nn.Module) -> tuple[int, int, bool]:
    total = 0
    finite = 0
    for param in model.parameters():
        if param.grad is None:
            continue
        total += 1
        if torch.isfinite(param.grad).all():
            finite += 1
    return finite, total, total > 0 and finite == total
