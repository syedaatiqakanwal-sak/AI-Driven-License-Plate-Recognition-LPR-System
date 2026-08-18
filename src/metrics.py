"""Binary segmentation metrics. Dice is the primary selection metric."""

from __future__ import annotations

import torch


def _binarize(logits: torch.Tensor, threshold: float = 0.5) -> torch.Tensor:
    return (torch.sigmoid(logits) > threshold).to(dtype=torch.float32)


def confusion_counts(logits: torch.Tensor, targets: torch.Tensor, threshold: float = 0.5) -> dict[str, torch.Tensor]:
    preds = _binarize(logits, threshold)
    targets = (targets > 0.5).to(dtype=torch.float32)
    tp = (preds * targets).sum()
    fp = (preds * (1.0 - targets)).sum()
    fn = ((1.0 - preds) * targets).sum()
    tn = ((1.0 - preds) * (1.0 - targets)).sum()
    return {"tp": tp, "fp": fp, "fn": fn, "tn": tn}


def metrics_from_counts(tp: float, fp: float, fn: float, tn: float, eps: float = 1e-7) -> dict[str, float]:
    dice = (2.0 * tp + eps) / (2.0 * tp + fp + fn + eps)
    iou = (tp + eps) / (tp + fp + fn + eps)
    precision = (tp + eps) / (tp + fp + eps)
    recall = (tp + eps) / (tp + fn + eps)
    pixel_acc = (tp + tn) / (tp + tn + fp + fn + eps)
    return {
        "dice": float(dice),
        "iou": float(iou),
        "precision": float(precision),
        "recall": float(recall),
        "pixel_acc": float(pixel_acc),
    }


class MetricMeter:
    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        self.tp = 0.0
        self.fp = 0.0
        self.fn = 0.0
        self.tn = 0.0
        self.per_image_dice: list[float] = []

    @torch.no_grad()
    def update(self, logits: torch.Tensor, targets: torch.Tensor) -> None:
        counts = confusion_counts(logits, targets)
        self.tp += float(counts["tp"].item())
        self.fp += float(counts["fp"].item())
        self.fn += float(counts["fn"].item())
        self.tn += float(counts["tn"].item())
        preds = _binarize(logits)
        gts = (targets > 0.5).to(dtype=torch.float32)
        for pred, gt in zip(preds, gts, strict=True):
            tp = float((pred * gt).sum().item())
            fp = float((pred * (1.0 - gt)).sum().item())
            fn = float(((1.0 - pred) * gt).sum().item())
            if tp == 0.0 and fp == 0.0 and fn == 0.0:
                self.per_image_dice.append(1.0)
            else:
                self.per_image_dice.append((2.0 * tp) / (2.0 * tp + fp + fn + 1e-7))

    def compute(self) -> dict[str, float]:
        out = metrics_from_counts(self.tp, self.fp, self.fn, self.tn)
        if self.per_image_dice:
            out["dice_macro"] = float(sum(self.per_image_dice) / len(self.per_image_dice))
        else:
            out["dice_macro"] = 0.0
        return out
