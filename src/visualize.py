"""Prediction panels and training curves for a TLPD run."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

from src.dataset import IMAGENET_MEAN, IMAGENET_STD


def denormalize(image: torch.Tensor) -> np.ndarray:
    arr = image.detach().cpu().float().numpy()
    mean = np.array(IMAGENET_MEAN, dtype=np.float32)[:, None, None]
    std = np.array(IMAGENET_STD, dtype=np.float32)[:, None, None]
    rgb = np.clip(arr * std + mean, 0.0, 1.0)
    return np.transpose(rgb, (1, 2, 0))


def mask_to_image(mask: torch.Tensor) -> np.ndarray:
    arr = mask.detach().cpu().float().numpy()
    if arr.ndim == 3:
        arr = arr[0]
    return (arr > 0.5).astype(np.float32)


def overlay(image_rgb: np.ndarray, mask: np.ndarray, color=(1.0, 0.15, 0.1), alpha=0.45) -> np.ndarray:
    out = image_rgb.copy()
    fg = mask > 0.5
    color_arr = np.array(color, dtype=np.float32)
    out[fg] = (1.0 - alpha) * out[fg] + alpha * color_arr
    return np.clip(out, 0.0, 1.0)


def save_prediction_panel(
    image: torch.Tensor,
    gt_mask: torch.Tensor,
    pred_logits: torch.Tensor,
    out_path: Path,
    title: str,
) -> None:
    rgb = denormalize(image)
    gt = mask_to_image(gt_mask)
    pred = mask_to_image(torch.sigmoid(pred_logits) > 0.5)
    panels = [rgb, np.stack([gt, gt, gt], axis=-1), np.stack([pred, pred, pred], axis=-1), overlay(rgb, pred)]
    labels = ["1. Original", "2. Ground-truth mask", "3. Predicted mask", "4. Overlay"]
    fig, axes = plt.subplots(1, 4, figsize=(16, 4.2))
    for ax, panel, label in zip(axes, panels, labels):
        ax.imshow(panel)
        ax.set_title(label)
        ax.axis("off")
    fig.suptitle(title, fontsize=11)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def save_training_curves(history: list[dict], out_dir: Path) -> dict[str, str]:
    out_dir.mkdir(parents=True, exist_ok=True)
    epochs = [row["epoch"] for row in history]
    series = {
        "train_loss": [row["train_loss"] for row in history],
        "val_loss": [row["val_loss"] for row in history],
        "val_dice": [row["val_dice"] for row in history],
        "val_iou": [row["val_iou"] for row in history],
    }
    titles = {
        "train_loss": "Training loss vs epoch",
        "val_loss": "Validation loss vs epoch",
        "val_dice": "Validation Dice vs epoch",
        "val_iou": "Validation IoU vs epoch",
    }
    paths = {}
    for key, values in series.items():
        fig, ax = plt.subplots(figsize=(8, 4.5))
        ax.plot(epochs, values, marker="o", linewidth=1.5)
        ax.set_xlabel("Epoch")
        ax.set_ylabel(key)
        ax.set_title(titles[key])
        ax.grid(True, alpha=0.3)
        path = out_dir / f"{key}.png"
        fig.tight_layout()
        fig.savefig(path, dpi=120)
        plt.close(fig)
        paths[key] = str(path)

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    plot_items = [
        (axes[0, 0], "train_loss", "Training loss"),
        (axes[0, 1], "val_loss", "Validation loss"),
        (axes[1, 0], "val_dice", "Validation Dice"),
        (axes[1, 1], "val_iou", "Validation IoU"),
    ]
    for ax, key, title in plot_items:
        ax.plot(epochs, series[key], marker="o", linewidth=1.5)
        ax.set_title(title)
        ax.set_xlabel("Epoch")
        ax.grid(True, alpha=0.3)
    combined = out_dir / "training_curves.png"
    fig.tight_layout()
    fig.savefig(combined, dpi=130)
    plt.close(fig)
    paths["combined"] = str(combined)
    return paths
