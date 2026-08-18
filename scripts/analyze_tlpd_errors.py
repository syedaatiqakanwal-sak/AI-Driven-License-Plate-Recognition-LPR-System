"""Per-image error analysis for the TLPD best checkpoint. Does not retrain."""

from __future__ import annotations

import csv
import json
import statistics
import sys
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.dataset import TLPDSegDataset, find_image
from src.metrics import metrics_from_counts
from src.model import build_model
from src.visualize import denormalize, mask_to_image

CKPT = ROOT / "runs" / "tlpd_unet_resnet34_20260818_171435" / "checkpoints" / "best_dice.pt"
OUT_DIR = ROOT / "results" / "tlpd_error_analysis"
WORST_DIR = OUT_DIR / "worst_cases"
EPS = 1e-7


def image_metrics(pred: np.ndarray, gt: np.ndarray) -> dict[str, float]:
    pred_b = pred > 0.5
    gt_b = gt > 0.5
    tp = float(np.logical_and(pred_b, gt_b).sum())
    fp = float(np.logical_and(pred_b, ~gt_b).sum())
    fn = float(np.logical_and(~pred_b, gt_b).sum())
    tn = float(np.logical_and(~pred_b, ~gt_b).sum())
    return metrics_from_counts(tp, fp, fn, tn)


def original_mask_stats(root: Path, stem: str) -> dict:
    mask_path = root / "masks" / f"{stem}.png"
    image_path = find_image(root / "images", stem)
    mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
    image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if mask is None or image is None:
        raise RuntimeError(f"Failed to read original files for {stem}")
    fg = int((mask > 0).sum())
    h, w = mask.shape[:2]
    area = int(h * w)
    return {
        "orig_height": h,
        "orig_width": w,
        "orig_area": area,
        "orig_fg_pixels": fg,
        "orig_fg_pct": 100.0 * fg / max(area, 1),
        "aspect_ratio": w / max(h, 1),
    }


def json_keys_sample(labels_dir: Path, stems: list[str], n: int = 40) -> set[str]:
    keys: set[str] = set()
    for stem in stems[:n]:
        path = labels_dir / f"{stem}.json"
        if not path.is_file():
            continue
        data = json.loads(path.read_text(encoding="utf-8"))
        keys.update(data.keys())
        if data.get("shapes"):
            keys.update(f"shapes[0].{k}" for k in data["shapes"][0].keys())
    return keys


def gt_pred_overlay(rgb: np.ndarray, gt: np.ndarray, pred: np.ndarray) -> np.ndarray:
    out = rgb.copy()
    gt_b = gt > 0.5
    pred_b = pred > 0.5
    tp = np.logical_and(gt_b, pred_b)
    fp = np.logical_and(pred_b, ~gt_b)
    fn = np.logical_and(gt_b, ~pred_b)
    alpha = 0.5
    out[tp] = (1 - alpha) * out[tp] + alpha * np.array([1.0, 1.0, 0.15])
    out[fp] = (1 - alpha) * out[fp] + alpha * np.array([1.0, 0.15, 0.1])
    out[fn] = (1 - alpha) * out[fn] + alpha * np.array([0.1, 0.85, 0.2])
    return np.clip(out, 0.0, 1.0)


def save_error_panel(image, gt_mask, pred_mask, out_path: Path, title: str) -> None:
    rgb = denormalize(image)
    gt = mask_to_image(gt_mask)
    pred = mask_to_image(pred_mask)
    panels = [
        rgb,
        np.stack([gt, gt, gt], axis=-1),
        np.stack([pred, pred, pred], axis=-1),
        gt_pred_overlay(rgb, gt, pred),
    ]
    labels = ["1. Original", "2. Ground-truth mask", "3. Predicted mask", "4. GT vs pred overlay"]
    fig, axes = plt.subplots(1, 4, figsize=(16, 4.3))
    for ax, panel, label in zip(axes, panels, labels):
        ax.imshow(panel)
        ax.set_title(label, fontsize=10)
        ax.axis("off")
    fig.suptitle(title, fontsize=11)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def tertile_label(value: float, q1: float, q2: float) -> str:
    if value <= q1:
        return "small"
    if value <= q2:
        return "medium"
    return "large"


def summarize(values: list[float]) -> dict[str, float]:
    return {
        "n": len(values),
        "mean": float(statistics.fmean(values)),
        "median": float(statistics.median(values)),
        "stdev": float(statistics.pstdev(values)) if len(values) > 1 else 0.0,
        "min": float(min(values)),
        "max": float(max(values)),
    }


def group_metrics(rows: list[dict], key: str) -> dict[str, dict]:
    buckets: dict[str, list[dict]] = {"small": [], "medium": [], "large": []}
    for row in rows:
        buckets[row[key]].append(row)
    out = {}
    for name, items in buckets.items():
        if not items:
            out[name] = {"n": 0}
            continue
        out[name] = {
            "n": len(items),
            "dice_mean": float(statistics.fmean(r["dice"] for r in items)),
            "iou_mean": float(statistics.fmean(r["iou"] for r in items)),
            "precision_mean": float(statistics.fmean(r["precision"] for r in items)),
            "recall_mean": float(statistics.fmean(r["recall"] for r in items)),
            "fg_pct_mean": float(statistics.fmean(r["orig_fg_pct"] for r in items)),
            "fg_pixels_mean": float(statistics.fmean(r["orig_fg_pixels"] for r in items)),
        }
    return out


def main() -> int:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    payload = torch.load(CKPT, map_location=device, weights_only=False)
    config = payload.get("config")
    if config is None:
        with (ROOT / "config.yaml").open("r", encoding="utf-8") as handle:
            config = yaml.safe_load(handle)

    verify = {
        "checkpoint": str(CKPT),
        "epoch": payload.get("epoch"),
        "best_dice": payload.get("best_dice"),
        "has_model_state": "model" in payload,
        "encoder": config["model"]["encoder_name"],
        "encoder_weights": config["model"]["encoder_weights"],
        "in_channels": config["model"]["in_channels"],
        "classes": config["model"]["classes"],
        "image_size": config["data"]["image_size"],
        "dataset_root": config["data"]["root"],
        "split_files": {
            "train": "data/evan6007-TLPD/splits/train.txt",
            "val": "data/evan6007-TLPD/splits/val.txt",
            "test": "data/evan6007-TLPD/splits/test.txt",
        },
        "is_epoch_28": payload.get("epoch") == 28,
    }
    print(json.dumps(verify, indent=2))
    if payload.get("epoch") != 28:
        print("WARNING: checkpoint epoch is not 28")

    data_root = ROOT / config["data"]["root"]
    dataset = TLPDSegDataset(data_root, split="test", image_size=int(config["data"]["image_size"]), augment=False)
    loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0, pin_memory=True)

    model = build_model(
        encoder_name=config["model"]["encoder_name"],
        encoder_weights=None,
        in_channels=config["model"]["in_channels"],
        classes=config["model"]["classes"],
    )
    model.load_state_dict(payload["model"])
    model.to(device)
    model.eval()

    json_keys = sorted(json_keys_sample(data_root / "labels", dataset.stems))
    country_available = any("country" in k.lower() for k in json_keys)

    rows = []
    tensors = {}
    with torch.no_grad():
        for batch in loader:
            stem = batch["stem"][0]
            image = batch["image"].to(device, non_blocking=True)
            mask = batch["mask"].to(device, non_blocking=True)
            with torch.amp.autocast("cuda", enabled=device.type == "cuda"):
                logits = model(image)
            pred = (torch.sigmoid(logits.float()) > 0.5).float()
            metrics = image_metrics(pred[0, 0].cpu().numpy(), mask[0, 0].cpu().numpy())
            orig = original_mask_stats(data_root, stem)
            letter_fg = float((mask[0, 0] > 0.5).sum().item())
            row = {
                "stem": stem,
                **metrics,
                **orig,
                "letterbox_fg_pixels": letter_fg,
                "letterbox_fg_pct": 100.0 * letter_fg / (384.0 * 384.0),
            }
            rows.append(row)
            tensors[stem] = {
                "image": image[0].detach().cpu(),
                "gt": mask[0].detach().cpu(),
                "pred": pred[0].detach().cpu(),
            }

    if len(rows) != 455:
        print(f"WARNING: expected 455 test images, got {len(rows)}")

    rows_by_dice = sorted(rows, key=lambda r: r["dice"])
    fg_pixels = [r["orig_fg_pixels"] for r in rows]
    q1, q2 = np.quantile(fg_pixels, [1 / 3, 2 / 3]).tolist()
    for row in rows:
        row["size_group"] = tertile_label(row["orig_fg_pixels"], q1, q2)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    csv_path = OUT_DIR / "test_per_image_metrics.csv"
    fieldnames = [
        "stem", "dice", "iou", "precision", "recall", "pixel_acc",
        "orig_width", "orig_height", "orig_area", "orig_fg_pixels", "orig_fg_pct",
        "aspect_ratio", "letterbox_fg_pixels", "letterbox_fg_pct", "size_group",
    ]
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows_by_dice:
            writer.writerow(row)

    def top_n(metric: str, n: int = 20, reverse: bool = False) -> list[dict]:
        return sorted(rows, key=lambda r: r[metric], reverse=reverse)[:n]

    worst = {
        "dice": top_n("dice", 20),
        "iou": top_n("iou", 20),
        "recall": top_n("recall", 20),
        "precision": top_n("precision", 20),
    }
    WORST_DIR.mkdir(parents=True, exist_ok=True)
    saved = {}
    for metric, items in worst.items():
        paths = []
        for rank, row in enumerate(items, start=1):
            stem = row["stem"]
            title = (
                f"{metric} rank {rank}/20  {stem}  "
                f"Dice={row['dice']:.4f} IoU={row['iou']:.4f} "
                f"P={row['precision']:.4f} R={row['recall']:.4f} "
                f"fg={row['orig_fg_pct']:.1f}%"
            )
            path = WORST_DIR / f"{metric}_{rank:02d}_{stem}.png"
            pack = tensors[stem]
            save_error_panel(pack["image"], pack["gt"], pack["pred"], path, title)
            paths.append(str(path.relative_to(ROOT)))
        saved[metric] = paths

    micro_tp = micro_fp = micro_fn = micro_tn = 0.0
    for row in rows:
        # reconstruct from precision/recall is messy; recompute micro from CSV fields isn't stored.
        pass

    dice_s = summarize([r["dice"] for r in rows])
    iou_s = summarize([r["iou"] for r in rows])
    prec_s = summarize([r["precision"] for r in rows])
    rec_s = summarize([r["recall"] for r in rows])
    size_groups = group_metrics(rows, "size_group")

    report = {
        "checkpoint_verification": verify,
        "test_count": len(rows),
        "json_keys_sampled": json_keys,
        "country_available": country_available,
        "country_note": (
            "No country field exists in TLPD LabelMe JSON. TLPD is a Taiwan-only dataset; "
            "country-wise metrics were not computed because labels were not present."
        ),
        "per_image_dice": dice_s,
        "per_image_iou": iou_s,
        "per_image_precision": prec_s,
        "per_image_recall": rec_s,
        "size_tertiles_orig_fg_pixels": {"q1": q1, "q2": q2},
        "size_groups": size_groups,
        "worst20_dice": [
            {k: r[k] for k in ["stem", "dice", "iou", "precision", "recall", "orig_fg_pct", "orig_width", "orig_height", "size_group"]}
            for r in worst["dice"]
        ],
        "worst20_iou_stems": [r["stem"] for r in worst["iou"]],
        "worst20_recall_stems": [r["stem"] for r in worst["recall"]],
        "worst20_precision_stems": [r["stem"] for r in worst["precision"]],
        "visualization_paths": saved,
        "csv": str(csv_path.relative_to(ROOT)),
    }
    (OUT_DIR / "error_analysis_summary.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({
        "epoch": verify["epoch"],
        "is_epoch_28": verify["is_epoch_28"],
        "best_dice": verify["best_dice"],
        "test_count": len(rows),
        "dice": dice_s,
        "iou": iou_s,
        "size_groups": size_groups,
        "country_available": country_available,
        "csv": str(csv_path),
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
