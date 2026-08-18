"""Evaluate a TLPD segmentation checkpoint on val/test splits."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
import yaml
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from src.dataset import TLPDSegDataset
from src.losses import BCEDiceLoss
from src.metrics import MetricMeter
from src.model import build_model


def load_config(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


@torch.no_grad()
def run_split(model, loader, device, amp: bool, criterion) -> dict[str, float]:
    model.eval()
    meter = MetricMeter()
    losses = []
    for batch in loader:
        images = batch["image"].to(device, non_blocking=True)
        masks = batch["mask"].to(device, non_blocking=True)
        with torch.amp.autocast("cuda", enabled=amp and device.type == "cuda"):
            logits = model(images)
            loss = criterion(logits, masks)
        losses.append(float(loss.detach().cpu()))
        meter.update(logits.float(), masks)
    metrics = meter.compute()
    metrics["loss"] = float(sum(losses) / max(len(losses), 1))
    return metrics


def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate TLPD U-Net checkpoint")
    parser.add_argument("--config", type=Path, default=ROOT / "config.yaml")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--split", choices=["val", "test", "both"], default="both")
    args = parser.parse_args()

    config = load_config(args.config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data_root = ROOT / config["data"]["root"]
    image_size = int(config["data"]["image_size"])
    amp = bool(config["train"]["amp"])

    model = build_model(
        encoder_name=config["model"]["encoder_name"],
        encoder_weights=None,
        in_channels=config["model"]["in_channels"],
        classes=config["model"]["classes"],
    )
    payload = torch.load(args.checkpoint, map_location=device, weights_only=False)
    state = payload["model"] if isinstance(payload, dict) and "model" in payload else payload
    model.load_state_dict(state)
    model.to(device)

    criterion = BCEDiceLoss(
        bce_weight=config["loss"]["bce_weight"],
        dice_weight=config["loss"]["dice_weight"],
        smooth=config["loss"]["dice_smooth"],
    )

    splits = ["val", "test"] if args.split == "both" else [args.split]
    results = {}
    for split in splits:
        dataset = TLPDSegDataset(data_root, split=split, image_size=image_size, augment=False)
        loader = DataLoader(
            dataset,
            batch_size=config["train"]["batch_size"],
            shuffle=False,
            num_workers=config["train"]["num_workers"],
            pin_memory=config["train"]["pin_memory"],
        )
        results[split] = run_split(model, loader, device, amp, criterion)
        print(split, json.dumps(results[split], indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
