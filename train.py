"""Train U-Net + ResNet34 on TLPD. Full training requires --train."""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from src.dataset import TLPDSegDataset
from src.losses import BCEDiceLoss
from src.metrics import MetricMeter
from src.model import build_model, count_finite_grads, param_groups, parameter_count


def load_config(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def make_loader(dataset: TLPDSegDataset, config: dict, shuffle: bool) -> DataLoader:
    train_cfg = config["train"]
    return DataLoader(
        dataset,
        batch_size=train_cfg["batch_size"],
        shuffle=shuffle,
        num_workers=train_cfg["num_workers"],
        pin_memory=train_cfg["pin_memory"],
        drop_last=shuffle,
    )


def build_optimizer(model, config: dict):
    train_cfg = config["train"]
    return torch.optim.AdamW(
        param_groups(model, train_cfg["encoder_lr"], train_cfg["decoder_lr"]),
        weight_decay=train_cfg["weight_decay"],
    )


def build_scheduler(optimizer, config: dict):
    train_cfg = config["train"]
    warmup_epochs = int(train_cfg["warmup_epochs"])
    total_epochs = int(train_cfg["epochs"])
    cosine_epochs = max(1, total_epochs - warmup_epochs)
    warmup = torch.optim.lr_scheduler.LinearLR(
        optimizer, start_factor=0.1, total_iters=warmup_epochs
    )
    cosine = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=cosine_epochs, eta_min=1e-7
    )
    return torch.optim.lr_scheduler.SequentialLR(
        optimizer, schedulers=[warmup, cosine], milestones=[warmup_epochs]
    )


def checkpoint_payload(model, optimizer, scaler, epoch: int, best_dice: float, config: dict) -> dict:
    return {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scaler": scaler.state_dict() if scaler is not None else None,
        "epoch": epoch,
        "best_dice": best_dice,
        "config": config,
    }


def save_checkpoint(path: Path, payload: dict, overwrite: bool) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    target = path
    if path.exists() and not overwrite:
        target = path.with_name(f"sanity_{path.name}")
    torch.save(payload, target)
    return target


def gpu_memory() -> dict[str, float]:
    if not torch.cuda.is_available():
        return {"allocated_mb": 0.0, "reserved_mb": 0.0}
    return {
        "allocated_mb": torch.cuda.memory_allocated() / (1024 ** 2),
        "reserved_mb": torch.cuda.memory_reserved() / (1024 ** 2),
    }


def run_sanity(config: dict, config_path: Path) -> dict:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data_root = ROOT / config["data"]["root"]
    image_size = int(config["data"]["image_size"])
    train_cfg = config["train"]

    dataset = TLPDSegDataset(data_root, split="train", image_size=image_size, augment=True)
    sample = dataset[0]
    image = sample["image"]
    mask = sample["mask"]
    unique = sorted(t.item() for t in mask.unique())

    model = build_model(
        encoder_name=config["model"]["encoder_name"],
        encoder_weights=config["model"]["encoder_weights"],
        in_channels=config["model"]["in_channels"],
        classes=config["model"]["classes"],
    ).to(device)
    criterion = BCEDiceLoss(
        bce_weight=config["loss"]["bce_weight"],
        dice_weight=config["loss"]["dice_weight"],
        smooth=config["loss"]["dice_smooth"],
    )
    optimizer = build_optimizer(model, config)
    scaler = torch.amp.GradScaler("cuda", enabled=bool(train_cfg["amp"]) and device.type == "cuda")

    model.train()
    image_b = image.unsqueeze(0).to(device, non_blocking=True)
    mask_b = mask.unsqueeze(0).to(device, non_blocking=True)
    optimizer.zero_grad(set_to_none=True)
    with torch.amp.autocast("cuda", enabled=bool(train_cfg["amp"]) and device.type == "cuda"):
        logits = model(image_b)
        loss = criterion(logits, mask_b)
    scaler.scale(loss).backward()
    finite_grads = count_finite_grads(model)
    single_loss = float(loss.detach().cpu())
    single_finite = bool(torch.isfinite(loss).item()) and not bool(torch.isnan(loss).item())
    single_out_finite = bool(torch.isfinite(logits).all().item())

    loader = make_loader(dataset, config, shuffle=True)
    accum = int(train_cfg["grad_accum"])
    n_batches = int(train_cfg["sanity_batches"])
    batch_losses = []
    optimizer.zero_grad(set_to_none=True)
    last_backward_grads_finite = False
    for step, batch in enumerate(loader, start=1):
        images = batch["image"].to(device, non_blocking=True)
        masks = batch["mask"].to(device, non_blocking=True)
        with torch.amp.autocast("cuda", enabled=bool(train_cfg["amp"]) and device.type == "cuda"):
            logits_b = model(images)
            batch_loss = criterion(logits_b, masks)
        scaler.scale(batch_loss / accum).backward()
        last_backward_grads_finite = count_finite_grads(model)[2]
        batch_losses.append(float(batch_loss.detach().cpu()))
        if step % accum == 0 or step == n_batches:
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
        if step >= n_batches:
            break

    mem = gpu_memory()
    ckpt_dir = ROOT / config["paths"]["checkpoints"]
    payload = checkpoint_payload(model, optimizer, scaler, epoch=0, best_dice=0.0, config=config)
    last_path = save_checkpoint(ckpt_dir / config["paths"]["last_name"], payload, overwrite=False)
    best_path = save_checkpoint(ckpt_dir / config["paths"]["best_name"], payload, overwrite=False)

    report = {
        "stem": sample["stem"],
        "image_shape": list(image.shape),
        "mask_shape": list(mask.shape),
        "mask_unique": unique,
        "forward_output_shape": list(logits.shape),
        "single_sample_loss": single_loss,
        "single_sample_loss_finite": single_finite,
        "single_sample_output_finite": single_out_finite,
        "single_sample_grads_finite": finite_grads[2],
        "grad_tensors_finite": finite_grads[0],
        "grad_tensors_total": finite_grads[1],
        "batch_losses": batch_losses,
        "batch_losses_finite": all(np.isfinite(v) for v in batch_losses),
        "grads_after_batches_finite": last_backward_grads_finite,
        "parameter_count": parameter_count(model),
        "device": str(device),
        "cuda_available": torch.cuda.is_available(),
        "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "gpu_memory": mem,
        "batch_size": train_cfg["batch_size"],
        "grad_accum": train_cfg["grad_accum"],
        "amp": bool(train_cfg["amp"]),
        "checkpoint_last": str(last_path.relative_to(ROOT)),
        "checkpoint_best": str(best_path.relative_to(ROOT)),
        "checkpoint_last_exists": last_path.is_file(),
        "checkpoint_best_exists": best_path.is_file(),
        "config_path": str(config_path),
    }
    out_path = ROOT / "results" / "tlpd_inspection" / "sanity_check_report.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    print(f"wrote {out_path}")
    return report


@torch.no_grad()
def evaluate_split(model, loader, device, amp: bool, criterion) -> dict[str, float]:
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


def run_train(config: dict) -> None:
    raise SystemExit(
        "Full 40-epoch training is disabled until it is explicitly approved. "
        "Re-run with --sanity only."
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="TLPD U-Net training")
    parser.add_argument("--config", type=Path, default=ROOT / "config.yaml")
    parser.add_argument("--sanity", action="store_true", help="Run the required sanity check only")
    parser.add_argument("--train", action="store_true", help="Run full training (currently blocked)")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config = load_config(args.config)
    set_seed(int(config["seed"]))
    if args.train and not args.sanity:
        run_train(config)
        return 1
    if not args.sanity:
        print("No action. Use --sanity to run the pre-training check. Full training is blocked.")
        return 0
    run_sanity(config, args.config)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
