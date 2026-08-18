"""Train U-Net + ResNet34 on TLPD."""

from __future__ import annotations

import argparse
import json
import random
import shutil
import sys
import time
from datetime import datetime
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
from src.visualize import save_prediction_panel, save_training_curves


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


def current_lrs(optimizer) -> list[float]:
    return [float(group["lr"]) for group in optimizer.param_groups]


def log_line(run_dir: Path, message: str) -> None:
    stamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{stamp}] {message}"
    print(line, flush=True)
    with (run_dir / "train.log").open("a", encoding="utf-8") as handle:
        handle.write(line + "\n")


def grad_norm(model) -> float:
    total = 0.0
    for param in model.parameters():
        if param.grad is None:
            continue
        total += float(param.grad.detach().float().norm(2).item() ** 2)
    return total ** 0.5


def make_run_dir(config: dict) -> Path:
    existing_best = ROOT / config["paths"]["checkpoints"] / config["paths"]["best_name"]
    existing_last = ROOT / config["paths"]["checkpoints"] / config["paths"]["last_name"]
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = ROOT / "runs" / f"tlpd_unet_resnet34_{stamp}"
    run_dir.mkdir(parents=True, exist_ok=False)
    (run_dir / "checkpoints").mkdir()
    (run_dir / "predictions").mkdir()
    (run_dir / "curves").mkdir()
    shutil.copy2(ROOT / "config.yaml", run_dir / "config.yaml")
    note = {
        "existing_root_checkpoints": {
            "best_dice.pt": existing_best.is_file(),
            "last.pt": existing_last.is_file(),
        },
        "note": "This run writes only under the run directory and does not overwrite checkpoints/.",
    }
    (run_dir / "run_meta.json").write_text(json.dumps(note, indent=2), encoding="utf-8")
    return run_dir


@torch.no_grad()
def save_qualitative(model, loader, device, amp: bool, out_dir: Path, n: int = 8) -> list[str]:
    model.eval()
    saved = []
    count = 0
    for batch in loader:
        images = batch["image"].to(device, non_blocking=True)
        masks = batch["mask"].to(device, non_blocking=True)
        stems = batch["stem"]
        with torch.amp.autocast("cuda", enabled=amp and device.type == "cuda"):
            logits = model(images)
        for i in range(images.size(0)):
            count += 1
            path = out_dir / f"pred_{count:02d}_{stems[i]}.png"
            save_prediction_panel(images[i], masks[i], logits[i], path, title=stems[i])
            saved.append(str(path))
            if len(saved) >= n:
                return saved
    return saved


def run_train(config: dict) -> int:
    if not torch.cuda.is_available():
        print("ERROR: CUDA is required for the approved training run.")
        return 1

    device = torch.device("cuda")
    torch.cuda.reset_peak_memory_stats()
    run_dir = make_run_dir(config)
    data_root = ROOT / config["data"]["root"]
    image_size = int(config["data"]["image_size"])
    train_cfg = config["train"]
    amp = bool(train_cfg["amp"])
    accum = int(train_cfg["grad_accum"])
    epochs = int(train_cfg["epochs"])
    patience = int(train_cfg["early_stopping_patience"])
    min_delta = float(train_cfg["min_delta"])

    log_line(run_dir, f"Run directory: {run_dir}")
    log_line(run_dir, f"GPU: {torch.cuda.get_device_name(0)}")
    log_line(run_dir, "Existing checkpoints/best_dice.pt and checkpoints/last.pt will not be overwritten.")

    train_ds = TLPDSegDataset(data_root, split="train", image_size=image_size, augment=True)
    val_ds = TLPDSegDataset(data_root, split="val", image_size=image_size, augment=False)
    test_ds = TLPDSegDataset(data_root, split="test", image_size=image_size, augment=False)
    train_loader = make_loader(train_ds, config, shuffle=True)
    val_loader = make_loader(val_ds, config, shuffle=False)
    test_loader = make_loader(test_ds, config, shuffle=False)
    log_line(run_dir, f"samples train={len(train_ds)} val={len(val_ds)} test={len(test_ds)}")

    model = build_model(
        encoder_name=config["model"]["encoder_name"],
        encoder_weights=config["model"]["encoder_weights"],
        in_channels=config["model"]["in_channels"],
        classes=config["model"]["classes"],
    ).to(device)
    log_line(run_dir, f"parameters={parameter_count(model)}")
    criterion = BCEDiceLoss(
        bce_weight=config["loss"]["bce_weight"],
        dice_weight=config["loss"]["dice_weight"],
        smooth=config["loss"]["dice_smooth"],
    )
    optimizer = build_optimizer(model, config)
    scheduler = build_scheduler(optimizer, config)
    scaler = torch.amp.GradScaler("cuda", enabled=amp)

    best_dice = -1.0
    best_epoch = 0
    best_val_metrics: dict[str, float] = {}
    epochs_no_improve = 0
    history: list[dict] = []
    stopped_early = False
    stop_reason = "completed_max_epochs"
    train_start = time.perf_counter()
    peak_reserved = 0.0

    try:
        for epoch in range(1, epochs + 1):
            epoch_start = time.perf_counter()
            model.train()
            running_loss = 0.0
            n_batches = 0
            optimizer.zero_grad(set_to_none=True)
            for step, batch in enumerate(train_loader, start=1):
                images = batch["image"].to(device, non_blocking=True)
                masks = batch["mask"].to(device, non_blocking=True)
                with torch.amp.autocast("cuda", enabled=amp):
                    logits = model(images)
                    loss = criterion(logits, masks)
                if not torch.isfinite(loss):
                    stop_reason = f"non_finite_loss_epoch_{epoch}_step_{step}"
                    log_line(run_dir, f"ERROR: NaN/Inf loss at epoch {epoch} step {step}: {float(loss)}")
                    raise RuntimeError(stop_reason)
                scaler.scale(loss / accum).backward()
                finite, total, all_finite = count_finite_grads(model)
                if not all_finite:
                    stop_reason = f"non_finite_grad_epoch_{epoch}_step_{step}"
                    log_line(run_dir, f"ERROR: NaN/Inf gradients at epoch {epoch} step {step} ({finite}/{total})")
                    raise RuntimeError(stop_reason)
                if step % accum == 0 or step == len(train_loader):
                    scaler.unscale_(optimizer)
                    gn = grad_norm(model)
                    if gn > 50.0:
                        log_line(run_dir, f"WARNING: large grad norm {gn:.3f} at epoch {epoch} step {step}")
                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad(set_to_none=True)
                running_loss += float(loss.detach().cpu())
                n_batches += 1

            train_loss = running_loss / max(n_batches, 1)
            val_metrics = evaluate_split(model, val_loader, device, amp, criterion)
            scheduler.step()
            lrs = current_lrs(optimizer)
            elapsed = time.perf_counter() - epoch_start
            mem = gpu_memory()
            peak_alloc = torch.cuda.max_memory_allocated() / (1024 ** 2)
            peak_reserved = max(peak_reserved, torch.cuda.max_memory_reserved() / (1024 ** 2))
            row = {
                "epoch": epoch,
                "train_loss": train_loss,
                "val_loss": val_metrics["loss"],
                "val_dice": val_metrics["dice"],
                "val_iou": val_metrics["iou"],
                "val_precision": val_metrics["precision"],
                "val_recall": val_metrics["recall"],
                "val_pixel_acc": val_metrics["pixel_acc"],
                "lr_encoder": lrs[0] if lrs else None,
                "lr_decoder": lrs[1] if len(lrs) > 1 else None,
                "epoch_seconds": elapsed,
                "gpu_allocated_mb": mem["allocated_mb"],
                "gpu_reserved_mb": mem["reserved_mb"],
                "gpu_peak_allocated_mb": peak_alloc,
                "gpu_peak_reserved_mb": peak_reserved,
            }
            history.append(row)
            (run_dir / "metrics.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
            log_line(
                run_dir,
                (
                    f"epoch={epoch}/{epochs} train_loss={train_loss:.4f} val_loss={val_metrics['loss']:.4f} "
                    f"val_dice={val_metrics['dice']:.4f} val_iou={val_metrics['iou']:.4f} "
                    f"val_p={val_metrics['precision']:.4f} val_r={val_metrics['recall']:.4f} "
                    f"lr_enc={lrs[0]:.2e} lr_dec={lrs[1]:.2e} time={elapsed:.1f}s "
                    f"mem_alloc={mem['allocated_mb']:.0f}MB mem_res={mem['reserved_mb']:.0f}MB "
                    f"peak_alloc={peak_alloc:.0f}MB"
                ),
            )

            payload = checkpoint_payload(model, optimizer, scaler, epoch, best_dice, config)
            last_path = run_dir / "checkpoints" / "last.pt"
            torch.save(payload, last_path)
            improved = val_metrics["dice"] > best_dice + min_delta
            if improved:
                best_dice = val_metrics["dice"]
                best_epoch = epoch
                best_val_metrics = dict(val_metrics)
                payload["best_dice"] = best_dice
                torch.save(payload, run_dir / "checkpoints" / "best_dice.pt")
                epochs_no_improve = 0
                log_line(run_dir, f"new best val Dice={best_dice:.4f} at epoch {epoch}")
            else:
                epochs_no_improve += 1
                log_line(run_dir, f"no val Dice improvement for {epochs_no_improve}/{patience} epochs")
                if epochs_no_improve >= patience:
                    stopped_early = True
                    stop_reason = f"early_stopping_patience_{patience}"
                    log_line(run_dir, f"Early stopping at epoch {epoch}")
                    break
    except torch.cuda.OutOfMemoryError as exc:
        mem = gpu_memory()
        oom = {
            "error": "cuda_oom",
            "message": str(exc),
            "gpu_memory": mem,
            "peak_allocated_mb": torch.cuda.max_memory_allocated() / (1024 ** 2),
            "peak_reserved_mb": torch.cuda.max_memory_reserved() / (1024 ** 2),
            "history": history,
            "proposed_smallest_change": (
                "Keep 384 resolution and the same model. Reduce batch size from 4 to 2 "
                "and keep gradient accumulation at 4 (effective batch 8), or keep batch 4 "
                "and increase accumulation from 4 to 8 if batch 2 is not required."
            ),
        }
        (run_dir / "oom_report.json").write_text(json.dumps(oom, indent=2), encoding="utf-8")
        log_line(run_dir, f"CUDA OOM: {exc}; allocated={mem['allocated_mb']:.1f}MB reserved={mem['reserved_mb']:.1f}MB")
        return 2
    except Exception as exc:
        log_line(run_dir, f"Training stopped with error: {exc}")
        (run_dir / "error.json").write_text(json.dumps({"error": str(exc), "history": history}, indent=2), encoding="utf-8")
        if "non_finite" in str(exc):
            return 3
        raise

    total_seconds = time.perf_counter() - train_start
    best_path = run_dir / "checkpoints" / "best_dice.pt"
    if not best_path.is_file():
        log_line(run_dir, "ERROR: no best checkpoint was saved.")
        return 4

    payload = torch.load(best_path, map_location=device, weights_only=False)
    model.load_state_dict(payload["model"])
    val_best = evaluate_split(model, val_loader, device, amp, criterion)
    test_metrics = evaluate_split(model, test_loader, device, amp, criterion)
    pred_paths = save_qualitative(model, test_loader, device, amp, run_dir / "predictions", n=8)
    curve_paths = save_training_curves(history, run_dir / "curves")
    final_lrs = current_lrs(optimizer)
    summary = {
        "run_dir": str(run_dir),
        "stopped_early": stopped_early,
        "stop_reason": stop_reason,
        "best_epoch": best_epoch,
        "best_val_dice": best_dice,
        "best_val_metrics_at_save": best_val_metrics,
        "val_metrics_from_best_checkpoint": val_best,
        "test_metrics_from_best_checkpoint": test_metrics,
        "total_training_seconds": total_seconds,
        "peak_gpu_allocated_mb": torch.cuda.max_memory_allocated() / (1024 ** 2),
        "peak_gpu_reserved_mb": torch.cuda.max_memory_reserved() / (1024 ** 2),
        "final_lr_encoder": final_lrs[0] if final_lrs else None,
        "final_lr_decoder": final_lrs[1] if len(final_lrs) > 1 else None,
        "epochs_ran": len(history),
        "paths": {
            "best_checkpoint": str(run_dir / "checkpoints" / "best_dice.pt"),
            "last_checkpoint": str(run_dir / "checkpoints" / "last.pt"),
            "train_log": str(run_dir / "train.log"),
            "config": str(run_dir / "config.yaml"),
            "metrics": str(run_dir / "metrics.json"),
            "evaluation": str(run_dir / "evaluation.json"),
            "predictions": str(run_dir / "predictions"),
            "curves": str(run_dir / "curves"),
        },
        "prediction_files": pred_paths,
        "curve_files": curve_paths,
    }
    (run_dir / "evaluation.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    log_line(run_dir, json.dumps({k: summary[k] for k in [
        "best_epoch", "best_val_dice", "stopped_early", "stop_reason", "total_training_seconds"
    ]}, indent=2))
    log_line(run_dir, f"test metrics: {json.dumps(test_metrics)}")
    log_line(run_dir, "Training run finished.")
    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="TLPD U-Net training")
    parser.add_argument("--config", type=Path, default=ROOT / "config.yaml")
    parser.add_argument("--sanity", action="store_true", help="Run the required sanity check only")
    parser.add_argument("--train", action="store_true", help="Run full training")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config = load_config(args.config)
    set_seed(int(config["seed"]))
    if args.sanity:
        run_sanity(config, args.config)
        return 0
    if args.train:
        return run_train(config)
    print("Use --sanity for the pre-training check or --train to start full training.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
