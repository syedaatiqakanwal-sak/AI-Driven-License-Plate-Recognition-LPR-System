"""Validate TLPD split files against existing images and masks."""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.dataset import find_image, read_stems

DATA_ROOT = ROOT / "data" / "evan6007-TLPD"
EXPECTED = {"train": 2122, "val": 455, "test": 455}


def main() -> int:
    images_dir = DATA_ROOT / "images"
    masks_dir = DATA_ROOT / "masks"
    splits_dir = DATA_ROOT / "splits"
    report = {"counts": {}, "missing_images": [], "missing_masks": [], "overlap": {}}
    loaded = {}
    for name in ("train", "val", "test"):
        stems = read_stems(splits_dir / f"{name}.txt")
        loaded[name] = stems
        missing_images = []
        missing_masks = []
        for stem in stems:
            try:
                find_image(images_dir, stem)
            except FileNotFoundError:
                missing_images.append(stem)
            if not (masks_dir / f"{stem}.png").is_file():
                missing_masks.append(stem)
        report["counts"][name] = len(stems)
        report["missing_images"].extend(missing_images)
        report["missing_masks"].extend(missing_masks)

    report["overlap"] = {
        "train_val": sorted(set(loaded["train"]) & set(loaded["val"])),
        "train_test": sorted(set(loaded["train"]) & set(loaded["test"])),
        "val_test": sorted(set(loaded["val"]) & set(loaded["test"])),
    }
    report["total"] = sum(report["counts"].values())
    report["expected"] = EXPECTED
    report["ok"] = (
        report["counts"] == EXPECTED
        and report["total"] == 3032
        and not report["missing_images"]
        and not report["missing_masks"]
        and not any(report["overlap"].values())
    )
    print(json.dumps({
        "ok": report["ok"],
        "counts": report["counts"],
        "total": report["total"],
        "missing_images": len(report["missing_images"]),
        "missing_masks": len(report["missing_masks"]),
        "overlap": {k: len(v) for k, v in report["overlap"].items()},
    }, indent=2))
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
