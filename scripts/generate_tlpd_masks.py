"""Convert TLPD LabelMe polygons into binary segmentation masks.

Match image/JSON pairs by JSON filename stem (not JSON imagePath).
Fill the annotated 4-point polygon exactly; do not expand coordinates.
Mask values: 0 = background, 1 = annotated region.
"""

from __future__ import annotations

import json
import random
import statistics
import sys
from collections import Counter
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]
DATASET_DIR = ROOT / "data" / "evan6007-TLPD"
IMAGES_DIR = DATASET_DIR / "images"
LABELS_DIR = DATASET_DIR / "labels"
MASKS_DIR = DATASET_DIR / "masks"
INSPECTION_DIR = ROOT / "results" / "tlpd_inspection"
REPORT_PATH = INSPECTION_DIR / "mask_validation_report.json"
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
EXPECTED_COUNT = 3032
SPLIT_SEED = 42
TRAIN_RATIO = 0.70
VAL_RATIO = 0.15


def index_images(images_dir: Path) -> tuple[dict[str, Path], list[str]]:
    by_stem: dict[str, Path] = {}
    issues: list[str] = []
    for path in images_dir.iterdir():
        if not path.is_file() or path.suffix.lower() not in IMAGE_EXTS:
            continue
        if path.stem in by_stem:
            issues.append(f"duplicate image stem: {path.stem}")
            continue
        by_stem[path.stem] = path
    return by_stem, issues


def load_label(json_path: Path) -> dict:
    with json_path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def validate_annotation(stem: str, data: dict) -> tuple[list[tuple[float, float]], int, int, list[str]]:
    issues: list[str] = []
    shapes = data.get("shapes")
    if not isinstance(shapes, list) or len(shapes) != 1:
        issues.append(f"{stem}: expected exactly 1 shape, got {0 if not isinstance(shapes, list) else len(shapes)}")
        return [], 0, 0, issues

    shape = shapes[0]
    shape_type = shape.get("shape_type")
    points_raw = shape.get("points")
    if shape_type != "polygon":
        issues.append(f"{stem}: expected shape_type polygon, got {shape_type!r}")
    if not isinstance(points_raw, list) or len(points_raw) != 4:
        issues.append(
            f"{stem}: expected exactly 4 polygon points, got {0 if not isinstance(points_raw, list) else len(points_raw)}"
        )
        return [], 0, 0, issues

    points: list[tuple[float, float]] = []
    for point in points_raw:
        if not isinstance(point, (list, tuple)) or len(point) != 2:
            issues.append(f"{stem}: invalid polygon point {point!r}")
            return [], 0, 0, issues
        points.append((float(point[0]), float(point[1])))

    width = data.get("imageWidth")
    height = data.get("imageHeight")
    if not isinstance(width, int) or not isinstance(height, int) or width <= 0 or height <= 0:
        issues.append(f"{stem}: invalid imageWidth/imageHeight ({width}, {height})")
        return points, 0, 0, issues
    return points, width, height, issues


def polygon_to_mask(width: int, height: int, points: list[tuple[float, float]]) -> Image.Image:
    mask = Image.new("L", (width, height), 0)
    draw = ImageDraw.Draw(mask)
    draw.polygon(points, fill=1)
    return mask


def save_mask(path: Path, mask: Image.Image) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    mask.save(path, format="PNG")


def draw_polygon_on_image(image: Image.Image, points: list[tuple[float, float]]) -> Image.Image:
    rgb = image.convert("RGB")
    overlay = rgb.copy()
    draw = ImageDraw.Draw(overlay)
    draw.polygon(points, outline=(255, 0, 0), width=max(1, min(rgb.size) // 120 or 1))
    for x, y in points:
        r = max(2, min(rgb.size) // 80)
        draw.ellipse((x - r, y - r, x + r, y + r), fill=(0, 255, 0))
    return overlay


def make_overlay(image: Image.Image, mask: Image.Image) -> Image.Image:
    rgb = np.asarray(image.convert("RGB"), dtype=np.uint8)
    mask_arr = np.asarray(mask, dtype=np.uint8)
    color = np.zeros_like(rgb)
    color[..., 0] = 255
    alpha = 0.45
    blended = rgb.copy()
    fg = mask_arr > 0
    blended[fg] = np.clip(
        (1 - alpha) * rgb[fg] + alpha * color[fg], 0, 255
    ).astype(np.uint8)
    return Image.fromarray(blended)


def visible_mask(mask: Image.Image) -> Image.Image:
    arr = np.asarray(mask, dtype=np.uint8)
    return Image.fromarray(arr * 255, mode="L")


def panel(images: list[Image.Image], titles: list[str]) -> Image.Image:
    converted = [im.convert("RGB") for im in images]
    widths, heights = zip(*(im.size for im in converted))
    cell_w, cell_h = max(widths), max(heights)
    pad = 16
    title_h = 36
    canvas = Image.new("RGB", (cell_w * 2 + pad * 3, cell_h * 2 + pad * 3 + title_h * 2), (18, 18, 18))
    draw = ImageDraw.Draw(canvas)
    try:
        font = ImageFont.truetype("arial.ttf", 18)
    except OSError:
        font = ImageFont.load_default()
    positions = [
        (pad, pad + title_h),
        (pad * 2 + cell_w, pad + title_h),
        (pad, pad * 2 + title_h * 2 + cell_h),
        (pad * 2 + cell_w, pad * 2 + title_h * 2 + cell_h),
    ]
    title_xy = [
        (pad, pad),
        (pad * 2 + cell_w, pad),
        (pad, pad * 2 + title_h + cell_h),
        (pad * 2 + cell_w, pad * 2 + title_h + cell_h),
    ]
    for im, title, pos, tpos in zip(converted, titles, positions, title_xy):
        canvas.paste(im, pos)
        draw.text(tpos, title, fill=(255, 255, 255), font=font)
    return canvas


def choose_samples(records: list[dict], k: int = 9) -> list[dict]:
    ranked = sorted(records, key=lambda r: r["foreground_pct"])
    picks = [
        ranked[0],
        ranked[len(ranked) // 4],
        ranked[len(ranked) // 2],
        ranked[(3 * len(ranked)) // 4],
        ranked[-1],
        min(records, key=lambda r: r["width"] * r["height"]),
        max(records, key=lambda r: r["width"] * r["height"]),
    ]
    rng = random.Random(SPLIT_SEED)
    extras = rng.sample(records, k=min(k, len(records)))
    selected = []
    seen = set()
    for rec in picks + extras:
        if rec["stem"] not in seen:
            selected.append(rec)
            seen.add(rec["stem"])
        if len(selected) >= k:
            break
    return selected


def proposed_split(stems: list[str]) -> dict:
    shuffled = list(stems)
    rng = random.Random(SPLIT_SEED)
    rng.shuffle(shuffled)
    n = len(shuffled)
    n_train = int(round(n * TRAIN_RATIO))
    n_val = int(round(n * VAL_RATIO))
    n_test = n - n_train - n_val
    return {
        "seed": SPLIT_SEED,
        "method": "shuffle all stems with random.Random(seed), then sequential 70/15/15 cut",
        "ratios": {"train": TRAIN_RATIO, "val": VAL_RATIO, "test": 1.0 - TRAIN_RATIO - VAL_RATIO},
        "counts": {"train": n_train, "val": n_val, "test": n_test, "total": n},
        "status": "proposed_only_not_written",
    }


def main() -> int:
    if not IMAGES_DIR.is_dir() or not LABELS_DIR.is_dir():
        print("ERROR: TLPD images/ or labels/ directory is missing.")
        return 1

    image_by_stem, index_issues = index_images(IMAGES_DIR)
    json_paths = sorted(p for p in LABELS_DIR.iterdir() if p.is_file() and p.suffix.lower() == ".json")
    json_stems = [p.stem for p in json_paths]
    json_stem_counts = Counter(json_stems)
    pairing_issues = list(index_issues)
    pairing_issues.extend([f"duplicate JSON stem: {stem}" for stem, count in json_stem_counts.items() if count > 1])

    image_stems = set(image_by_stem)
    json_stem_set = set(json_stems)
    missing_images = sorted(json_stem_set - image_stems)
    missing_jsons = sorted(image_stems - json_stem_set)
    pairing_issues.extend([f"JSON without image: {stem}" for stem in missing_images])
    pairing_issues.extend([f"image without JSON: {stem}" for stem in missing_jsons])

    annotation_issues: list[str] = []
    valid_pairs: list[dict] = []
    for json_path in json_paths:
        stem = json_path.stem
        if stem not in image_by_stem:
            continue
        data = load_label(json_path)
        points, width, height, issues = validate_annotation(stem, data)
        annotation_issues.extend(issues)
        if issues:
            continue
        valid_pairs.append(
            {
                "stem": stem,
                "image_path": image_by_stem[stem],
                "json_path": json_path,
                "points": points,
                "json_width": width,
                "json_height": height,
            }
        )

    print(f"images={len(image_by_stem)}")
    print(f"json={len(json_paths)}")
    print(f"valid_pairs_before_masks={len(valid_pairs)}")
    print(f"missing_images={len(missing_images)}")
    print(f"missing_jsons={len(missing_jsons)}")
    print(f"pairing_issues={len(pairing_issues)}")
    print(f"annotation_issues={len(annotation_issues)}")

    significant = (
        len(image_by_stem) != EXPECTED_COUNT
        or len(json_paths) != EXPECTED_COUNT
        or missing_images
        or missing_jsons
        or annotation_issues
        or pairing_issues
    )
    if significant:
        print("ERROR: image/annotation mismatch or invalid polygons. Mask generation aborted.")
        for line in (pairing_issues + annotation_issues)[:40]:
            print("  ", line)
        return 1

    MASKS_DIR.mkdir(parents=True, exist_ok=True)
    existing_masks = list(MASKS_DIR.glob("*.png"))
    generated = 0
    skipped_existing = 0
    if existing_masks:
        print(f"Existing masks found ({len(existing_masks)}); not overwriting. Validating instead.")
    else:
        for pair in valid_pairs:
            mask = polygon_to_mask(pair["json_width"], pair["json_height"], pair["points"])
            save_mask(MASKS_DIR / f"{pair['stem']}.png", mask)
            generated += 1
        print(f"generated_masks={generated}")

    records = []
    empty_masks = []
    corrupted_masks = []
    dimension_mismatches = []
    unique_value_issues = []
    missing_mask_files = []
    unique_values_global: set[int] = set()
    image_sizes = Counter()
    mask_sizes = Counter()

    for pair in valid_pairs:
        mask_path = MASKS_DIR / f"{pair['stem']}.png"
        if not mask_path.is_file():
            missing_mask_files.append(pair["stem"])
            continue
        try:
            with Image.open(pair["image_path"]) as im:
                image = im.convert("RGB")
                image_size = image.size
            with Image.open(mask_path) as mk:
                mask = mk.copy()
                if mask.mode != "L":
                    mask = mask.convert("L")
        except Exception as exc:
            corrupted_masks.append(f"{pair['stem']}: {exc}")
            continue

        mask_arr = np.asarray(mask, dtype=np.uint8)
        unique = sorted(int(v) for v in np.unique(mask_arr))
        unique_values_global.update(unique)
        fg = int((mask_arr == 1).sum()) if 1 in unique else int((mask_arr > 0).sum())
        total = int(mask_arr.size)
        fg_pct = (100.0 * fg / total) if total else 0.0
        if fg == 0:
            empty_masks.append(pair["stem"])
        if unique not in ([0, 1], [0], [1]):
            unique_value_issues.append(f"{pair['stem']}: unique={unique}")
        json_size = (pair["json_width"], pair["json_height"])
        if image_size != json_size or mask.size != json_size or mask.size != image_size:
            dimension_mismatches.append(
                f"{pair['stem']}: image={image_size} json={json_size} mask={mask.size}"
            )
        image_sizes[f"{image_size[0]}x{image_size[1]}"] += 1
        mask_sizes[f"{mask.size[0]}x{mask.size[1]}"] += 1
        records.append(
            {
                "stem": pair["stem"],
                "image_path": str(pair["image_path"]),
                "json_path": str(pair["json_path"]),
                "mask_path": str(mask_path),
                "width": mask.size[0],
                "height": mask.size[1],
                "unique_values": unique,
                "foreground_pixels": fg,
                "foreground_pct": fg_pct,
                "points": pair["points"],
            }
        )

    fg_pcts = [r["foreground_pct"] for r in records]
    fg_stats = {
        "min": min(fg_pcts) if fg_pcts else None,
        "max": max(fg_pcts) if fg_pcts else None,
        "mean": float(statistics.fmean(fg_pcts)) if fg_pcts else None,
        "median": float(statistics.median(fg_pcts)) if fg_pcts else None,
    }

    INSPECTION_DIR.mkdir(parents=True, exist_ok=True)
    samples = choose_samples(records, k=9) if records else []
    sample_paths = []
    for idx, rec in enumerate(samples, start=1):
        image = Image.open(rec["image_path"]).convert("RGB")
        mask = Image.open(rec["mask_path"]).convert("L")
        polygon_im = draw_polygon_on_image(image, rec["points"])
        overlay = make_overlay(image, mask)
        combined = panel(
            [image, polygon_im, visible_mask(mask).convert("RGB"), overlay],
            ["1. Original", "2. Polygon", "3. Binary mask", "4. Overlay"],
        )
        stem_safe = rec["stem"]
        out = INSPECTION_DIR / f"mask_sample_{idx:02d}_{stem_safe}_panel.png"
        combined.save(out)
        sample_paths.append(str(out.relative_to(ROOT)))
        print(f"wrote {out.name}")

    split = proposed_split([r["stem"] for r in records])
    report = {
        "total_images": len(image_by_stem),
        "total_json": len(json_paths),
        "total_masks": len(list(MASKS_DIR.glob('*.png'))),
        "valid_image_mask_pairs": len(records),
        "invalid_or_missing_pairs": {
            "missing_images": missing_images,
            "missing_jsons": missing_jsons,
            "missing_masks": missing_mask_files,
            "empty_masks": empty_masks,
            "corrupted_masks": corrupted_masks,
            "dimension_mismatches": dimension_mismatches,
            "unique_value_issues": unique_value_issues,
            "annotation_issues": annotation_issues,
        },
        "mask_dimensions": {
            "unique_sizes": dict(mask_sizes.most_common()),
            "unique_size_count": len(mask_sizes),
            "min_width": min((r["width"] for r in records), default=None),
            "max_width": max((r["width"] for r in records), default=None),
            "min_height": min((r["height"] for r in records), default=None),
            "max_height": max((r["height"] for r in records), default=None),
        },
        "unique_mask_values": sorted(unique_values_global),
        "foreground_percentage": fg_stats,
        "example_visualization_paths": sample_paths,
        "masks_generated_this_run": generated,
        "masks_skipped_existing": skipped_existing,
        "proposed_split": split,
        "annotation_limitation": (
            "TLPD polygons may represent the license-plate character/text region "
            "rather than the complete physical outer boundary of the plate. "
            "Polygons were filled exactly as annotated; they were not enlarged."
        ),
    }
    REPORT_PATH.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({k: report[k] for k in [
        "total_images",
        "total_json",
        "total_masks",
        "valid_image_mask_pairs",
        "unique_mask_values",
        "foreground_percentage",
        "proposed_split",
    ]}, indent=2))
    print(f"empty_masks={len(empty_masks)}")
    print(f"corrupted_masks={len(corrupted_masks)}")
    print(f"dimension_mismatches={len(dimension_mismatches)}")
    print(f"unique_value_issues={len(unique_value_issues)}")
    print(f"missing_masks={len(missing_mask_files)}")
    print(f"report={REPORT_PATH}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
