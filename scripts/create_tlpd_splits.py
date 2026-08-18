"""Check TLPD leakage, then write a seed-42 train/val/test split."""

from __future__ import annotations

import hashlib
import json
import random
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
DATASET_DIR = ROOT / "data" / "evan6007-TLPD"
IMAGES_DIR = DATASET_DIR / "images"
LABELS_DIR = DATASET_DIR / "labels"
MASKS_DIR = DATASET_DIR / "masks"
SPLITS_DIR = DATASET_DIR / "splits"
REPORT_PATH = ROOT / "results" / "tlpd_inspection" / "split_leakage_report.json"
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
EXPECTED = 3032
N_TRAIN = 2122
N_VAL = 455
N_TEST = 455
SEED = 42
DHASH_SIZE = 8
NEAR_DUP_HAMMING = 5
IDENTICAL_HAMMING = 0
PAREN_SUFFIX = re.compile(r"(?:\([^)]*\))+$")


class UnionFind:
    def __init__(self, items: list[str]) -> None:
        self.parent = {item: item for item in items}

    def find(self, item: str) -> str:
        while self.parent[item] != item:
            self.parent[item] = self.parent[self.parent[item]]
            item = self.parent[item]
        return item

    def union(self, a: str, b: str) -> None:
        pa, pb = self.find(a), self.find(b)
        if pa != pb:
            self.parent[pb] = pa

    def groups(self) -> list[list[str]]:
        buckets: dict[str, list[str]] = defaultdict(list)
        for item in self.parent:
            buckets[self.find(item)].append(item)
        return [sorted(v) for v in buckets.values()]


def index_images() -> dict[str, Path]:
    by_stem: dict[str, Path] = {}
    for path in IMAGES_DIR.iterdir():
        if path.is_file() and path.suffix.lower() in IMAGE_EXTS:
            if path.stem in by_stem:
                raise RuntimeError(f"duplicate image stem: {path.stem}")
            by_stem[path.stem] = path
    return by_stem


def plate_id(stem: str) -> str:
    return PAREN_SUFFIX.sub("", stem)


def dhash64(image: Image.Image, hash_size: int = DHASH_SIZE) -> int:
    gray = image.convert("L").resize((hash_size + 1, hash_size), Image.Resampling.BILINEAR)
    pixels = np.asarray(gray, dtype=np.int16)
    bits = (pixels[:, 1:] > pixels[:, :-1]).flatten()
    value = 0
    for bit in bits:
        value = (value << 1) | int(bit)
    return value


def hamming(a: int, b: int) -> int:
    return (a ^ b).bit_count()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_pixels(image: Image.Image) -> str:
    arr = np.asarray(image.convert("RGB"), dtype=np.uint8)
    return hashlib.sha256(arr.tobytes()).hexdigest()


def groups_from_map(mapping: dict[str, list[str]]) -> list[list[str]]:
    return [sorted(v) for v in mapping.values() if len(v) > 1]


def assign_split(groups: list[list[str]]) -> dict[str, list[str]]:
    """Shuffle groups with seed 42, flatten, then repair so no group crosses splits."""
    groups = [sorted(g) for g in groups]
    groups.sort(key=lambda g: g[0])
    rng = random.Random(SEED)
    rng.shuffle(groups)

    order = [stem for group in groups for stem in group]
    if len(order) != EXPECTED:
        raise RuntimeError(f"expected {EXPECTED} stems, got {len(order)}")

    train = set(order[:N_TRAIN])
    val = set(order[N_TRAIN:N_TRAIN + N_VAL])
    test = set(order[N_TRAIN + N_VAL:])
    membership = {stem: "train" for stem in train}
    membership.update({stem: "val" for stem in val})
    membership.update({stem: "test" for stem in test})

    def counts() -> dict[str, int]:
        return {
            "train": sum(1 for s in membership.values() if s == "train"),
            "val": sum(1 for s in membership.values() if s == "val"),
            "test": sum(1 for s in membership.values() if s == "test"),
        }

    # Keep each duplicate group in one split (majority, then earliest in shuffle order).
    for group in groups:
        if len(group) == 1:
            continue
        votes = Counter(membership[stem] for stem in group)
        target = votes.most_common(1)[0][0]
        if len(votes) == 1:
            continue
        for stem in group:
            membership[stem] = target

    target_counts = {"train": N_TRAIN, "val": N_VAL, "test": N_TEST}
    singleton_of = {g[0]: g[0] for g in groups if len(g) == 1}
    movable = [stem for stem in order if stem in singleton_of]

    def move_one(src: str, dst: str) -> bool:
        for stem in movable:
            if membership[stem] == src:
                membership[stem] = dst
                return True
        return False

    current = counts()
    for split_name, needed in target_counts.items():
        while current[split_name] < needed:
            donor = max(
                (name for name in target_counts if current[name] > target_counts[name]),
                key=lambda name: current[name] - target_counts[name],
                default=None,
            )
            if donor is None or not move_one(donor, split_name):
                raise RuntimeError(f"could not rebalance into {split_name}: {current}")
            current = counts()
        while current[split_name] > needed:
            receiver = min(
                (name for name in target_counts if current[name] < target_counts[name]),
                key=lambda name: current[name] - target_counts[name],
                default=None,
            )
            if receiver is None or not move_one(split_name, receiver):
                raise RuntimeError(f"could not rebalance out of {split_name}: {current}")
            current = counts()

    if counts() != target_counts:
        raise RuntimeError(f"final counts {counts()} != {target_counts}")

    result = {"train": [], "val": [], "test": []}
    for stem, split_name in membership.items():
        result[split_name].append(stem)
    for split_name in result:
        result[split_name].sort()
    return result


def write_split_files(splits: dict[str, list[str]]) -> None:
    SPLITS_DIR.mkdir(parents=True, exist_ok=True)
    for name, stems in splits.items():
        path = SPLITS_DIR / f"{name}.txt"
        path.write_text("\n".join(stems) + "\n", encoding="utf-8")


def main() -> int:
    image_by_stem = index_images()
    json_stems = sorted(p.stem for p in LABELS_DIR.glob("*.json"))
    mask_stems = sorted(p.stem for p in MASKS_DIR.glob("*.png"))
    image_stems = sorted(image_by_stem)

    stem_counts = Counter(image_stems)
    duplicate_stems = sorted(stem for stem, n in stem_counts.items() if n > 1)

    missing_images = sorted(set(json_stems) - set(image_stems))
    missing_jsons = sorted(set(image_stems) - set(json_stems))
    missing_masks = sorted(set(image_stems) - set(mask_stems))
    extra_masks = sorted(set(mask_stems) - set(image_stems))

    file_hash_map: dict[str, list[str]] = defaultdict(list)
    pixel_hash_map: dict[str, list[str]] = defaultdict(list)
    plate_map: dict[str, list[str]] = defaultdict(list)
    dhash_of: dict[str, int] = {}

    for stem, path in sorted(image_by_stem.items()):
        file_hash_map[sha256_file(path)].append(stem)
        with Image.open(path) as im:
            pixel_hash_map[sha256_pixels(im)].append(stem)
            dhash_of[stem] = dhash64(im)
        plate_map[plate_id(stem)].append(stem)

    exact_file_groups = groups_from_map(file_hash_map)
    identical_pixel_groups = groups_from_map(pixel_hash_map)
    plate_groups = groups_from_map(plate_map)

    near_pairs = []
    identical_appearance_pairs = []
    perceptual_false_positives = []
    stems = image_stems
    hashes = [dhash_of[s] for s in stems]
    for i, stem_a in enumerate(stems):
        ha = hashes[i]
        for j in range(i + 1, len(stems)):
            dist = hamming(ha, hashes[j])
            if dist > NEAR_DUP_HAMMING:
                continue
            stem_b = stems[j]
            pair = (stem_a, stem_b, dist)
            near_pairs.append(pair)
            same_plate = plate_id(stem_a) == plate_id(stem_b)
            if dist == IDENTICAL_HAMMING:
                identical_appearance_pairs.append(pair)
            if not same_plate:
                perceptual_false_positives.append(pair)

    # Split constraint: keep exact duplicates, visually identical images,
    # and same-plate filename variants in one split. Do not merge different
    # plates that only have a close perceptual hash (false positives).
    uf = UnionFind(image_stems)
    for group in exact_file_groups + identical_pixel_groups + plate_groups:
        for stem in group[1:]:
            uf.union(group[0], stem)
    for stem_a, stem_b, dist in identical_appearance_pairs:
        uf.union(stem_a, stem_b)

    leak_groups = [g for g in uf.groups() if len(g) > 1]
    split_groups = uf.groups()
    group_sizes = sorted((len(g) for g in leak_groups), reverse=True)

    splits = assign_split(split_groups)

    # Validate pairing and overlap.
    train, val, test = splits["train"], splits["val"], splits["test"]
    overlap = {
        "train_val": sorted(set(train) & set(val)),
        "train_test": sorted(set(train) & set(test)),
        "val_test": sorted(set(val) & set(test)),
    }
    split_missing_images = sorted(s for s in train + val + test if s not in image_by_stem)
    split_missing_jsons = sorted(s for s in train + val + test if not (LABELS_DIR / f"{s}.json").is_file())
    split_missing_masks = sorted(s for s in train + val + test if not (MASKS_DIR / f"{s}.png").is_file())

    crossed_leak_groups = []
    membership = {s: "train" for s in train}
    membership.update({s: "val" for s in val})
    membership.update({s: "test" for s in test})
    for group in leak_groups:
        splits_hit = sorted({membership[s] for s in group})
        if len(splits_hit) > 1:
            crossed_leak_groups.append({"stems": group, "splits": splits_hit})

    crossed_plate_groups = []
    for group in plate_groups:
        splits_hit = sorted({membership[s] for s in group})
        if len(splits_hit) > 1:
            crossed_plate_groups.append(
                {"plate_id": plate_id(group[0]), "stems": group, "splits": splits_hit, "count": len(group)}
            )

    if overlap["train_val"] or overlap["train_test"] or overlap["val_test"]:
        print("ERROR: overlap between splits.")
        return 1
    if crossed_leak_groups or crossed_plate_groups:
        print("ERROR: duplicate/near-duplicate or same-plate group crossed splits.")
        return 1
    if (
        len(train) != N_TRAIN
        or len(val) != N_VAL
        or len(test) != N_TEST
        or len(train) + len(val) + len(test) != EXPECTED
    ):
        print("ERROR: split counts are wrong.")
        return 1
    if split_missing_images or split_missing_jsons or split_missing_masks or extra_masks:
        print("ERROR: missing paired files in the split.")
        return 1

    write_split_files(splits)
    manifest = {
        "seed": SEED,
        "counts": {"train": len(train), "val": len(val), "test": len(test), "total": EXPECTED},
        "method": (
            "Group stems by exact file hash, identical pixels, visually identical dhash "
            "(hamming 0), and same plate-id prefix after stripping trailing parenthetical "
            "suffixes. Sort groups by first stem, shuffle groups with random.Random(42), "
            "flatten, cut 2122/455/455, then keep each group in one split and rebalance "
            "using singleton stems only. Different plates with close dhash are not merged."
        ),
        "identifier": "filename stem; pair with images/<stem>.jpg, labels/<stem>.json, masks/<stem>.png",
        "files": {
            "train": "data/evan6007-TLPD/splits/train.txt",
            "val": "data/evan6007-TLPD/splits/val.txt",
            "test": "data/evan6007-TLPD/splits/test.txt",
        },
    }
    (SPLITS_DIR / "split_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    report = {
        "duplicate_filename_stems": duplicate_stems,
        "exact_file_duplicate_groups": exact_file_groups,
        "identical_pixel_groups": identical_pixel_groups,
        "near_duplicate_pair_count": len(near_pairs),
        "near_duplicate_pairs_hamming_le_5": [
            {"a": a, "b": b, "hamming": d} for a, b, d in near_pairs[:200]
        ],
        "near_duplicate_groups_perceptual_only": [
            sorted({a, b}) for a, b, _d in near_pairs if plate_id(a) == plate_id(b)
        ],
        "perceptual_false_positives_different_plates": [
            {"a": a, "b": b, "hamming": d} for a, b, d in perceptual_false_positives
        ],
        "identical_appearance_pairs": [
            {"a": a, "b": b, "hamming": d} for a, b, d in identical_appearance_pairs
        ],
        "split_constraint_groups": leak_groups,
        "split_constraint_group_count": len(leak_groups),
        "split_constraint_group_size_max": max((len(g) for g in leak_groups), default=1),
        "same_plate_id_groups": [
            {"plate_id": plate_id(g[0]), "count": len(g), "stems": g} for g in plate_groups
        ],
        "same_plate_id_group_count": len(plate_groups),
        "images_in_multi_capture_plate_ids": sum(len(g) for g in plate_groups),
        "split": manifest,
        "overlap": overlap,
        "missing_images": split_missing_images,
        "missing_jsons": split_missing_jsons,
        "missing_masks": split_missing_masks,
        "crossed_near_duplicate_groups": crossed_leak_groups,
        "crossed_same_plate_id_groups": crossed_plate_groups,
    }
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print(f"images={len(image_stems)}")
    print(f"json={len(json_stems)}")
    print(f"masks={len(mask_stems)}")
    print(f"duplicate_stems={len(duplicate_stems)}")
    print(f"exact_file_groups={len(exact_file_groups)} sizes={[len(g) for g in exact_file_groups]}")
    print(f"identical_pixel_groups={len(identical_pixel_groups)} sizes={[len(g) for g in identical_pixel_groups]}")
    print(f"near_duplicate_pairs_hamming_le_5={len(near_pairs)}")
    print(f"identical_appearance_pairs={len(identical_appearance_pairs)}")
    print(f"perceptual_false_positives={len(perceptual_false_positives)}")
    print(f"split_constraint_groups={len(leak_groups)} max_size={max((len(g) for g in leak_groups), default=1)}")
    print(f"same_plate_id_groups={len(plate_groups)}")
    print(f"images_in_multi_capture_plate_ids={sum(len(g) for g in plate_groups)}")
    print(f"train={len(train)} val={len(val)} test={len(test)} total={len(train)+len(val)+len(test)}")
    print(f"overlap={ {k: len(v) for k, v in overlap.items()} }")
    print(f"missing_images={len(split_missing_images)} missing_jsons={len(split_missing_jsons)} missing_masks={len(split_missing_masks)}")
    print(f"crossed_constraint_groups={len(crossed_leak_groups)}")
    print(f"crossed_same_plate_id_groups={len(crossed_plate_groups)}")
    print(f"report={REPORT_PATH}")
    print(f"splits={SPLITS_DIR}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
