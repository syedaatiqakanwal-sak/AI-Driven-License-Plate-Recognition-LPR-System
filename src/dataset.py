"""TLPD image/mask dataset with letterbox preprocessing."""

from __future__ import annotations

from pathlib import Path

import albumentations as A
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
IMAGE_PAD_RGB = tuple(int(round(v * 255.0)) for v in IMAGENET_MEAN)
IMAGE_EXTS = (".jpg", ".jpeg", ".png", ".bmp")


def read_stems(split_path: Path) -> list[str]:
    lines = split_path.read_text(encoding="utf-8").splitlines()
    return [line.strip() for line in lines if line.strip()]


def find_image(images_dir: Path, stem: str) -> Path:
    for ext in IMAGE_EXTS:
        path = images_dir / f"{stem}{ext}"
        if path.is_file():
            return path
        path = images_dir / f"{stem}{ext.upper()}"
        if path.is_file():
            return path
    raise FileNotFoundError(f"No image for stem {stem!r} in {images_dir}")


def letterbox(image: np.ndarray, mask: np.ndarray, size: int) -> tuple[np.ndarray, np.ndarray]:
    height, width = image.shape[:2]
    scale = size / max(height, width)
    new_w = max(1, int(round(width * scale)))
    new_h = max(1, int(round(height * scale)))
    if (new_w, new_h) != (width, height):
        image = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
        mask = cv2.resize(mask, (new_w, new_h), interpolation=cv2.INTER_NEAREST)

    pad_w = size - new_w
    pad_h = size - new_h
    left = pad_w // 2
    right = pad_w - left
    top = pad_h // 2
    bottom = pad_h - top
    image = cv2.copyMakeBorder(
        image, top, bottom, left, right, cv2.BORDER_CONSTANT, value=IMAGE_PAD_RGB
    )
    mask = cv2.copyMakeBorder(mask, top, bottom, left, right, cv2.BORDER_CONSTANT, value=0)
    mask = (mask > 0).astype(np.uint8)
    return image, mask


def _gauss_noise(p: float):
    try:
        return A.GaussNoise(std_range=(0.02, 0.08), p=p)
    except TypeError:
        return A.GaussNoise(var_limit=(8.0, 25.0), p=p)


def _rotate():
    kwargs = {
        "limit": 10,
        "interpolation": cv2.INTER_LINEAR,
        "border_mode": cv2.BORDER_CONSTANT,
        "p": 1.0,
    }
    try:
        return A.Rotate(**kwargs, fill=0, fill_mask=0)
    except TypeError:
        return A.Rotate(**kwargs, value=0, mask_value=0)


def train_augmentations() -> A.Compose:
    return A.Compose(
        [
            A.HorizontalFlip(p=0.5),
            _rotate(),
            A.RandomScale(scale_limit=0.10, interpolation=cv2.INTER_LINEAR, p=1.0),
            A.RandomBrightnessContrast(brightness_limit=0.15, contrast_limit=0.15, p=1.0),
            A.GaussianBlur(blur_limit=(3, 3), p=0.15),
            _gauss_noise(0.10),
        ]
    )


class TLPDSegDataset(Dataset):
    def __init__(
        self,
        root: str | Path,
        split: str,
        image_size: int = 384,
        augment: bool = False,
    ) -> None:
        self.root = Path(root)
        self.images_dir = self.root / "images"
        self.masks_dir = self.root / "masks"
        self.image_size = image_size
        self.augment = augment
        self.stems = read_stems(self.root / "splits" / f"{split}.txt")
        self.aug = train_augmentations() if augment else None
        self.normalize = A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)

    def __len__(self) -> int:
        return len(self.stems)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor | str]:
        stem = self.stems[index]
        image_path = find_image(self.images_dir, stem)
        mask_path = self.masks_dir / f"{stem}.png"
        if not mask_path.is_file():
            raise FileNotFoundError(f"Missing mask for stem {stem!r}: {mask_path}")

        image_bgr = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if image_bgr is None:
            raise RuntimeError(f"Failed to read image: {image_path}")
        image = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        if mask is None:
            raise RuntimeError(f"Failed to read mask: {mask_path}")
        mask = (mask > 0).astype(np.uint8)

        if self.aug is not None:
            transformed = self.aug(image=image, mask=mask)
            image, mask = transformed["image"], transformed["mask"]

        image, mask = letterbox(image, mask, self.image_size)
        unique = np.unique(mask)
        if not set(unique.tolist()).issubset({0, 1}):
            raise RuntimeError(f"Mask for {stem} is not binary after letterbox: {unique}")

        image = self.normalize(image=image)["image"]
        image_t = torch.from_numpy(np.transpose(image, (2, 0, 1))).contiguous().float()
        mask_t = torch.from_numpy(mask).unsqueeze(0).contiguous().float()
        return {"image": image_t, "mask": mask_t, "stem": stem}
