# TLPD local dataset notes

Primary dataset for this license-plate **segmentation** project: [evan6007/TLPD](https://huggingface.co/datasets/evan6007/TLPD).

Do not use the Simon Graves license-plate dataset for training.
Do not use Kvasir-SEG.
Do not combine datasets at this stage.

## Layout

```
data/evan6007-TLPD/
├── images/    # original JPEG images
├── labels/    # original LabelMe JSON annotations (kept unmodified)
├── masks/     # generated binary PNG masks
└── splits/    # train.txt, val.txt, test.txt (filename stems; seed 42)
```

## Pairing rule

Image and JSON files are matched by **JSON filename stem**, not by the JSON `imagePath` field. That field can disagree with the actual image filename.

## Mask generation

For each valid pair:

1. Read `imageWidth` and `imageHeight` from the JSON.
2. Create an `H × W` background mask.
3. Fill the annotated 4-point polygon exactly (`PIL.ImageDraw.polygon`).
4. Save a PNG with values `0` (background) and `1` (annotated region).

Polygons are not converted from bounding boxes.
Coordinates are not modified or expanded.

## Annotation limitation

The TLPD polygon may represent the **license-plate character/text region** rather than the complete physical outer boundary of the license plate.

Masks preserve the original annotation exactly. They are **not** automatically enlarged to cover the full physical plate. A model trained on these masks will therefore target the annotated text/plate region, which can be smaller than the full plate.

## Splits

`splits/train.txt`, `splits/val.txt`, and `splits/test.txt` list filename stems (2,122 / 455 / 455, seed 42). Each stem pairs with `images/`, `labels/`, and `masks/` of the same name. Image and mask files are not copied.

Same-plate filename variants (trailing parenthetical suffixes such as `(0)` or `(1)(0)(1)`) are kept in the same split so those captures do not leak across train/validation/test.
