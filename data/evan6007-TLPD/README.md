---
license: mit
pretty_name: Taiwan License Plate Dataset
annotations_creators:
- expert-generated
language:
- en
- zh
multilinguality: monolingual
size_categories:
- 1K<n<10K
source_datasets:
- original
task_categories:
- object-detection
tags:
- license-plate
- polygon
- labelme
---

# TLPD: Taiwan License Plate Dataset

TLPD is a dataset containing over 3,000 images of vehicles with annotated license plates. Each image is labeled using the [LabelMe](https://github.com/wkentaro/labelme) format, with polygon annotations describing the boundary of each license plate.

This dataset is designed for tasks such as license plate detection, polygon segmentation, and scene text detection.

## 📁 Dataset Structure

All image files are stored in the `images/` directory, and their corresponding polygon annotations are in the `labels/` directory:

```
images/
├── 0001.jpg
├── 0002.jpg
└── ...

labels/
├── 0001.json
├── 0002.json
└── ...
```

Each `.jpg` image is paired with a `.json` file of the same name containing the polygon annotation (in LabelMe format).

## 🏷️ Annotation Format (LabelMe)

Each `.json` file includes:

- `"imagePath"`: the name of the image
- `"shapes"[0]["label"]`: `"carplate"`
- `"shapes"[0]["points"]`: polygon points in the format `[[x1, y1], [x2, y2], ...]`
- `"imageHeight"`, `"imageWidth"`: image dimensions

### Example JSON snippet:

```json
{
  "imagePath": "0001.jpg",
  "shapes": [
    {
      "label": "carplate",
      "points": [
        [5.0, 8.0],
        [117.0, 12.0],
        [115.0, 52.0],
        [3.0, 48.0]
      ],
      "shape_type": "polygon"
    }
  ],
  "imageHeight": 60,
  "imageWidth": 121
}
```

## 💻 How to Use

To load this dataset using the Hugging Face 🤗 `datasets` library:

```python
from datasets import load_dataset

ds = load_dataset("evan6007/TLPD", data_dir=".", trust_remote_code=True)
sample = ds["train"][0]

image = sample["image"]  # PIL image
label = sample["label"]  # "carplate"
points = sample["points"]  # [[x1, y1], ..., [x4, y4]]
```

> Note: This dataset requires a custom loading script (`dataset.py`). Be sure to set `trust_remote_code=True`.

## 🧑‍🔬 Intended Use

- Object Detection (license plate)
- Polygon segmentation
- Scene text analysis
- Few-shot detection tasks

## 🪪 License

This dataset is licensed under the **MIT License**. You are free to use, share, and modify it for both academic and commercial purposes, with attribution.

## ✍️ Citation

```
@misc{TLPD2025,
  title={Taiwan License Plate Dataset},
  author={Hoi Lee ,Jui-Hung Weng, Chao-Hsiang Hsiao},
  year={2025},
  howpublished={\url{https://huggingface.co/datasets/evan6007/TLPD}}
}
```