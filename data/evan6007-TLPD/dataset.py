from datasets import (
    GeneratorBasedBuilder, DatasetInfo, SplitGenerator, Split,
    Features, Value, Sequence, Image, Version             # ← 多了 Version
)
import os, json

class TLPD(GeneratorBasedBuilder):
    VERSION = Version("0.0.3")   # ✔ 正確型別

    def _info(self):
        return DatasetInfo(
            description="Taiwan License Plate Dataset with LabelMe polygon annotations.",
            features=Features({
                "image":   Image(),                     # 顯示在 Viewer
                "label":   Value("string"),             # 顯示在 Viewer
                # ↓➜ 轉成字串，Viewer 就能看；模型端再 json.loads() 還原即可
                "points":  Value("string"),             
            }),
        )

    def _split_generators(self, dl_manager):
        root = dl_manager.manual_dir or dl_manager.download_and_extract({})
        return [SplitGenerator(
            name=Split.TRAIN,
            gen_kwargs={
                "images_dir": os.path.join(root, "images"),
                "labels_dir": os.path.join(root, "labels"),
            },
        )]

    def _generate_examples(self, images_dir, labels_dir):
        for fname in sorted(os.listdir(labels_dir)):
            if not fname.endswith(".json"):
                continue
            with open(os.path.join(labels_dir, fname), "r") as f:
                meta = json.load(f)

            shape  = meta["shapes"][0]      # 只有一個 polygon
            yield fname[:-5], {
                "image":  os.path.join(images_dir, meta["imagePath"]),
                "label":  shape["label"],                       # "carplate"
                "points": json.dumps(shape["points"]),          # <- 字串
            }
