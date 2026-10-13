"""Data: COCO-format plates, training patches, and the patch dataset.

Everything inside the project uses plain COCO json:
  images:      {id, file_name, width, height, background, colonies_number}
  annotations: {id, image_id, bbox [x, y, w, h], category_id}
  categories:  {id, name}
AGAR's per-image json files are converted to this by scripts/agar_to_coco.py.
"""
import json
import os
import random
from collections import defaultdict

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

# defects and contamination are annotated in AGAR but are not counted as colonies
NOT_COUNTED = ("defect", "contamination")


def load_coco(path):
    with open(path) as f:
        c = json.load(f)
    anns = defaultdict(list)
    for a in c["annotations"]:
        anns[a["image_id"]].append(a)
    return c, anns


def counted_ids(cats):
    return {c["id"] for c in cats if not any(k in c["name"].lower() for k in NOT_COUNTED)}


def plate_status(img, n_boxes):
    """0 = empty, 1 = countable, 2 = uncountable (AGAR marks these colonies_number = -1)."""
    n = img.get("colonies_number")
    if n is None:
        n = n_boxes
    if n < 0:
        return 2
    return 0 if n == 0 else 1


def read_rgb(path):
    return np.array(Image.open(path).convert("RGB"))


def make_patches(coco_path, img_dir, out_dir, size=512, per_img=12, empty_frac=0.05, seed=0, min_vis=0.5):
    """Cut training patches that each contain at least one colony, plus a few empty ones.

    This mirrors the AGAR paper's training setup: full-resolution plates are far too
    big to feed a detector, and random crops would mostly be empty agar.
    """
    rng = random.Random(seed)
    c, anns = load_coco(coco_path)
    os.makedirs(os.path.join(out_dir, "img"), exist_ok=True)
    out = {"images": [], "annotations": [], "categories": c["categories"]}
    aid = 1
    for im in c["images"]:
        boxes = anns[im["id"]]
        if plate_status(im, len(boxes)) != 1:
            continue  # empty / uncountable plates train the plate classifier, not the detector
        arr = read_rgb(os.path.join(img_dir, im["file_name"]))
        h, w = arr.shape[:2]
        # a few percent empty patches teach the detector what plain agar looks like
        n_empty = 1 if rng.random() < per_img * empty_frac else 0
        for k in range(per_img + n_empty):
            if k < per_img:  # centre the window so a random colony is inside it
                b = rng.choice(boxes)["bbox"]
                cx, cy = b[0] + b[2] / 2, b[1] + b[3] / 2
                x0 = int(np.clip(cx - rng.uniform(0.1, 0.9) * size, 0, max(w - size, 0)))
                y0 = int(np.clip(cy - rng.uniform(0.1, 0.9) * size, 0, max(h - size, 0)))
            else:
                x0, y0 = rng.randint(0, max(w - size, 0)), rng.randint(0, max(h - size, 0))
            crop = arr[y0:y0 + size, x0:x0 + size]
            keep = []
            for a in boxes:
                x, y, bw, bh = a["bbox"]
                ix0, iy0 = max(x, x0), max(y, y0)
                ix1, iy1 = min(x + bw, x0 + size), min(y + bh, y0 + size)
                if ix1 <= ix0 or iy1 <= iy0:
                    continue
                if (ix1 - ix0) * (iy1 - iy0) < min_vis * bw * bh:
                    continue  # mostly cut off: drop rather than teach half-colonies
                keep.append((ix0 - x0, iy0 - y0, ix1 - ix0, iy1 - iy0, a["category_id"]))
            if k < per_img and not keep:
                continue
            pid = len(out["images"]) + 1
            fn = f"{im['id']}_{k}.jpg"
            Image.fromarray(crop).save(os.path.join(out_dir, "img", fn), quality=95)
            out["images"].append({"id": pid, "file_name": fn, "width": crop.shape[1], "height": crop.shape[0],
                                  "source": im["id"], "background": im.get("background")})
            for x, y, bw, bh, cid in keep:
                out["annotations"].append({"id": aid, "image_id": pid, "bbox": [x, y, bw, bh],
                                           "category_id": cid, "area": bw * bh, "iscrowd": 0})
                aid += 1
    with open(os.path.join(out_dir, "patches.json"), "w") as f:
        json.dump(out, f)
    return len(out["images"]), len(out["annotations"])


def to_tensor(img):
    return torch.from_numpy(np.ascontiguousarray(img)).permute(2, 0, 1).float() / 255


class PatchSet(Dataset):
    """Patches + boxes for torchvision detection models (labels are 1-based)."""

    def __init__(self, coco_path, img_dir, aug=True):
        c, self.anns = load_coco(coco_path)
        self.imgs, self.dir, self.aug = c["images"], img_dir, aug
        self.cat2lab = {cat["id"]: i + 1 for i, cat in enumerate(c["categories"])}

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, i):
        im = self.imgs[i]
        img = read_rgb(os.path.join(self.dir, im["file_name"]))
        b = np.array([a["bbox"] for a in self.anns[im["id"]]], np.float32).reshape(-1, 4)
        b[:, 2:] += b[:, :2]  # xywh -> xyxy
        lab = np.array([self.cat2lab[a["category_id"]] for a in self.anns[im["id"]]], np.int64)
        if self.aug:
            img, b = augment(img, b)
        t = {"boxes": torch.from_numpy(b), "labels": torch.from_numpy(lab)}
        return to_tensor(img), t


def augment(img, b, rng=np.random):
    h, w = img.shape[:2]
    if rng.rand() < 0.5:
        img, b = img[:, ::-1], np.stack([w - b[:, 2], b[:, 1], w - b[:, 0], b[:, 3]], 1)
    if rng.rand() < 0.5:
        img, b = img[::-1], np.stack([b[:, 0], h - b[:, 3], b[:, 2], h - b[:, 1]], 1)
    if rng.rand() < 0.5:  # 90 degree rotation (plates have no "up")
        img = np.rot90(img)
        b = np.stack([b[:, 1], w - b[:, 2], b[:, 3], w - b[:, 0]], 1)
    # lighting varies a lot between AGAR subsets (bright / dark / vague)
    img = np.clip(img.astype(np.float32) * rng.uniform(0.8, 1.2) + rng.uniform(-20, 20), 0, 255)
    return img.astype(np.uint8), b.astype(np.float32).reshape(-1, 4)


def collate(batch):
    return tuple(zip(*batch))
