"""Convert AGAR (one image + one json per sample) into COCO train/val/test files.

    python scripts/inspect_agar.py --src raw/agar          # look at one json first
    python scripts/agar_to_coco.py --src raw/agar --out data/agar

AGAR's paper describes each json as holding: background category, species list,
colony count (-1 = uncountable) and boxes (top-left x, y, width, height, class).
The exact key names below are a best guess - check them with inspect_agar.py and
change the KEYS dict if your copy differs. If your copy already has COCO files,
skip this script and just split them.

The split follows the paper: random, stratified by background category.
"""
import argparse
import glob
import json
import os
import random

# best-guess AGAR key names - edit to match your files
KEYS = {
    "labels": "labels",            # list of boxes
    "x": "x", "y": "y", "w": "width", "h": "height",
    "cls": "class",                # species / defect / contamination name
    "count": "colonies_number",    # -1 = uncountable
    "bg": "background",            # bright / dark / vague / lower-resolution
}
IMG_EXT = (".jpg", ".jpeg", ".png", ".tif", ".tiff")


def find_image(js_path):
    stem = os.path.splitext(js_path)[0]
    for e in IMG_EXT:
        for p in (stem + e, stem + e.upper()):
            if os.path.exists(p):
                return p
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default="raw/agar")
    ap.add_argument("--out", default="data/agar")
    ap.add_argument("--val", type=float, default=0.10)
    ap.add_argument("--test", type=float, default=0.25)
    ap.add_argument("--skip-bg", default="vague", help="comma list of backgrounds to drop ('' keeps all)")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    skip = {s for s in a.skip_bg.split(",") if s}
    recs, names = [], set()
    from PIL import Image
    for js in sorted(glob.glob(os.path.join(a.src, "**", "*.json"), recursive=True)):
        ip = find_image(js)
        if ip is None:
            continue
        with open(js) as f:
            d = json.load(f)
        bg = str(d.get(KEYS["bg"], "unknown"))
        if bg in skip:
            continue
        boxes = [(b[KEYS["x"]], b[KEYS["y"]], b[KEYS["w"]], b[KEYS["h"]], str(b[KEYS["cls"]]))
                 for b in d.get(KEYS["labels"], [])]
        names.update(c for *_, c in boxes)
        w, h = Image.open(ip).size
        recs.append({"path": os.path.relpath(ip, a.src), "w": w, "h": h, "bg": bg,
                     "count": d.get(KEYS["count"], len(boxes)), "boxes": boxes})
    if not recs:
        raise SystemExit("no json/image pairs found - check --src and the KEYS dict")

    cats = [{"id": i + 1, "name": n} for i, n in enumerate(sorted(names))]
    cid = {c["name"]: c["id"] for c in cats}
    rng = random.Random(a.seed)
    split = {"train": [], "val": [], "test": []}
    for bg in sorted({r["bg"] for r in recs}):  # stratify by background, like the paper
        g = [r for r in recs if r["bg"] == bg]
        rng.shuffle(g)
        nt, nv = round(len(g) * a.test), round(len(g) * a.val)
        split["test"] += g[:nt]
        split["val"] += g[nt:nt + nv]
        split["train"] += g[nt + nv:]

    os.makedirs(a.out, exist_ok=True)
    for s, rs in split.items():
        out, aid = {"images": [], "annotations": [], "categories": cats}, 1
        for i, r in enumerate(rs, 1):
            out["images"].append({"id": i, "file_name": r["path"], "width": r["w"], "height": r["h"],
                                  "background": r["bg"], "colonies_number": r["count"]})
            for x, y, w, h, c in r["boxes"]:
                out["annotations"].append({"id": aid, "image_id": i, "bbox": [x, y, w, h], "category_id": cid[c],
                                           "area": w * h, "iscrowd": 0})
                aid += 1
        with open(os.path.join(a.out, f"{s}.json"), "w") as f:
            json.dump(out, f)
        print(f"{s}: {len(rs)} plates, {aid - 1} boxes")
    print("classes:", [c["name"] for c in cats])


if __name__ == "__main__":
    main()
