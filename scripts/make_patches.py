"""Cut 512-px training patches from the full-resolution training plates.

    python scripts/make_patches.py --data data/agar --img-root raw/agar
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from colonycount.data import make_patches  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--data", default="data/agar")
ap.add_argument("--img-root", default="raw/agar")
ap.add_argument("--size", type=int, default=512)
ap.add_argument("--per-img", type=int, default=12)
a = ap.parse_args()

n, b = make_patches(os.path.join(a.data, "train.json"), a.img_root, os.path.join(a.data, "patches", "train"),
                    size=a.size, per_img=a.per_img)
print(f"train patches: {n} ({b} boxes)")
