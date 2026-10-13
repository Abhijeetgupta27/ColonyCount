"""Print one AGAR annotation file so you can confirm the key names before converting.

    python scripts/inspect_agar.py --src raw/agar
"""
import argparse
import glob
import json
import os

ap = argparse.ArgumentParser()
ap.add_argument("--src", default="raw/agar")
ap.add_argument("--n", type=int, default=1)
a = ap.parse_args()

files = sorted(glob.glob(os.path.join(a.src, "**", "*.json"), recursive=True))
print(f"{len(files)} json files under {a.src}")
for f in files[:a.n]:
    with open(f) as fh:
        d = json.load(fh)
    print("\n==", f)
    if isinstance(d, dict) and "images" in d and "annotations" in d:
        print("looks like COCO format already:", {k: len(v) for k, v in d.items() if isinstance(v, list)})
        print("first image:", d["images"][0])
        print("first annotation:", d["annotations"][0] if d["annotations"] else None)
        print("categories:", d.get("categories"))
    else:
        s = json.dumps(d, indent=2)
        print(s[:1500] + ("\n..." if len(s) > 1500 else ""))
