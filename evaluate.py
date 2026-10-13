"""Final test-set evaluation: detection mAP, counting error, plate classification.

    python evaluate.py --ckpt runs/det/best.pt --plate runs/plate/best.pt --count-thr 0.6
"""
import argparse
import json
import os

import numpy as np
import pandas as pd
import torch

from colonycount.evalplates import run, score
from colonycount.model import load_detector, load_plate, pick_device


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/agar")
    ap.add_argument("--img-root", default="raw/agar")
    ap.add_argument("--ckpt", default="runs/det/best.pt")
    ap.add_argument("--plate", default="", help="plate classifier checkpoint (optional)")
    ap.add_argument("--count-thr", type=float, default=0.5)
    ap.add_argument("--split", default="test")
    ap.add_argument("--out", default="results")
    a = ap.parse_args()

    os.makedirs(a.out, exist_ok=True)
    dev = pick_device()
    model, ck = load_detector(a.ckpt, dev)
    dets, gt = run(model, ck["classes"], os.path.join(a.data, f"{a.split}.json"), a.img_root, dev)
    m = score(dets, gt, a.count_thr)

    pd.DataFrame({"image": [d["image"]["file_name"] for d in dets],
                  "background": [d["image"].get("background") for d in dets],
                  "true": m["true_counts"], "pred": m["pred_counts"]}).to_csv(
        os.path.join(a.out, "counts.csv"), index=False)
    summ = {k: (round(v, 4) if isinstance(v, float) else v) for k, v in m.items()
            if k not in ("pred_counts", "true_counts")}
    summ["per_class"] = {k: round(v, 4) for k, v in m["per_class"].items()}
    summ["plates"] = len(dets)
    summ["count_thr"] = a.count_thr

    if a.plate:
        from train_plate import Plates, evaluate as eval_plate, thumbs
        pm, _ = load_plate(a.plate, dev)
        items = thumbs(os.path.join(a.data, f"{a.split}.json"), a.img_root, os.path.join(a.data, "thumbs"))
        summ["plate_classifier"] = eval_plate(pm, torch.utils.data.DataLoader(Plates(items, False), batch_size=16), dev)

    with open(os.path.join(a.out, "summary.json"), "w") as f:
        json.dump(summ, f, indent=2)
    np.save(os.path.join(a.out, "dets.npy"), np.array(dets, dtype=object), allow_pickle=True)
    print(json.dumps(summ, indent=2))


if __name__ == "__main__":
    main()
