"""Pick the score threshold that gives the best colony COUNTS on validation plates.

mAP uses every box at every confidence; a lab needs one number per plate, so the
counting threshold is a separate choice - tuned here on validation, never on test.

    python tune.py --ckpt runs/det/best.pt
"""
import argparse
import os

import numpy as np

from colonycount.evalplates import counts, run
from colonycount.model import load_detector, pick_device


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/agar")
    ap.add_argument("--img-root", default="raw/agar")
    ap.add_argument("--ckpt", default="runs/det/best.pt")
    ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()

    dev = pick_device()
    model, ck = load_detector(a.ckpt, dev)
    dets, gt = run(model, ck["classes"], os.path.join(a.data, "val.json"), a.img_root, dev, limit=a.limit or None)
    res = []
    for thr in np.arange(0.3, 0.95, 0.05):
        c = counts(dets, gt, thr)
        res.append((c["count_mae"], round(float(thr), 2)))
        print(f"thr={thr:.2f} val count MAE={c['count_mae']:.2f}  sMAPE={c['count_smape']:.1f}%")
    mae, thr = min(res)
    print(f"best: --count-thr {thr} (val MAE {mae:.2f})")


if __name__ == "__main__":
    main()
