"""Count colonies on one plate photo.

    python count.py plate.jpg --ckpt runs/det/best.pt --plate runs/plate/best.pt --count-thr 0.6

Writes <out>/<name>_counts.json and <name>_annotated.jpg.
"""
import argparse
import json
import os
import time

import cv2
import torch

from colonycount.data import read_rgb
from colonycount.infer import count, detect, plate_input
from colonycount.model import load_detector, load_plate, pick_device

COLORS = [(42, 120, 214), (235, 104, 52), (27, 175, 122), (237, 161, 0), (232, 123, 164), (98, 80, 214), (227, 73, 72)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("path")
    ap.add_argument("--ckpt", default="runs/det/best.pt")
    ap.add_argument("--plate", default="")
    ap.add_argument("--count-thr", type=float, default=0.5)
    ap.add_argument("--out", default="out")
    a = ap.parse_args()

    os.makedirs(a.out, exist_ok=True)
    dev = pick_device()
    img = read_rgb(a.path)
    name = os.path.splitext(os.path.basename(a.path))[0]
    res = {"image": a.path}
    t0 = time.time()
    if a.plate:
        pm, pk = load_plate(a.plate, dev)
        with torch.no_grad():
            pr = torch.softmax(pm(plate_input(img)[None].to(dev)), 1)[0].cpu()
        res["plate"] = {n: round(float(p), 3) for n, p in zip(pk["names"], pr)}
        status = pk["names"][int(pr.argmax())]
        res["status"] = status
        if status != "countable":  # don't invent a number for empty / confluent plates
            res["counts"] = {"total": 0} if status == "empty" else None
            print(json.dumps(res, indent=2))
            with open(os.path.join(a.out, f"{name}_counts.json"), "w") as f:
                json.dump(res, f, indent=2)
            return
    model, ck = load_detector(a.ckpt, dev)
    b, s, lab = detect(model, img, device=dev)
    res["counts"] = count(lab, s, ck["classes"], a.count_thr)
    res["seconds"] = round(time.time() - t0, 2)
    ann = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    for (x0, y0, x1, y1), sc, l in zip(b.tolist(), s.tolist(), lab.tolist()):
        if sc >= a.count_thr:
            r, g, bb = COLORS[(l - 1) % len(COLORS)]
            cv2.rectangle(ann, (int(x0), int(y0)), (int(x1), int(y1)), (bb, g, r), max(2, img.shape[1] // 1500))
    cv2.imwrite(os.path.join(a.out, f"{name}_annotated.jpg"), ann, [cv2.IMWRITE_JPEG_QUALITY, 88])
    with open(os.path.join(a.out, f"{name}_counts.json"), "w") as f:
        json.dump(res, f, indent=2)
    print(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
