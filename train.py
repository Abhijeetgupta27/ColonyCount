"""Fine-tune a COCO-pretrained Faster R-CNN on colony patches.

    python scripts/make_patches.py --data data/agar --img-root raw/agar
    python train.py --data data/agar --img-root raw/agar --epochs 12
"""
import argparse
import csv
import json
import math
import os
import time

import torch
from torch.utils.data import DataLoader

from colonycount.data import PatchSet, collate, load_coco
from colonycount.evalplates import run, score
from colonycount.model import detector, pick_device, save


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/agar")
    ap.add_argument("--img-root", default="raw/agar")
    ap.add_argument("--out", default="runs/det")
    ap.add_argument("--arch", default="r50", choices=["r50", "mobilenet"])
    ap.add_argument("--epochs", type=int, default=12)
    ap.add_argument("--bs", type=int, default=4)
    ap.add_argument("--lr", type=float, default=0.01)
    ap.add_argument("--size", type=int, default=512)
    ap.add_argument("--workers", type=int, default=0)
    ap.add_argument("--val-plates", type=int, default=20, help="whole plates scored each epoch")
    ap.add_argument("--max-steps", type=int, default=0, help="cap steps per epoch (0 = all)")
    ap.add_argument("--cpu", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    torch.manual_seed(a.seed)
    dev = pick_device(a.cpu)
    os.makedirs(a.out, exist_ok=True)
    pdir = os.path.join(a.data, "patches", "train")
    ds = PatchSet(os.path.join(pdir, "patches.json"), os.path.join(pdir, "img"))
    classes = [c["name"] for c in load_coco(os.path.join(pdir, "patches.json"))[0]["categories"]]
    dl = DataLoader(ds, batch_size=a.bs, shuffle=True, num_workers=a.workers, collate_fn=collate, drop_last=True)
    steps = min(len(dl), a.max_steps) if a.max_steps else len(dl)
    print(f"{len(ds)} patches | {steps} steps/epoch | classes {classes} | device {dev}")

    model = detector(len(classes) + 1, a.arch, a.size).to(dev)
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.SGD(params, lr=a.lr, momentum=0.9, weight_decay=1e-4)
    total, warm = a.epochs * steps, min(500, steps)
    sched = torch.optim.lr_scheduler.LambdaLR(  # linear warm-up then cosine decay
        opt, lambda i: (i + 1) / warm if i < warm else 0.5 * (1 + math.cos(math.pi * (i - warm) / max(1, total - warm))))

    best, log = -1.0, []
    for ep in range(1, a.epochs + 1):
        model.train()
        t0, tot = time.time(), 0.0
        for k, (x, t) in enumerate(dl):
            if k >= steps:
                break
            x = [i.to(dev) for i in x]
            t = [{kk: v.to(dev) for kk, v in d.items()} for d in t]
            loss = sum(model(x, t).values())
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, 10.0)
            opt.step()
            sched.step()
            tot += loss.item()
        model.eval()
        dets, gt = run(model, classes, os.path.join(a.data, "val.json"), a.img_root, dev, limit=a.val_plates)
        m = score(dets, gt)
        row = {"epoch": ep, "loss": round(tot / steps, 4), "val_map": round(m["map"], 4),
               "val_map50": round(m["map50"], 4), "val_count_mae": round(m["count_mae"], 2),
               "sec": round(time.time() - t0, 1)}
        log.append(row)
        print(json.dumps(row), flush=True)
        meta = dict(classes=classes, arch=a.arch, size=a.size, epoch=ep, val_map=m["map"])
        save(os.path.join(a.out, "last.pt"), model, **meta)
        if m["map"] > best:
            best = m["map"]
            save(os.path.join(a.out, "best.pt"), model, **meta)
        with open(os.path.join(a.out, "log.csv"), "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=row.keys())
            w.writeheader()
            w.writerows(log)
    print(f"best val mAP@[.5:.95] {best:.4f}")


if __name__ == "__main__":
    main()
