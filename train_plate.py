"""Train the plate classifier: empty / countable / uncountable.

Labs need this first step - an uncountable (confluent) plate should be flagged,
not given a meaningless count, and empty plates need no detector pass.

    python train_plate.py --data data/agar --img-root raw/agar --epochs 8
"""
import argparse
import json
import os

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset

from colonycount.data import load_coco, plate_status
from colonycount.infer import plate_input
from colonycount.model import pick_device, plate_classifier, save

NAMES = ["empty", "countable", "uncountable"]


def thumbs(coco_path, img_root, cache, size=448):
    """Downscaled copies of every plate, made once (full-res decoding is slow)."""
    c, anns = load_coco(coco_path)
    os.makedirs(cache, exist_ok=True)
    out = []
    for im in c["images"]:
        p = os.path.join(cache, f"{os.path.splitext(os.path.basename(coco_path))[0]}_{im['id']}.jpg")
        if not os.path.exists(p):
            src = Image.open(os.path.join(img_root, im["file_name"]))
            src.draft("RGB", (size * 2, size * 2))  # fast JPEG downscaled decode
            src.convert("RGB").resize((size, size), Image.BILINEAR).save(p, quality=92)
        out.append((p, plate_status(im, len(anns[im["id"]]))))
    return out


class Plates(Dataset):
    def __init__(self, items, aug):
        self.items, self.aug = items, aug

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        p, y = self.items[i]
        img = np.array(Image.open(p).convert("RGB"))
        if self.aug:
            img = np.rot90(img, np.random.randint(4))
            if np.random.rand() < 0.5:
                img = img[:, ::-1]
            img = np.clip(img * np.random.uniform(0.8, 1.2) + np.random.uniform(-20, 20), 0, 255).astype(np.uint8)
        return plate_input(np.ascontiguousarray(img)), y


def evaluate(model, dl, dev):
    model.eval()
    cm = np.zeros((3, 3), int)
    with torch.no_grad():
        for x, y in dl:
            p = model(x.to(dev)).argmax(1).cpu()
            for t, q in zip(y.tolist(), p.tolist()):
                cm[t, q] += 1
    f1 = []
    for k in range(3):
        tp, fp, fn = cm[k, k], cm[:, k].sum() - cm[k, k], cm[k].sum() - cm[k, k]
        if tp + fp + fn:  # skip classes absent from both truth and predictions
            f1.append(2 * tp / (2 * tp + fp + fn))
    return {"acc": float(np.trace(cm) / max(cm.sum(), 1)), "macro_f1": float(np.mean(f1)), "cm": cm.tolist()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/agar")
    ap.add_argument("--img-root", default="raw/agar")
    ap.add_argument("--out", default="runs/plate")
    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--bs", type=int, default=16)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--cpu", action="store_true")
    a = ap.parse_args()

    dev = pick_device(a.cpu)
    os.makedirs(a.out, exist_ok=True)
    cache = os.path.join(a.data, "thumbs")
    tr = thumbs(os.path.join(a.data, "train.json"), a.img_root, cache)
    va = thumbs(os.path.join(a.data, "val.json"), a.img_root, cache)
    freq = np.bincount([y for _, y in tr], minlength=3)
    print("train plates per class:", dict(zip(NAMES, freq.tolist())), "| device", dev)
    w = torch.tensor(freq.sum() / np.maximum(freq, 1) / 3, dtype=torch.float32, device=dev)  # class balance

    model = plate_classifier().to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=1e-4)
    dl = DataLoader(Plates(tr, True), batch_size=a.bs, shuffle=True)
    vl = DataLoader(Plates(va, False), batch_size=a.bs)
    best = -1
    for ep in range(1, a.epochs + 1):
        model.train()
        for x, y in dl:
            loss = F.cross_entropy(model(x.to(dev)), y.to(dev), weight=w)
            opt.zero_grad()
            loss.backward()
            opt.step()
        m = evaluate(model, vl, dev)
        print(json.dumps({"epoch": ep, "val_acc": round(m["acc"], 4), "val_macro_f1": round(m["macro_f1"], 4)}), flush=True)
        if m["macro_f1"] > best:
            best = m["macro_f1"]
            save(os.path.join(a.out, "best.pt"), model, names=NAMES, val=m)
    print(f"best val macro-F1 {best:.4f}")


if __name__ == "__main__":
    main()
