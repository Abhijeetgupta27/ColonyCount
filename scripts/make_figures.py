"""Regenerate every README figure from your checkpoint and results.

    python scripts/make_figures.py --ckpt runs/det/best.pt --log runs/det/log.csv \
        --results results --count-thr 0.6
"""
import argparse
import json
import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from colonycount.data import NOT_COUNTED, load_coco, plate_status, read_rgb  # noqa: E402
from colonycount.infer import detect, starts  # noqa: E402
from colonycount.model import load_detector, pick_device  # noqa: E402

SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#6250d6", "#e34948"]
NEUTRAL = "#8a8985"
INK, INK2, GRID, BG = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"
STATUS = ["empty", "countable", "uncountable"]
plt.rcParams.update({
    "figure.facecolor": BG, "axes.facecolor": BG, "savefig.facecolor": BG, "font.size": 10,
    "axes.titlesize": 11, "axes.titleweight": "bold", "text.color": INK, "axes.labelcolor": INK2,
    "xtick.color": INK2, "ytick.color": INK2, "axes.edgecolor": GRID, "axes.spines.top": False,
    "axes.spines.right": False, "axes.grid": True, "grid.color": GRID, "axes.axisbelow": True,
})


def palette(cats):
    """Species get categorical colours in fixed order; defects/contamination stay neutral grey."""
    out, k = {}, 0
    for c in cats:
        if any(n in c["name"].lower() for n in NOT_COUNTED):
            out[c["id"]] = NEUTRAL
        else:
            out[c["id"]] = SERIES[k % len(SERIES)]
            k += 1
    return out


def draw(ax, img, boxes, colors, title, lw=1.2):
    ax.imshow(img)
    for (x, y, w, h), col in zip(boxes, colors):
        ax.add_patch(plt.Rectangle((x, y), w, h, fill=False, lw=lw, ec=col))
    ax.set_title(title)
    ax.axis("off")


def save(fig, path):
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)
    print("wrote", path)


def legend(fig, cats, col):
    from matplotlib.patches import Patch
    h = [Patch(fc="none", ec=col[c["id"]], lw=2, label=c["name"]) for c in cats]
    fig.legend(handles=h, loc="lower center", ncol=len(h), frameon=False, fontsize=9)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/agar")
    ap.add_argument("--img-root", default="raw/agar")
    ap.add_argument("--ckpt", default="runs/det/best.pt")
    ap.add_argument("--log", default="runs/det/log.csv")
    ap.add_argument("--results", default="results")
    ap.add_argument("--count-thr", type=float, default=0.5)
    ap.add_argument("--out", default="assets")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    o = lambda n: os.path.join(a.out, n)  # noqa: E731

    c, anns = load_coco(os.path.join(a.data, "test.json"))
    cats = c["categories"]
    col = palette(cats)
    st = {im["id"]: plate_status(im, len(anns[im["id"]])) for im in c["images"]}

    # 1. example plates: one per status, GT boxes coloured by species
    pick = []
    for s in (1, 1, 0, 2):
        cand = [im for im in c["images"] if st[im["id"]] == s and im not in pick]
        if cand:
            pick.append(cand[0] if s != 1 or len(pick) == 0 else max(cand, key=lambda i: len(anns[i["id"]])))
    fig, ax = plt.subplots(1, len(pick), figsize=(3.6 * len(pick), 4.2))
    for axi, im in zip(np.atleast_1d(ax), pick):
        n = im.get("colonies_number", len(anns[im["id"]]))
        draw(axi, read_rgb(os.path.join(a.img_root, im["file_name"])), [b["bbox"] for b in anns[im["id"]]],
             [col[b["category_id"]] for b in anns[im["id"]]],
             f"{STATUS[st[im['id']]]} ({im.get('background', '?')})\n" + ("colonies: " + str(n) if n >= 0 else "colonies: too many"))
    legend(fig, cats, col)
    fig.subplots_adjust(left=0.01, right=0.99, top=0.86, bottom=0.1, wspace=0.04)
    fig.savefig(o("step1_plates.png"), dpi=110)
    plt.close(fig)
    print("wrote", o("step1_plates.png"))

    # 2. training patches
    pj = os.path.join(a.data, "patches", "train", "patches.json")
    if os.path.exists(pj):
        pc, pa = load_coco(pj)
        ims = [im for im in pc["images"] if pa[im["id"]]]
        ims = ims[:: max(1, len(ims) // 4)][:4]  # spread across different plates
        fig, ax = plt.subplots(1, len(ims), figsize=(3 * len(ims), 3.4))
        for axi, im in zip(np.atleast_1d(ax), ims):
            draw(axi, read_rgb(os.path.join(a.data, "patches", "train", "img", im["file_name"])),
                 [b["bbox"] for b in pa[im["id"]]], [col.get(b["category_id"], NEUTRAL) for b in pa[im["id"]]],
                 f"patch {im['width']}×{im['height']} px, {len(pa[im['id']])} boxes", lw=1.5)
        save(fig, o("step2_patches.png"))

    # 3. training curves (separate panels, never a dual axis)
    if os.path.exists(a.log):
        lg = pd.read_csv(a.log)
        fig, ax = plt.subplots(1, 3, figsize=(12, 3.3))
        for axi, k, t in [(ax[0], "loss", "Training loss (RPN + box heads)"),
                          (ax[1], "val_map", "Validation mAP@[.5:.95]"),
                          (ax[2], "val_count_mae", "Validation count MAE (colonies)")]:
            axi.plot(lg.epoch, lg[k], color=SERIES[0], lw=2, marker="o", ms=4)
            axi.set(title=t, xlabel="epoch")
        save(fig, o("step3_training.png"))

    # 4 + 5. tiling + detection on the densest test plate
    dev = pick_device()
    model, ck = load_detector(a.ckpt, dev)
    im = max([i for i in c["images"] if st[i["id"]] == 1], key=lambda i: len(anns[i["id"]]))
    img = read_rgb(os.path.join(a.img_root, im["file_name"]))
    h, w = img.shape[:2]
    tile, ov = 512, 128
    fig, ax = plt.subplots(1, 2, figsize=(10, 4.8))
    ax[0].imshow(img)
    for y in starts(h, tile, tile - ov):
        for x in starts(w, tile, tile - ov):
            ax[0].add_patch(plt.Rectangle((x, y), tile, tile, fill=False, lw=1.2, ec=SERIES[0]))
    ax[0].set_title(f"{w}×{h} plate → {len(starts(h, tile, tile - ov)) * len(starts(w, tile, tile - ov))} overlapping tiles")
    ax[0].axis("off")
    b, s, lab = detect(model, img, tile=tile, overlap=ov, device=dev, merge_thr=1.01)  # no merging
    b2, s2, lab2 = detect(model, img, tile=tile, overlap=ov, device=dev)
    raw, kept = int((s >= a.count_thr).sum()), int((s2 >= a.count_thr).sum())
    ax[1].bar(["before merge", "after merge", "ground truth"],
              [raw, kept, len([x for x in anns[im["id"]] if col[x["category_id"]] != NEUTRAL])],
              color=["#c3c2b7", SERIES[0], INK2], width=0.55)
    ax[1].set(title="Boxes on this plate (tile duplicates removed)", ylabel="boxes")
    ax[1].grid(axis="x", visible=False)
    save(fig, o("step4_tiling.png"))

    keep = s2 >= a.count_thr
    pb = [[x0, y0, x1 - x0, y1 - y0] for x0, y0, x1, y1 in b2[keep].tolist()]
    name2cid = {cat["name"]: cat["id"] for cat in cats}
    pcol = [col[name2cid[ck["classes"][i - 1]]] for i in lab2[keep].tolist()]
    cy, cx = h // 2, w // 2
    half = min(h, w) // 4
    sl = (slice(cy - half, cy + half), slice(cx - half, cx + half))
    shift = lambda bb: [[x - (cx - half), y - (cy - half), ww, hh] for x, y, ww, hh in bb]  # noqa: E731
    fig, ax = plt.subplots(1, 2, figsize=(10, 5.4))
    gb = [x["bbox"] for x in anns[im["id"]]]
    draw(ax[0], img[sl], shift(gb), [col[x["category_id"]] for x in anns[im["id"]]], "Ground truth (zoomed centre)", 1.5)
    draw(ax[1], img[sl], shift(pb), pcol, f"Prediction (score ≥ {a.count_thr})", 1.5)
    for axi in ax:
        axi.set_xlim(0, 2 * half)
        axi.set_ylim(2 * half, 0)
    legend(fig, cats, col)
    fig.subplots_adjust(left=0.01, right=0.99, top=0.93, bottom=0.09, wspace=0.04)
    fig.savefig(o("step5_detection.png"), dpi=110)
    plt.close(fig)
    print("wrote", o("step5_detection.png"))

    # 6. counting accuracy + per-class AP
    cf, sf = os.path.join(a.results, "counts.csv"), os.path.join(a.results, "summary.json")
    if os.path.exists(cf) and os.path.exists(sf):
        cc = pd.read_csv(cf)
        sm = json.load(open(sf))
        fig, ax = plt.subplots(1, 2, figsize=(11, 4.2))
        m = max(cc.true.max(), cc.pred.max()) * 1.05 + 1
        ax[0].plot([0, m], [0, m], color=INK2, lw=1, ls="--")
        ax[0].scatter(cc.true, cc.pred, s=28, color=SERIES[0], edgecolor=BG, lw=1)
        ax[0].set(xlim=(0, m), ylim=(0, m), xlabel="true colonies", ylabel="predicted colonies",
                  title=f"Counting: MAE {sm['count_mae']:.2f}, sMAPE {sm['count_smape']:.1f}%")
        pcs = sm["per_class"]
        names = list(pcs)
        vals = [pcs[n] for n in names]
        bc = [NEUTRAL if any(k in n.lower() for k in NOT_COUNTED) else SERIES[0] for n in names]
        ax[1].barh(names, vals, color=bc, height=0.6)
        for i, v in enumerate(vals):
            ax[1].text(v + 0.01, i, f"{v:.2f}", va="center", color=INK2, fontsize=9)
        ax[1].set(xlim=(0, 1), xlabel="AP@[.5:.95]", title=f"Per-class AP (overall mAP {sm['map']:.3f})")
        ax[1].grid(axis="y", visible=False)
        save(fig, o("step6_results.png"))

        # 7. plate classifier confusion matrix
        if "plate_classifier" in sm:
            cm = np.array(sm["plate_classifier"]["cm"])
            fig, axi = plt.subplots(figsize=(4.6, 4))
            axi.imshow(cm, cmap="Blues")
            for i in range(3):
                for j in range(3):
                    axi.text(j, i, cm[i, j], ha="center", va="center",
                             color="white" if cm[i, j] > cm.max() / 2 else INK)
            axi.set_xticks(range(3), STATUS)
            axi.set_yticks(range(3), STATUS)
            axi.set(xlabel="predicted", ylabel="true",
                    title=f"Plate classifier (acc {sm['plate_classifier']['acc']:.2f})")
            axi.grid(False)
            save(fig, o("step7_plate_classifier.png"))


if __name__ == "__main__":
    main()
