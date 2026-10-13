"""Run the detector over whole plates and score detection + counting."""
import os

import numpy as np

from .data import counted_ids, load_coco, plate_status, read_rgb
from .infer import detect
from .metrics import coco_map, count_errors


def run(model, classes, coco_path, img_root, device, limit=None, score_thr=0.05, tile=512, overlap=128):
    """Returns (detections per plate, gt coco restricted to countable plates)."""
    c, anns = load_coco(coco_path)
    ims = [im for im in c["images"] if plate_status(im, len(anns[im["id"]])) == 1]
    if limit:
        ims = ims[:limit]
    name2cid = {cat["name"]: cat["id"] for cat in c["categories"]}
    dets = []
    for im in ims:
        img = read_rgb(os.path.join(img_root, im["file_name"]))
        b, s, lab = detect(model, img, tile=tile, overlap=overlap, device=device, score_thr=score_thr)
        dets.append({"image": im, "boxes": b.numpy(), "scores": s.numpy(),
                     "cids": np.array([name2cid[classes[i - 1]] for i in lab.tolist()], int)})
    keep = {im["id"] for im in ims}
    gt = {"images": ims, "categories": c["categories"],
          "annotations": [a for a in c["annotations"] if a["image_id"] in keep]}
    return dets, gt


def score(dets, gt, count_thr=0.5):
    res = []
    for d in dets:
        for b, s, cid in zip(d["boxes"], d["scores"], d["cids"]):
            res.append({"image_id": d["image"]["id"], "category_id": int(cid),
                        "bbox": [float(b[0]), float(b[1]), float(b[2] - b[0]), float(b[3] - b[1])], "score": float(s)})
    m = coco_map(gt, res)
    m.update(counts(dets, gt, count_thr))
    return m


def counts(dets, gt, thr):
    ok = counted_ids(gt["categories"])
    true = {}
    for a in gt["annotations"]:
        if a["category_id"] in ok:
            true[a["image_id"]] = true.get(a["image_id"], 0) + 1
    p = [int(((d["scores"] >= thr) & np.isin(d["cids"], list(ok))).sum()) for d in dets]
    t = [true.get(d["image"]["id"], 0) for d in dets]
    e = count_errors(p, t)
    return {"count_mae": e["mae"], "count_smape": e["smape"], "pred_counts": p, "true_counts": t}
