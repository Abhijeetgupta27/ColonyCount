"""Whole-plate inference: sliding window + merge of duplicate boxes.

AGAR plates are up to 6000 x 4000 px; the detector sees 512-px tiles. A colony on a
tile border appears twice - once whole, once cut. Plain IoU-NMS often keeps both
(a half box vs a whole box has IoU ~0.5), so we merge with intersection-over-the-
SMALLER-box: a cut copy lies inside the whole one, so its IoMin is ~1. Boxes that
touch an interior tile edge are ranked below complete ones, so the whole copy wins.
"""
import numpy as np
import torch

from .data import to_tensor


def starts(n, tile, stride):
    if n <= tile:
        return [0]
    s = list(range(0, n - tile, stride))
    return s + [n - tile]


def iomin(a, b):
    """Intersection over the smaller box's area, pairwise (N, M)."""
    lt = torch.maximum(a[:, None, :2], b[None, :, :2])
    rb = torch.minimum(a[:, None, 2:], b[None, :, 2:])
    inter = (rb - lt).clamp(min=0).prod(-1)
    area = lambda x: (x[:, 2:] - x[:, :2]).clamp(min=0).prod(-1)  # noqa: E731
    return inter / torch.minimum(area(a)[:, None], area(b)[None, :]).clamp(min=1e-6)


def merge(boxes, scores, labels, cut, thr=0.7):
    """Greedy suppression: complete boxes first, then by score; drop boxes mostly inside a kept one."""
    if len(boxes) == 0:
        return boxes, scores, labels
    order = torch.argsort(scores - cut.float() * 10, descending=True)  # cut boxes rank last
    boxes, scores, labels = boxes[order], scores[order], labels[order]
    ov = iomin(boxes, boxes)
    keep = torch.ones(len(boxes), dtype=torch.bool)
    for i in range(len(boxes)):
        if keep[i]:
            dup = ov[i] > thr
            dup[: i + 1] = False
            keep &= ~dup
    return boxes[keep], scores[keep], labels[keep]


@torch.no_grad()
def detect(model, img, tile=512, overlap=128, batch=4, device="cpu", score_thr=0.05, merge_thr=0.7):
    """img: (H, W, 3) uint8 -> boxes (N, 4) xyxy, scores (N,), labels (N,) on the full plate.

    `overlap` should exceed the largest colony, so every colony is whole in some tile.
    """
    h, w = img.shape[:2]
    pos = [(y, x) for y in starts(h, tile, tile - overlap) for x in starts(w, tile, tile - overlap)]
    B, S, L, C = [], [], [], []
    for i in range(0, len(pos), batch):
        chunk = pos[i:i + batch]
        x = [to_tensor(img[y:y + tile, x:x + tile]).to(device) for y, x in chunk]
        for (y0, x0), o in zip(chunk, model(x)):
            b, s, lab = o["boxes"].cpu(), o["scores"].cpu(), o["labels"].cpu()
            k = s >= score_thr
            b, s, lab = b[k], s[k], lab[k]
            th, tw = min(tile, h - y0), min(tile, w - x0)
            # touching an interior tile edge = possibly a cut colony
            e = 2
            cut = ((b[:, 0] < e) & (x0 > 0)) | ((b[:, 1] < e) & (y0 > 0)) | \
                  ((b[:, 2] > tw - e) & (x0 + tw < w)) | ((b[:, 3] > th - e) & (y0 + th < h))
            B.append(b + torch.tensor([x0, y0, x0, y0], dtype=b.dtype))
            S.append(s)
            L.append(lab)
            C.append(cut)
    if not B:
        return torch.zeros((0, 4)), torch.zeros(0), torch.zeros(0, dtype=torch.long)
    return merge(torch.cat(B), torch.cat(S), torch.cat(L), torch.cat(C), merge_thr)


def count(labels, scores, classes, thr):
    """Per-class colony counts above a score threshold, skipping defects/contamination."""
    from .data import NOT_COUNTED
    out = {}
    for i, name in enumerate(classes, 1):
        if any(k in name.lower() for k in NOT_COUNTED):
            continue
        out[name] = int(((labels == i) & (scores >= thr)).sum())
    out["total"] = sum(out.values())
    return out


def plate_input(img, size=384):
    """Downscale a whole plate for the plate classifier (ImageNet normalisation)."""
    import cv2
    x = cv2.resize(img, (size, size), interpolation=cv2.INTER_AREA).astype(np.float32) / 255
    x = (x - [0.485, 0.456, 0.406]) / [0.229, 0.224, 0.225]
    return torch.from_numpy(x.transpose(2, 0, 1)).float()
