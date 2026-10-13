"""Detection (mAP via pycocotools) and counting (MAE, sMAPE) metrics."""
import contextlib
import io

import numpy as np


def coco_map(gt_coco, results, max_det=1000):
    """gt_coco: COCO dict; results: list of {image_id, category_id, bbox xywh, score}.

    pycocotools caps detections at 100 per image by default - far too few for a
    crowded plate - so the cap is raised. Returns overall + per-class AP.
    """
    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval
    with contextlib.redirect_stdout(io.StringIO()):
        g = COCO()
        g.dataset = gt_coco
        g.createIndex()
        if not results:
            return {"map": 0.0, "map50": 0.0, "map75": 0.0, "per_class": {}}
        d = g.loadRes(results)
        e = COCOeval(g, d, "bbox")
        e.params.maxDets = [10, 100, max_det]
        e.evaluate()
        e.accumulate()
        e.summarize()
    out = {"map": float(e.stats[0]), "map50": float(e.stats[1]), "map75": float(e.stats[2]), "per_class": {}}
    prec = e.eval["precision"]  # [iou, recall, class, area, maxdet]
    for k, cid in enumerate(e.params.catIds):
        p = prec[:, :, k, 0, -1]
        p = p[p > -1]
        out["per_class"][g.cats[cid]["name"]] = float(p.mean()) if p.size else float("nan")
    return out


def count_errors(pred, true):
    """MAE and sMAPE (in %) between predicted and true colony counts.

    sMAPE here = mean(|p - t| / ((|p| + |t|) / 2)) * 100, with 0/0 counted as 0.
    Papers use slightly different sMAPE variants, so compare across papers with care.
    """
    p, t = np.asarray(pred, float), np.asarray(true, float)
    d = np.abs(p - t)
    den = (np.abs(p) + np.abs(t)) / 2
    sm = np.where(den > 0, d / np.where(den > 0, den, 1), 0)
    return {"mae": float(d.mean()) if len(d) else 0.0, "smape": float(sm.mean() * 100) if len(d) else 0.0}
