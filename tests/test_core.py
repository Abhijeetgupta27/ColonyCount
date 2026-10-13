import json

import cv2
import numpy as np
import torch

from colonycount.data import augment, make_patches, plate_status
from colonycount.infer import count, detect, iomin, merge
from colonycount.metrics import coco_map, count_errors


def test_iomin_contained_box_is_one():
    a = torch.tensor([[0., 0., 10., 10.]])
    b = torch.tensor([[0., 0., 5., 10.], [20., 20., 30., 30.]])
    o = iomin(a, b)[0]
    assert abs(o[0] - 1) < 1e-6 and o[1] == 0


def test_merge_drops_cut_copy_keeps_neighbours():
    boxes = torch.tensor([[0., 0., 20., 20.], [10., 0., 20., 20.], [40., 40., 60., 60.]])
    scores = torch.tensor([0.6, 0.9, 0.8])  # the cut copy even has the higher score
    cut = torch.tensor([False, True, False])
    b, s, _ = merge(boxes, scores, torch.ones(3, dtype=torch.long), cut)
    assert len(b) == 2
    assert any(torch.equal(x, boxes[0]) for x in b) and not any(torch.equal(x, boxes[1]) for x in b)


class Oracle:
    """Fake detector: one box per connected dark blob in each tile."""
    def __call__(self, xs):
        out = []
        for x in xs:
            g = (x.mean(0).numpy() < 0.5).astype(np.uint8)
            n, _, st, _ = cv2.connectedComponentsWithStats(g)
            b = [[s[0], s[1], s[0] + s[2], s[1] + s[3]] for s in st[1:]]
            b = torch.tensor(b, dtype=torch.float32).reshape(-1, 4)
            out.append({"boxes": b, "scores": torch.ones(len(b)), "labels": torch.ones(len(b), dtype=torch.long)})
        return out


def test_sliding_window_counts_each_colony_once():
    rng = np.random.default_rng(0)
    img = np.full((1500, 1300, 3), 255, np.uint8)
    pts = [(y, x) for y in range(40, 1460, 70) for x in range(40, 1260, 70)]
    for y, x in pts:  # many colonies land on tile borders
        cv2.circle(img, (int(x + rng.integers(-8, 9)), int(y + rng.integers(-8, 9))), int(rng.integers(8, 20)), (0, 0, 0), -1)
    b, s, lab = detect(Oracle(), img, tile=512, overlap=128)
    assert len(b) == len(pts)


def test_count_skips_defects():
    labels = torch.tensor([1, 1, 2, 3])
    scores = torch.tensor([0.9, 0.2, 0.9, 0.9])
    c = count(labels, scores, ["E.coli", "S.aureus", "Defect"], thr=0.5)
    assert c == {"E.coli": 1, "S.aureus": 1, "total": 2}


def test_count_errors():
    e = count_errors([10, 0, 5], [10, 0, 10])
    assert abs(e["mae"] - 5 / 3) < 1e-9
    assert abs(e["smape"] - (5 / 7.5) / 3 * 100) < 1e-9


def test_map_perfect():
    gt = {"images": [{"id": 1, "width": 100, "height": 100}], "categories": [{"id": 1, "name": "E.coli"}],
          "annotations": [{"id": 1, "image_id": 1, "category_id": 1, "bbox": [10, 10, 20, 20], "area": 400, "iscrowd": 0}]}
    m = coco_map(gt, [{"image_id": 1, "category_id": 1, "bbox": [10, 10, 20, 20], "score": 0.9}])
    assert m["map"] > 0.99


def test_augment_keeps_boxes_on_their_objects():
    img = np.zeros((100, 120, 3), np.uint8)
    img[20:40, 70:90] = 255
    b = np.array([[70, 20, 90, 40]], np.float32)
    for seed in range(8):
        im2, b2 = augment(img, b.copy(), np.random.RandomState(seed))
        x0, y0, x1, y1 = b2[0].astype(int)
        assert im2[y0:y1, x0:x1].mean() > 150  # still covers the bright square
        assert im2.mean() > 0


def test_plate_status():
    assert plate_status({"colonies_number": -1}, 0) == 2
    assert plate_status({"colonies_number": 0}, 0) == 0
    assert plate_status({}, 5) == 1


def test_make_patches(tmp_path):
    img = np.full((800, 900, 3), 200, np.uint8)
    cv2.imwrite(str(tmp_path / "p.jpg"), img)
    coco = {"images": [{"id": 1, "file_name": "p.jpg", "width": 900, "height": 800, "colonies_number": 2}],
            "categories": [{"id": 1, "name": "E.coli"}],
            "annotations": [{"id": 1, "image_id": 1, "bbox": [100, 100, 20, 20], "category_id": 1},
                            {"id": 2, "image_id": 1, "bbox": [700, 600, 30, 30], "category_id": 1}]}
    (tmp_path / "c.json").write_text(json.dumps(coco))
    n, nb = make_patches(str(tmp_path / "c.json"), str(tmp_path), str(tmp_path / "out"), size=256, per_img=6)
    out = json.loads((tmp_path / "out" / "patches.json").read_text())
    assert n >= 6 and nb >= 6
    for a in out["annotations"]:
        x, y, w, h = a["bbox"]
        assert x >= 0 and y >= 0 and x + w <= 256 and y + h <= 256
