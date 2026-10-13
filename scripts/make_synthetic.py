"""Generate synthetic Petri-dish photos in an AGAR-like layout (image + json per plate).

ONLY for testing that the pipeline runs end to end without the real dataset.
Numbers from synthetic plates mean nothing about real colony detection.

    python scripts/make_synthetic.py --out raw/synthetic --n 160
"""
import argparse
import json
import os

import cv2
import numpy as np

# name: (BGR colour, radius range, irregular edge?)
SPECIES = {
    "S.aureus": ((60, 190, 230), (6, 10), False),
    "B.subtilis": ((205, 215, 220), (14, 22), True),
    "P.aeruginosa": ((120, 170, 110), (9, 14), False),
    "E.coli": ((150, 175, 190), (8, 12), False),
    "C.albicans": ((240, 245, 248), (8, 13), False),
}


def blob(img, cx, cy, r, col, irregular, rng):
    if irregular:
        t = np.linspace(0, 2 * np.pi, 40, endpoint=False)
        rr = r * (1 + 0.18 * rng.standard_normal(40))
        pts = np.stack([cx + rr * np.cos(t), cy + rr * np.sin(t)], 1).astype(np.int32)
        cv2.fillPoly(img, [pts], col, cv2.LINE_AA)
    else:
        cv2.circle(img, (cx, cy), r, col, -1, cv2.LINE_AA)
    # darker rim + highlight so colonies look domed, not flat discs
    cv2.circle(img, (cx, cy), r, tuple(int(c * 0.8) for c in col), 1, cv2.LINE_AA)
    cv2.circle(img, (cx - r // 3, cy - r // 3), max(1, r // 4), tuple(min(255, int(c * 1.1) + 10) for c in col), -1, cv2.LINE_AA)
    rx = ry = r + 1
    return [cx - rx, cy - ry, 2 * rx, 2 * ry]


def plate(rng, size=1200, kind="countable", bg="bright"):
    img = np.full((size, size, 3), 235 if bg == "bright" else 25, np.uint8)
    c, R = size // 2, int(size * 0.44)
    agar = np.array([150, 185, 205]) + rng.integers(-15, 15, 3)
    cv2.circle(img, (c, c), R + 8, (200, 200, 200) if bg == "bright" else (70, 70, 70), -1, cv2.LINE_AA)
    cv2.circle(img, (c, c), R, tuple(int(v) for v in agar), -1, cv2.LINE_AA)
    labels, n = [], 0

    def spot():
        while True:
            x, y = rng.integers(c - R + 30, c + R - 30, 2)
            if (x - c) ** 2 + (y - c) ** 2 < (R - 30) ** 2:
                return int(x), int(y)

    if kind == "countable":
        sp = rng.choice(list(SPECIES), size=rng.integers(1, 3), replace=False)
        for s in sp:
            col, (r0, r1), irr = SPECIES[s]
            for _ in range(int(rng.integers(5, 60))):
                x, y = spot()
                r = int(rng.integers(r0, r1 + 1))
                bx = blob(img, x, y, r, col, irr, rng)
                labels.append({"x": bx[0], "y": bx[1], "width": bx[2], "height": bx[3], "class": s})
                n += 1
        if rng.random() < 0.3:  # a defect: air bubble
            x, y = spot()
            cv2.circle(img, (x, y), 12, (90, 90, 90), 2, cv2.LINE_AA)
            labels.append({"x": x - 14, "y": y - 14, "width": 28, "height": 28, "class": "Defect"})
        if rng.random() < 0.15:  # contamination: big fuzzy mould patch
            x, y = spot()
            m = np.zeros(img.shape[:2], np.uint8)
            cv2.circle(m, (x, y), 30, 255, -1)
            m = cv2.GaussianBlur(m, (31, 31), 10)[..., None] / 255.0
            img[:] = (img * (1 - m) + np.array([110, 120, 100]) * m).astype(np.uint8)
            labels.append({"x": x - 35, "y": y - 35, "width": 70, "height": 70, "class": "Contamination"})
    elif kind == "uncountable":  # confluent lawn: too many to count
        s = rng.choice(list(SPECIES))
        col, (r0, _), irr = SPECIES[s]
        for _ in range(900):
            x, y = spot()
            blob(img, x, y, r0, col, False, rng)
        n = -1
    img = cv2.GaussianBlur(img, (3, 3), 0)
    img = np.clip(img + rng.normal(0, 4, img.shape), 0, 255).astype(np.uint8)
    return img, {"background": bg, "colonies_number": n, "labels": labels}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="raw/synthetic")
    ap.add_argument("--n", type=int, default=160)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    rng = np.random.default_rng(a.seed)
    for i in range(a.n):
        kind = rng.choice(["countable", "empty", "uncountable"], p=[0.7, 0.15, 0.15])
        img, meta = plate(rng, kind=kind, bg=rng.choice(["bright", "dark"]))
        cv2.imwrite(os.path.join(a.out, f"{i}.jpg"), img, [cv2.IMWRITE_JPEG_QUALITY, 92])
        with open(os.path.join(a.out, f"{i}.json"), "w") as f:
            json.dump({"sample_id": i, **meta}, f)
    print(f"wrote {a.n} synthetic plates to {a.out}")


if __name__ == "__main__":
    main()
