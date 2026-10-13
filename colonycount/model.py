"""Models: a Faster R-CNN colony detector and a ResNet plate classifier.

Both start from ImageNet / COCO-pretrained weights and are fine-tuned (transfer
learning) - AGAR is large, but pretrained features still converge much faster.
"""
import os

os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")  # some detection ops lack MPS kernels

import torch  # noqa: E402
import torch.nn as nn  # noqa: E402
import torchvision  # noqa: E402
from torchvision.models.detection import (fasterrcnn_mobilenet_v3_large_fpn,  # noqa: E402
                                          fasterrcnn_resnet50_fpn_v2)
from torchvision.models.detection.anchor_utils import AnchorGenerator  # noqa: E402
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor  # noqa: E402


def pick_device(cpu=False):
    if cpu:
        return "cpu"
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def detector(n_cls, arch="r50", size=512, pretrained=True, max_det=300):
    """n_cls includes background. Small anchors: colonies are often < 32 px."""
    kw = dict(min_size=size, max_size=size, box_detections_per_img=max_det)
    if arch == "r50":
        m = fasterrcnn_resnet50_fpn_v2(weights="DEFAULT" if pretrained else None, **kw)
        # COCO anchors start at 32 px; shift every level down one octave for small colonies
        m.rpn.anchor_generator = AnchorGenerator(((16,), (32,), (64,), (128,), (256,)), ((0.5, 1.0, 2.0),) * 5)
    elif arch == "mobilenet":  # ~4x faster, less accurate - good for laptops
        m = fasterrcnn_mobilenet_v3_large_fpn(weights="DEFAULT" if pretrained else None, **kw)
    else:
        raise ValueError(arch)
    m.roi_heads.box_predictor = FastRCNNPredictor(m.roi_heads.box_predictor.cls_score.in_features, n_cls)
    return m


def plate_classifier(n=3, pretrained=True):
    """empty / countable / uncountable, from a downscaled photo of the whole plate."""
    m = torchvision.models.resnet18(weights="DEFAULT" if pretrained else None)
    m.fc = nn.Linear(m.fc.in_features, n)
    return m


def save(path, model, **meta):
    torch.save({"model": model.state_dict(), **meta}, path)


def load_detector(path, device="cpu"):
    ck = torch.load(path, map_location="cpu", weights_only=False)
    m = detector(len(ck["classes"]) + 1, ck["arch"], ck["size"], pretrained=False)
    m.load_state_dict(ck["model"])
    return m.to(device).eval(), ck


def load_plate(path, device="cpu"):
    ck = torch.load(path, map_location="cpu", weights_only=False)
    m = plate_classifier(pretrained=False)
    m.load_state_dict(ck["model"])
    return m.to(device).eval(), ck
