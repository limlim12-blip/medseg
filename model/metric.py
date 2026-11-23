import torch


# --------------------------------------------------------------------------- #
# Metrics (take raw logits and {0,1} targets, return a python float)
# --------------------------------------------------------------------------- #
def _binarize(logits, target, threshold=0.5):
    pred = (torch.sigmoid(logits) > threshold).float().flatten(1)
    target = (target > 0.5).float().flatten(1)
    return pred, target


def dice_coeff(output, target, smooth=1e-6):
    pred, target = _binarize(output, target)
    inter = (pred * target).sum(1)
    total = pred.sum(1) + target.sum(1)
    return ((2.0 * inter + smooth) / (total + smooth)).mean().item()


def iou_score(output, target, smooth=1e-6):
    pred, target = _binarize(output, target)
    inter = (pred * target).sum(1)
    union = pred.sum(1) + target.sum(1) - inter
    return ((inter + smooth) / (union + smooth)).mean().item()
