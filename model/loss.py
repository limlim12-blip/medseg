import torch
import torch.nn.functional as F
from torch import nn


# --------------------------------------------------------------------------- #
# Losses
# --------------------------------------------------------------------------- #
class DiceLoss(nn.Module):
    def __init__(self, smooth=1e-6):
        super().__init__()
        self.smooth = smooth

    def forward(self, logits, targets):
        probs = torch.sigmoid(logits).flatten(1)
        targets = targets.flatten(1)
        inter = (probs * targets).sum(1)
        total = probs.sum(1) + targets.sum(1)
        return 1.0 - ((2.0 * inter + self.smooth) / (total + self.smooth)).mean()


class BCEDiceLoss(nn.Module):
    def __init__(self, bce_weight=1.0, dice_weight=1.0, smooth=1e-6):
        super().__init__()
        self.bce_weight = bce_weight
        self.dice_weight = dice_weight
        self.bce = nn.BCEWithLogitsLoss()
        self.dice = DiceLoss(smooth)

    def forward(self, logits, targets):
        return self.bce_weight * self.bce(
            logits, targets
        ) + self.dice_weight * self.dice(logits, targets)


class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, logits, targets):
        bce = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
        probs = torch.sigmoid(logits)
        p_t = probs * targets + (1 - probs) * (1 - targets)
        loss = bce * ((1 - p_t) ** self.gamma)
        if self.alpha >= 0:
            loss = (self.alpha * targets + (1 - self.alpha) * (1 - targets)) * loss
        return loss.mean()


class TverskyLoss(nn.Module):
    def __init__(self, alpha=0.7, beta=0.3, smooth=1e-6):
        super().__init__()
        self.alpha, self.beta, self.smooth = alpha, beta, smooth

    def forward(self, logits, targets):
        probs = torch.sigmoid(logits).flatten(1)
        targets = targets.flatten(1)
        tp = (probs * targets).sum(1)
        fp = (probs * (1 - targets)).sum(1)
        fn = ((1 - probs) * targets).sum(1)
        return (
            1.0
            - (
                (tp + self.smooth)
                / (tp + self.alpha * fp + self.beta * fn + self.smooth)
            ).mean()
        )


def build_loss(name="bce_dice"):
    losses = {
        "bce_dice": BCEDiceLoss,
        "dice": DiceLoss,
        "focal": FocalLoss,
        "tversky": TverskyLoss,
    }
    if name not in losses:
        raise ValueError(f"Unknown loss '{name}'. Choose from {list(losses)}")
    return losses[name]()
