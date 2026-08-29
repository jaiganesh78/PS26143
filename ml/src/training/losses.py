import torch
import torch.nn as nn


class DiceLoss(nn.Module):
    """
    Soft Dice loss for binary segmentation.
    """

    def __init__(self, smooth=1.0):
        super().__init__()

        self.smooth = smooth

    def forward(
        self,
        logits,
        targets
    ):
        probs = torch.sigmoid(
            logits
        )

        probs = probs.contiguous().view(
            probs.size(0),
            -1
        )

        targets = targets.contiguous().view(
            targets.size(0),
            -1
        )

        intersection = (
            probs * targets
        ).sum(dim=1)

        dice = (
            2.0 * intersection
            + self.smooth
        ) / (
            probs.sum(dim=1)
            + targets.sum(dim=1)
            + self.smooth
        )

        return 1.0 - dice.mean()


class BCEDiceLoss(nn.Module):
    """
    Combined BCE + Dice loss.
    """

    def __init__(
        self,
        bce_weight=0.5,
        dice_weight=0.5,
    ):
        super().__init__()

        self.bce_weight = bce_weight
        self.dice_weight = dice_weight

        self.bce = nn.BCEWithLogitsLoss()

        self.dice = DiceLoss()

    def forward(
        self,
        logits,
        targets
    ):
        bce = self.bce(
            logits,
            targets
        )

        dice = self.dice(
            logits,
            targets
        )

        return (
            self.bce_weight * bce
            + self.dice_weight * dice
        )