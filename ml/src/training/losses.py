import torch
import torch.nn as nn
import torch.nn.functional as F


class DiceLoss(nn.Module):
    def __init__(self, smooth=1.0):
        super().__init__()
        self.smooth = smooth

    def forward(self, logits, targets):
        probs = torch.sigmoid(logits)

        probs = probs.contiguous().view(probs.size(0), -1)
        targets = targets.contiguous().view(targets.size(0), -1)

        intersection = (probs * targets).sum(dim=1)

        dice = (
            2.0 * intersection + self.smooth
        ) / (
            probs.sum(dim=1)
            + targets.sum(dim=1)
            + self.smooth
        )

        return 1.0 - dice.mean()


class FocalBCELoss(nn.Module):
    """
    Focal BCE.

    Gives additional attention to difficult pixels and,
    importantly for this problem, confident false-positive
    pixels in negative scenes.
    """

    def __init__(self, alpha=0.75, gamma=2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, logits, targets):
        probs = torch.sigmoid(logits)

        bce = F.binary_cross_entropy_with_logits(
            logits,
            targets,
            reduction="none",
        )

        p_t = (
            probs * targets
            + (1.0 - probs) * (1.0 - targets)
        )

        alpha_t = (
            self.alpha * targets
            + (1.0 - self.alpha) * (1.0 - targets)
        )

        focal_weight = alpha_t * (1.0 - p_t).pow(self.gamma)

        return (focal_weight * bce).mean()


class BoundaryLoss(nn.Module):
    """
    Boundary-aware loss.

    Compares local gradient/boundary structure between the
    predicted probability map and the target mask.

    This encourages cleaner slick edges rather than only
    maximizing bulk pixel overlap.
    """

    def __init__(self):
        super().__init__()

        kernel_x = torch.tensor(
            [
                [-1.0, 0.0, 1.0],
                [-2.0, 0.0, 2.0],
                [-1.0, 0.0, 1.0],
            ]
        ).view(1, 1, 3, 3)

        kernel_y = torch.tensor(
            [
                [-1.0, -2.0, -1.0],
                [0.0, 0.0, 0.0],
                [1.0, 2.0, 1.0],
            ]
        ).view(1, 1, 3, 3)

        self.register_buffer("kernel_x", kernel_x)
        self.register_buffer("kernel_y", kernel_y)

    def _edges(self, x):
        gx = F.conv2d(
            x,
            self.kernel_x,
            padding=1,
        )

        gy = F.conv2d(
            x,
            self.kernel_y,
            padding=1,
        )

        return torch.sqrt(
            gx.pow(2) + gy.pow(2) + 1e-6
        )

    def forward(self, logits, targets):
        probs = torch.sigmoid(logits)

        pred_edges = self._edges(probs)
        true_edges = self._edges(targets)

        return F.l1_loss(
            pred_edges,
            true_edges,
        )


class NegativeSceneLoss(nn.Module):
    """
    Explicitly suppresses predictions on completely negative
    scenes (lookalike and no-oil).

    The loss is activated only when the target scene contains
    no positive pixels.

    This prevents the model from becoming over-aggressive
    on negative scenes while preserving normal segmentation
    learning on oil scenes.
    """

    def __init__(self, threshold=0.10):
        super().__init__()
        self.threshold = threshold

    def forward(self, logits, targets):
        probs = torch.sigmoid(logits)

        target_area = targets.flatten(1).sum(dim=1)

        negative = target_area <= 0

        if not negative.any():
            return logits.new_tensor(0.0)

        negative_probs = probs[negative]

        # Penalize probabilities above a small tolerance.
        excess = F.relu(
            negative_probs - self.threshold
        )

        return excess.pow(2).mean()


class V2SegmentationLoss(nn.Module):
    """
    PS26143 V2 objective.

    Components:
        Focal BCE       -> difficult pixels / false positives
        Dice            -> region overlap
        Boundary        -> slick geometry
        Negative scene  -> explicit lookalike/no-oil rejection
    """

    def __init__(
        self,
        focal_weight=0.30,
        dice_weight=0.45,
        boundary_weight=0.15,
        negative_weight=0.10,
        focal_alpha=0.75,
        focal_gamma=2.0,
        negative_threshold=0.10,
    ):
        super().__init__()

        self.focal_weight = focal_weight
        self.dice_weight = dice_weight
        self.boundary_weight = boundary_weight
        self.negative_weight = negative_weight

        self.focal = FocalBCELoss(
            alpha=focal_alpha,
            gamma=focal_gamma,
        )

        self.dice = DiceLoss()

        self.boundary = BoundaryLoss()

        self.negative = NegativeSceneLoss(
            threshold=negative_threshold,
        )

    def forward(self, logits, targets):
        focal = self.focal(logits, targets)
        dice = self.dice(logits, targets)
        boundary = self.boundary(logits, targets)
        negative = self.negative(logits, targets)

        total = (
            self.focal_weight * focal
            + self.dice_weight * dice
            + self.boundary_weight * boundary
            + self.negative_weight * negative
        )

        return total

    def components(self, logits, targets):
        """
        Useful for training logs/debugging.
        """
        with torch.no_grad():
            focal = self.focal(logits, targets)
            dice = self.dice(logits, targets)
            boundary = self.boundary(logits, targets)
            negative = self.negative(logits, targets)

        return {
            "focal": float(focal.item()),
            "dice": float(dice.item()),
            "boundary": float(boundary.item()),
            "negative": float(negative.item()),
        }