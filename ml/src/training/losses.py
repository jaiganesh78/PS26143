import torch
import torch.nn as nn
import torch.nn.functional as F


class DiceLoss(nn.Module):
    """
    Standard soft Dice loss.

    Used for foreground overlap.
    """

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


class TverskyLoss(nn.Module):
    """
    Recall-oriented Tversky loss.

    alpha controls FP penalty.
    beta controls FN penalty.

    beta > alpha means FN are penalized more strongly,
    which directly targets V1's low oil recall.
    """

    def __init__(
        self,
        alpha=0.30,
        beta=0.70,
        smooth=1.0,
    ):
        super().__init__()

        self.alpha = alpha
        self.beta = beta
        self.smooth = smooth

    def forward(self, logits, targets):
        probs = torch.sigmoid(logits)

        probs = probs.contiguous().view(
            probs.size(0), -1
        )
        targets = targets.contiguous().view(
            targets.size(0), -1
        )

        tp = (probs * targets).sum(dim=1)

        fp = (
            probs * (1.0 - targets)
        ).sum(dim=1)

        fn = (
            (1.0 - probs) * targets
        ).sum(dim=1)

        tversky = (
            tp + self.smooth
        ) / (
            tp
            + self.alpha * fp
            + self.beta * fn
            + self.smooth
        )

        return 1.0 - tversky.mean()


class FocalBCELoss(nn.Module):
    """
    Focal BCE.

    Keeps difficult-pixel learning from V2, but with
    balanced alpha so we do not repeat V2's aggressive
    false-positive behaviour.
    """

    def __init__(
        self,
        alpha=0.60,
        gamma=2.0,
    ):
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

        focal_weight = (
            alpha_t
            * (1.0 - p_t).pow(self.gamma)
        )

        return (
            focal_weight * bce
        ).mean()


class BoundaryLoss(nn.Module):
    """
    Boundary structure loss.

    Important:
    kernels are converted to the SAME dtype/device as
    the input. This avoids the AMP Half-vs-Float failure
    encountered during V2 training.
    """

    def __init__(self):
        super().__init__()

        kernel_x = torch.tensor(
            [
                [-1.0, 0.0, 1.0],
                [-2.0, 0.0, 2.0],
                [-1.0, 0.0, 1.0],
            ],
            dtype=torch.float32,
        ).view(1, 1, 3, 3)

        kernel_y = torch.tensor(
            [
                [-1.0, -2.0, -1.0],
                [0.0, 0.0, 0.0],
                [1.0, 2.0, 1.0],
            ],
            dtype=torch.float32,
        ).view(1, 1, 3, 3)

        self.register_buffer(
            "kernel_x",
            kernel_x,
        )

        self.register_buffer(
            "kernel_y",
            kernel_y,
        )

    def _edges(self, x):

        kernel_x = self.kernel_x.to(
            device=x.device,
            dtype=x.dtype,
        )

        kernel_y = self.kernel_y.to(
            device=x.device,
            dtype=x.dtype,
        )

        gx = F.conv2d(
            x,
            kernel_x,
            padding=1,
        )

        gy = F.conv2d(
            x,
            kernel_y,
            padding=1,
        )

        return torch.sqrt(
            gx.pow(2)
            + gy.pow(2)
            + 1e-6
        )

    def forward(self, logits, targets):

        probs = torch.sigmoid(logits)

        pred_edges = self._edges(probs)

        true_edges = self._edges(
            targets.to(dtype=probs.dtype)
        )

        return F.l1_loss(
            pred_edges,
            true_edges,
        )


class NegativeSceneLoss(nn.Module):
    """
    Controlled negative-scene suppression.

    Unlike V2, this is deliberately weak.

    It only penalizes confident predictions in completely
    negative scenes. The threshold is high enough that
    ordinary low-probability background is not aggressively
    suppressed.

    This prevents the loss from recreating V2's FP behaviour.
    """

    def __init__(
        self,
        threshold=0.30,
    ):
        super().__init__()

        self.threshold = threshold

    def forward(self, logits, targets):

        probs = torch.sigmoid(logits)

        target_area = (
            targets
            .flatten(1)
            .sum(dim=1)
        )

        negative = target_area <= 0

        if not negative.any():
            return logits.new_tensor(0.0)

        negative_probs = probs[negative]

        excess = F.relu(
            negative_probs
            - self.threshold
        )

        return excess.pow(2).mean()


class V4SegmentationLoss(nn.Module):
    """
    PS26143 V4 objective.

    Design target:

        V1 problem:
            recall = 0.6867

        V2 problem:
            false-positive rejection deteriorated badly.

    Therefore V4 uses:

        Tversky
            -> explicit FN pressure / recall recovery

        Dice
            -> region overlap stability

        Focal BCE
            -> difficult pixels

        Boundary
            -> geometry

        Weak negative-scene penalty
            -> preserve negative-scene discrimination

    This is intentionally NOT the V2 objective.
    """

    def __init__(
        self,
        tversky_weight=0.40,
        dice_weight=0.25,
        focal_weight=0.20,
        boundary_weight=0.10,
        negative_weight=0.05,
        tversky_alpha=0.30,
        tversky_beta=0.70,
        focal_alpha=0.60,
        focal_gamma=2.0,
        negative_threshold=0.30,
    ):
        super().__init__()

        self.tversky_weight = (
            tversky_weight
        )

        self.dice_weight = (
            dice_weight
        )

        self.focal_weight = (
            focal_weight
        )

        self.boundary_weight = (
            boundary_weight
        )

        self.negative_weight = (
            negative_weight
        )

        self.tversky = TverskyLoss(
            alpha=tversky_alpha,
            beta=tversky_beta,
        )

        self.dice = DiceLoss()

        self.focal = FocalBCELoss(
            alpha=focal_alpha,
            gamma=focal_gamma,
        )

        self.boundary = BoundaryLoss()

        self.negative = NegativeSceneLoss(
            threshold=negative_threshold,
        )

    def forward(self, logits, targets):

        tversky = self.tversky(
            logits,
            targets,
        )

        dice = self.dice(
            logits,
            targets,
        )

        focal = self.focal(
            logits,
            targets,
        )

        boundary = self.boundary(
            logits,
            targets,
        )

        negative = self.negative(
            logits,
            targets,
        )

        total = (
            self.tversky_weight * tversky
            + self.dice_weight * dice
            + self.focal_weight * focal
            + self.boundary_weight * boundary
            + self.negative_weight * negative
        )

        return total

    @torch.no_grad()
    def components(self, logits, targets):

        tversky = self.tversky(
            logits,
            targets,
        )

        dice = self.dice(
            logits,
            targets,
        )

        focal = self.focal(
            logits,
            targets,
        )

        boundary = self.boundary(
            logits,
            targets,
        )

        negative = self.negative(
            logits,
            targets,
        )

        return {
            "tversky": float(
                tversky.item()
            ),
            "dice": float(
                dice.item()
            ),
            "focal": float(
                focal.item()
            ),
            "boundary": float(
                boundary.item()
            ),
            "negative": float(
                negative.item()
            ),
        }


# ------------------------------------------------------------------
# BACKWARD COMPATIBILITY
# ------------------------------------------------------------------

# V1 / existing evaluator compatibility.
class BCEDiceLoss(nn.Module):

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

    def forward(self, logits, targets):

        bce = self.bce(
            logits,
            targets,
        )

        dice = self.dice(
            logits,
            targets,
        )

        return (
            self.bce_weight * bce
            + self.dice_weight * dice
        )


# V2 compatibility.
class V2Loss(V4SegmentationLoss):
    """
    Kept only so older imports do not break.

    The actual V2 experiment should remain reproducible
    from its existing checkpoint and source history.
    """

    pass