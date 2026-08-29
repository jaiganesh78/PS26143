from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


# ============================================================
# DICE LOSS
# ============================================================

class DiceLoss(nn.Module):
    """
    Soft Dice loss for binary segmentation.

    Designed to work safely with AMP / float16 inputs.
    """

    def __init__(self, smooth: float = 1.0):
        super().__init__()
        self.smooth = float(smooth)

    def forward(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
    ) -> torch.Tensor:

        # Perform probability calculation in float32.
        probs = torch.sigmoid(logits.float())
        targets = targets.float()

        probs = probs.contiguous().flatten(1)
        targets = targets.contiguous().flatten(1)

        intersection = (probs * targets).sum(dim=1)

        dice = (
            2.0 * intersection + self.smooth
        ) / (
            probs.sum(dim=1)
            + targets.sum(dim=1)
            + self.smooth
        )

        return 1.0 - dice.mean()


# ============================================================
# FOCAL BCE LOSS
# ============================================================

class FocalBCELoss(nn.Module):
    """
    Binary focal loss.

    Focusing parameter gamma emphasizes difficult pixels.
    Alpha controls positive/negative class weighting.

    Calculated in float32 for numerical stability under AMP.
    """

    def __init__(
        self,
        alpha: float = 0.75,
        gamma: float = 2.0,
    ):
        super().__init__()

        self.alpha = float(alpha)
        self.gamma = float(gamma)

    def forward(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
    ) -> torch.Tensor:

        # Float32 avoids unnecessary AMP precision loss
        # inside the loss calculation.
        logits = logits.float()
        targets = targets.float()

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


# ============================================================
# BOUNDARY LOSS
# ============================================================

class BoundaryLoss(nn.Module):
    """
    Boundary-aware loss.

    Compares Sobel-style local gradients between the
    predicted probability map and the target mask.

    IMPORTANT:
    The Sobel kernels are explicitly converted to the
    input tensor's dtype/device before convolution.

    This makes the loss AMP-safe.
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

    def _edges(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:

        # ----------------------------------------------------
        # CRITICAL AMP FIX
        # ----------------------------------------------------
        #
        # Match the kernels to the input exactly.
        #
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

        # Small epsilon prevents sqrt(0) instability.
        return torch.sqrt(
            gx.pow(2)
            + gy.pow(2)
            + 1e-6
        )

    def forward(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
    ) -> torch.Tensor:

        # Boundary calculation in float32.
        #
        # This is deliberately outside AMP precision because
        # this loss is specifically measuring small local
        # geometric differences.
        probs = torch.sigmoid(
            logits.float()
        )

        targets = targets.float()

        pred_edges = self._edges(
            probs
        )

        true_edges = self._edges(
            targets
        )

        return F.l1_loss(
            pred_edges,
            true_edges,
        )


# ============================================================
# NEGATIVE SCENE LOSS
# ============================================================

class NegativeSceneLoss(nn.Module):
    """
    Explicitly suppresses predictions on completely negative
    scenes such as lookalike and no-oil scenes.

    Only activates for samples whose ground-truth mask
    contains zero positive pixels.
    """

    def __init__(
        self,
        threshold: float = 0.10,
    ):
        super().__init__()

        self.threshold = float(
            threshold
        )

    def forward(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
    ) -> torch.Tensor:

        probs = torch.sigmoid(
            logits.float()
        )

        targets = targets.float()

        target_area = (
            targets
            .flatten(1)
            .sum(dim=1)
        )

        negative = (
            target_area <= 0
        )

        if not negative.any():
            return logits.new_tensor(
                0.0,
                dtype=torch.float32,
            )

        negative_probs = probs[
            negative
        ]

        # Penalize probabilities above
        # the allowed tolerance.
        excess = F.relu(
            negative_probs
            - self.threshold
        )

        return excess.pow(2).mean()


# ============================================================
# V2 COMBINED LOSS
# ============================================================

class V2SegmentationLoss(nn.Module):
    """
    PS26143 V2 segmentation objective.

    Components:

        Focal BCE
            Difficult pixels and false positives.

        Dice
            Region overlap.

        Boundary
            Slick geometry and edge quality.

        Negative Scene
            Explicit suppression of predictions in
            lookalike / no-oil scenes.
    """

    def __init__(
        self,
        focal_weight: float = 0.30,
        dice_weight: float = 0.45,
        boundary_weight: float = 0.15,
        negative_weight: float = 0.10,
        focal_alpha: float = 0.75,
        focal_gamma: float = 2.0,
        negative_threshold: float = 0.10,
    ):
        super().__init__()

        self.focal_weight = float(
            focal_weight
        )

        self.dice_weight = float(
            dice_weight
        )

        self.boundary_weight = float(
            boundary_weight
        )

        self.negative_weight = float(
            negative_weight
        )

        self.focal = FocalBCELoss(
            alpha=focal_alpha,
            gamma=focal_gamma,
        )

        self.dice = DiceLoss()

        self.boundary = BoundaryLoss()

        self.negative = NegativeSceneLoss(
            threshold=negative_threshold,
        )

    def forward(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
    ) -> torch.Tensor:

        focal = self.focal(
            logits,
            targets,
        )

        dice = self.dice(
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
            self.focal_weight * focal
            + self.dice_weight * dice
            + self.boundary_weight * boundary
            + self.negative_weight * negative
        )

        return total

    @torch.no_grad()
    def components(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
    ) -> dict[str, float]:
        """
        Return individual loss components for logging/debugging.
        """

        focal = self.focal(
            logits,
            targets,
        )

        dice = self.dice(
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
            "focal": float(
                focal.item()
            ),
            "dice": float(
                dice.item()
            ),
            "boundary": float(
                boundary.item()
            ),
            "negative": float(
                negative.item()
            ),
        }