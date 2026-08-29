from __future__ import annotations

import csv
import json
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from src.datasets.oil_dataset import OilSegmentationDataset
from src.models.segmentation_model import build_model
from src.training.losses import BCEDiceLoss


# ============================================================
# PS26143 — FINAL TEST EVALUATION
# ============================================================

SEED = 42

DRIVE_ROOT = Path("/content/drive/MyDrive/PS26143")

PROCESSED_ROOT = DRIVE_ROOT / "data/processed"

TEST_MANIFEST = (
    PROCESSED_ROOT / "test/manifest.csv"
)

CHECKPOINT = (
    DRIVE_ROOT / "checkpoints/oil_seg_v1_best.pt"
)

OUTPUT_DIR = (
    DRIVE_ROOT / "evaluation/oil_seg_v1"
)

RESULTS_CSV = OUTPUT_DIR / "test_predictions.csv"
SUMMARY_JSON = OUTPUT_DIR / "test_summary.json"

BATCH_SIZE = 16
NUM_WORKERS = 0

THRESHOLD = 0.5


# ============================================================
# REPRODUCIBILITY
# ============================================================

def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ============================================================
# METRICS
# ============================================================

def binary_metrics(
    prediction: torch.Tensor,
    target: torch.Tensor,
    eps: float = 1e-7,
):
    """
    Pixel-level binary segmentation metrics.

    prediction: [N,1,H,W] boolean
    target:     [N,1,H,W] boolean
    """

    prediction = prediction.bool()
    target = target.bool()

    tp = (prediction & target).sum().item()
    fp = (prediction & ~target).sum().item()
    fn = (~prediction & target).sum().item()
    tn = (~prediction & ~target).sum().item()

    intersection = tp
    union = tp + fp + fn

    dice = (
        2.0 * tp /
        (2.0 * tp + fp + fn + eps)
    )

    iou = (
        tp /
        (union + eps)
    )

    precision = (
        tp /
        (tp + fp + eps)
    )

    recall = (
        tp /
        (tp + fn + eps)
    )

    f1 = (
        2.0 * precision * recall /
        (precision + recall + eps)
    )

    accuracy = (
        (tp + tn) /
        (tp + tn + fp + fn + eps)
    )

    return {
        "tp": int(tp),
        "fp": int(fp),
        "fn": int(fn),
        "tn": int(tn),
        "dice": float(dice),
        "iou": float(iou),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "accuracy": float(accuracy),
        "target_positive_pixels": int(target.sum().item()),
        "predicted_positive_pixels": int(prediction.sum().item()),
    }


def aggregate_counts(records):
    """
    Aggregate TP/FP/FN/TN and derive metrics.
    """

    tp = sum(r["tp"] for r in records)
    fp = sum(r["fp"] for r in records)
    fn = sum(r["fn"] for r in records)
    tn = sum(r["tn"] for r in records)

    eps = 1e-7

    dice = (
        2.0 * tp /
        (2.0 * tp + fp + fn + eps)
    )

    iou = (
        tp /
        (tp + fp + fn + eps)
    )

    precision = (
        tp /
        (tp + fp + eps)
    )

    recall = (
        tp /
        (tp + fn + eps)
    )

    f1 = (
        2.0 * precision * recall /
        (precision + recall + eps)
    )

    accuracy = (
        (tp + tn) /
        (tp + tn + fp + fn + eps)
    )

    target_positive = sum(
        r["target_positive_pixels"]
        for r in records
    )

    predicted_positive = sum(
        r["predicted_positive_pixels"]
        for r in records
    )

    return {
        "samples": len(records),
        "tp": int(tp),
        "fp": int(fp),
        "fn": int(fn),
        "tn": int(tn),
        "dice": float(dice),
        "iou": float(iou),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "accuracy": float(accuracy),
        "target_positive_pixels": int(target_positive),
        "predicted_positive_pixels": int(predicted_positive),
    }


# ============================================================
# CHECKPOINT LOADING
# ============================================================

def load_checkpoint(model, checkpoint_path, device):
    """
    Supports checkpoints saved either as:

        {
            "model_state_dict": ...
        }

    or a raw state_dict.
    """

    checkpoint = torch.load(
        checkpoint_path,
        map_location=device,
        weights_only=False,
    )

    if isinstance(checkpoint, dict):

        if "model_state_dict" in checkpoint:
            state_dict = checkpoint["model_state_dict"]

        elif "state_dict" in checkpoint:
            state_dict = checkpoint["state_dict"]

        else:
            # Raw state_dict case.
            state_dict = checkpoint

    else:
        raise RuntimeError(
            "Unsupported checkpoint format."
        )

    # Handle DataParallel checkpoints.
    cleaned = {}

    for key, value in state_dict.items():

        if key.startswith("module."):
            key = key[len("module."):]

        cleaned[key] = value

    model.load_state_dict(
        cleaned,
        strict=True,
    )

    return checkpoint


# ============================================================
# MAIN EVALUATION
# ============================================================

def main():

    set_seed(SEED)

    print("=" * 70)
    print("PS26143 — FINAL TEST SET EVALUATION")
    print("=" * 70)

    # --------------------------------------------------------
    # Device
    # --------------------------------------------------------

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print()
    print("Device:", device)

    if device.type == "cuda":
        print(
            "GPU   :",
            torch.cuda.get_device_name(0),
        )

        print(
            "VRAM  :",
            round(
                torch.cuda.get_device_properties(0).total_memory
                / (1024 ** 3),
                2,
            ),
            "GiB",
        )

    # --------------------------------------------------------
    # Paths
    # --------------------------------------------------------

    print()
    print("=" * 70)
    print("INPUTS")
    print("=" * 70)

    print("Test manifest :", TEST_MANIFEST)
    print("Checkpoint    :", CHECKPOINT)
    print("Output        :", OUTPUT_DIR)

    if not TEST_MANIFEST.exists():
        raise FileNotFoundError(
            f"Missing test manifest: {TEST_MANIFEST}"
        )

    if not CHECKPOINT.exists():
        raise FileNotFoundError(
            f"Missing best checkpoint: {CHECKPOINT}"
        )

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    # --------------------------------------------------------
    # Test manifest
    # --------------------------------------------------------

    test_df = pd.read_csv(TEST_MANIFEST)

    expected = 135

    if len(test_df) != expected:
        raise RuntimeError(
            f"Expected {expected} test samples, "
            f"found {len(test_df)}."
        )

    if test_df["global_id"].duplicated().any():
        raise RuntimeError(
            "Duplicate global IDs found in test manifest."
        )

    print()
    print("Test samples:", len(test_df))

    print()
    print("Test distribution:")
    print(
        test_df["dataset"]
        .value_counts()
        .sort_index()
    )

    # --------------------------------------------------------
    # Verify referenced files
    # --------------------------------------------------------

    missing_images = [
        p for p in test_df["image"]
        if not Path(p).exists()
    ]

    missing_masks = [
        p for p in test_df["mask"]
        if not Path(p).exists()
    ]

    if missing_images:
        raise FileNotFoundError(
            f"Missing test images: "
            f"{missing_images[:5]}"
        )

    if missing_masks:
        raise FileNotFoundError(
            f"Missing test masks: "
            f"{missing_masks[:5]}"
        )

    print()
    print("Test image paths: VERIFIED")
    print("Test mask paths : VERIFIED")

    # --------------------------------------------------------
    # Dataset
    # --------------------------------------------------------

    records = test_df.to_dict(
        orient="records"
    )

    dataset = OilSegmentationDataset(
        records,
        augment=False,
    )

    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=(device.type == "cuda"),
    )

    # --------------------------------------------------------
    # Model
    # --------------------------------------------------------

    print()
    print("=" * 70)
    print("BUILDING MODEL")
    print("=" * 70)

    model = build_model(
        architecture="unet",
        encoder="resnet34",
        encoder_weights=None,
        in_channels=2,
        classes=1,
    )

    model.to(device)

    parameter_count = sum(
        p.numel()
        for p in model.parameters()
    )

    print("Architecture : U-Net")
    print("Encoder      : ResNet34")
    print("Input        : 2 channels")
    print("Output       : 1 channel")
    print(
        "Parameters   :",
        f"{parameter_count:,}",
    )

    # --------------------------------------------------------
    # Load BEST checkpoint
    # --------------------------------------------------------

    print()
    print("=" * 70)
    print("LOADING BEST CHECKPOINT")
    print("=" * 70)

    checkpoint = load_checkpoint(
        model,
        CHECKPOINT,
        device,
    )

    print("Checkpoint:", CHECKPOINT)
    print("Checkpoint loaded successfully.")

    if isinstance(checkpoint, dict):

        if "epoch" in checkpoint:
            print(
                "Checkpoint epoch:",
                checkpoint["epoch"],
            )

        if "best_val_dice" in checkpoint:
            print(
                "Best validation Dice:",
                checkpoint["best_val_dice"],
            )

    # --------------------------------------------------------
    # Evaluation
    # --------------------------------------------------------

    print()
    print("=" * 70)
    print("EVALUATING 135 HELD-OUT TEST SAMPLES")
    print("=" * 70)

    model.eval()

    scaler_enabled = (
        device.type == "cuda"
    )

    per_sample = []

    with torch.no_grad():

        for batch_index, batch in enumerate(
            loader,
            start=1,
        ):

            images = batch["image"].to(
                device,
                non_blocking=True,
            )

            masks = batch["mask"].to(
                device,
                non_blocking=True,
            )

            if scaler_enabled:

                with torch.autocast(
                    device_type="cuda",
                    dtype=torch.float16,
                ):
                    logits = model(images)

            else:
                logits = model(images)

            probabilities = torch.sigmoid(
                logits
            )

            predictions = (
                probabilities >= THRESHOLD
            )

            for i in range(images.size(0)):

                metric = binary_metrics(
                    predictions[i:i + 1],
                    masks[i:i + 1],
                )

                global_id = batch[
                    "global_id"
                ][i]

                dataset_name = batch[
                    "dataset"
                ][i]

                record = {
                    "global_id": global_id,
                    "dataset": dataset_name,
                    "dice": metric["dice"],
                    "iou": metric["iou"],
                    "precision": metric["precision"],
                    "recall": metric["recall"],
                    "f1": metric["f1"],
                    "accuracy": metric["accuracy"],
                    "tp": metric["tp"],
                    "fp": metric["fp"],
                    "fn": metric["fn"],
                    "tn": metric["tn"],
                    "target_positive_pixels":
                        metric[
                            "target_positive_pixels"
                        ],
                    "predicted_positive_pixels":
                        metric[
                            "predicted_positive_pixels"
                        ],
                }

                per_sample.append(record)

            completed = min(
                batch_index * BATCH_SIZE,
                len(dataset),
            )

            print(
                f"[{completed:3d}/{len(dataset)}] "
                f"{100.0 * completed / len(dataset):6.2f}%"
            )

    # --------------------------------------------------------
    # Save per-sample results
    # --------------------------------------------------------

    results_df = pd.DataFrame(
        per_sample
    )

    results_df.to_csv(
        RESULTS_CSV,
        index=False,
    )

    # --------------------------------------------------------
    # Aggregate metrics
    # --------------------------------------------------------

    overall = aggregate_counts(
        per_sample
    )

    by_dataset = {}

    for dataset_name in [
        "oil",
        "lookalike",
        "no_oil",
    ]:

        subset = [
            r for r in per_sample
            if r["dataset"] == dataset_name
        ]

        by_dataset[dataset_name] = (
            aggregate_counts(subset)
        )

    # Macro average of per-image metrics.
    macro = {
        "dice": float(
            results_df["dice"].mean()
        ),
        "iou": float(
            results_df["iou"].mean()
        ),
        "precision": float(
            results_df["precision"].mean()
        ),
        "recall": float(
            results_df["recall"].mean()
        ),
        "f1": float(
            results_df["f1"].mean()
        ),
        "accuracy": float(
            results_df["accuracy"].mean()
        ),
    }

    # --------------------------------------------------------
    # Oil-scene detection summary
    # --------------------------------------------------------

    oil_df = results_df[
        results_df["dataset"] == "oil"
    ]

    lookalike_df = results_df[
        results_df["dataset"] == "lookalike"
    ]

    no_oil_df = results_df[
        results_df["dataset"] == "no_oil"
    ]

    oil_target_positive = (
        oil_df["target_positive_pixels"] > 0
    )

    oil_prediction_positive = (
        oil_df["predicted_positive_pixels"] > 0
    )

    oil_scene_detection = {
        "oil_samples": int(len(oil_df)),
        "oil_samples_predicted_positive":
            int(oil_prediction_positive.sum()),
        "oil_samples_predicted_empty":
            int((~oil_prediction_positive).sum()),
    }

    # For negative classes, any predicted positive pixel
    # is a scene-level false alarm.
    lookalike_false_alarm_rate = float(
        (
            lookalike_df[
                "predicted_positive_pixels"
            ] > 0
        ).mean()
    )

    no_oil_false_alarm_rate = float(
        (
            no_oil_df[
                "predicted_positive_pixels"
            ] > 0
        ).mean()
    )

    scene_summary = {
        "oil_detection_rate": float(
            oil_prediction_positive.mean()
        ),
        "lookalike_false_alarm_rate":
            lookalike_false_alarm_rate,
        "no_oil_false_alarm_rate":
            no_oil_false_alarm_rate,
    }

    # --------------------------------------------------------
    # Summary
    # --------------------------------------------------------

    summary = {
        "experiment": "oil-seg-v1",
        "checkpoint": str(CHECKPOINT),
        "test_manifest": str(TEST_MANIFEST),
        "test_samples": len(test_df),
        "threshold": THRESHOLD,
        "device": str(device),
        "gpu": (
            torch.cuda.get_device_name(0)
            if device.type == "cuda"
            else None
        ),
        "architecture": "U-Net",
        "encoder": "ResNet34",
        "in_channels": 2,
        "classes": 1,
        "image_size": 512,
        "overall_micro": overall,
        "overall_macro": macro,
        "by_dataset": by_dataset,
        "scene_level": scene_summary,
        "oil_scene_detection": oil_scene_detection,
    }

    with open(
        SUMMARY_JSON,
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            summary,
            f,
            indent=2,
        )

    # --------------------------------------------------------
    # Final report
    # --------------------------------------------------------

    print()
    print("=" * 70)
    print("FINAL TEST RESULTS")
    print("=" * 70)

    print()
    print("MICRO-AGGREGATED PIXEL METRICS")
    print(
        f"Dice      : {overall['dice']:.4f}"
    )
    print(
        f"IoU       : {overall['iou']:.4f}"
    )
    print(
        f"Precision : {overall['precision']:.4f}"
    )
    print(
        f"Recall    : {overall['recall']:.4f}"
    )
    print(
        f"F1        : {overall['f1']:.4f}"
    )
    print(
        f"Accuracy  : {overall['accuracy']:.4f}"
    )

    print()
    print("MACRO-AVERAGED PER-SAMPLE METRICS")
    print(
        f"Dice      : {macro['dice']:.4f}"
    )
    print(
        f"IoU       : {macro['iou']:.4f}"
    )
    print(
        f"Precision : {macro['precision']:.4f}"
    )
    print(
        f"Recall    : {macro['recall']:.4f}"
    )
    print(
        f"F1        : {macro['f1']:.4f}"
    )

    print()
    print("BY DATASET")

    for name, metrics in by_dataset.items():

        print()
        print(
            f"{name.upper():12s} "
            f"({metrics['samples']} samples)"
        )

        print(
            f"  Dice      : {metrics['dice']:.4f}"
        )
        print(
            f"  IoU       : {metrics['iou']:.4f}"
        )
        print(
            f"  Precision : {metrics['precision']:.4f}"
        )
        print(
            f"  Recall    : {metrics['recall']:.4f}"
        )

    print()
    print("SCENE-LEVEL SIGNAL")

    print(
        "Oil detection rate       : "
        f"{scene_summary['oil_detection_rate']:.4f}"
    )

    print(
        "Lookalike false alarm    : "
        f"{scene_summary['lookalike_false_alarm_rate']:.4f}"
    )

    print(
        "No-oil false alarm       : "
        f"{scene_summary['no_oil_false_alarm_rate']:.4f}"
    )

    print()
    print("=" * 70)
    print("EVALUATION COMPLETE")
    print("=" * 70)

    print()
    print("Per-sample CSV:")
    print(RESULTS_CSV)

    print()
    print("Summary JSON:")
    print(SUMMARY_JSON)

    print()
    print(
        "IMPORTANT: "
        "The test set was evaluated exactly once "
        "with the frozen best checkpoint."
    )


if __name__ == "__main__":
    main()