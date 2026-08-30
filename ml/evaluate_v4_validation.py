
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from src.datasets.oil_dataset import OilSegmentationDataset
from src.models.segmentation_model import build_model


# ============================================================
# METRICS
# ============================================================

def safe_divide(a, b):
    if b == 0:
        return 0.0
    return float(a / b)


def segmentation_counts(prediction, target):
    prediction = prediction.astype(bool)
    target = target.astype(bool)

    tp = int(np.logical_and(prediction, target).sum())
    fp = int(np.logical_and(prediction, ~target).sum())
    fn = int(np.logical_and(~prediction, target).sum())
    tn = int(np.logical_and(~prediction, ~target).sum())

    return tp, fp, fn, tn


def metrics_from_counts(tp, fp, fn, tn):
    dice = safe_divide(2 * tp, 2 * tp + fp + fn)
    iou = safe_divide(tp, tp + fp + fn)
    precision = safe_divide(tp, tp + fp)
    recall = safe_divide(tp, tp + fn)

    return {
        "dice": dice,
        "iou": iou,
        "precision": precision,
        "recall": recall,
    }


# ============================================================
# CHECKPOINT
# ============================================================

def load_checkpoint(model, checkpoint_path, device):
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
            state_dict = checkpoint
    else:
        state_dict = checkpoint

    if any(k.startswith("module.") for k in state_dict.keys()):
        state_dict = {
            k.removeprefix("module."): v
            for k, v in state_dict.items()
        }

    model.load_state_dict(state_dict, strict=True)

    return checkpoint


# ============================================================
# SMALL COMPONENT FILTER
# ============================================================

def remove_small_components(mask, min_area):
    """
    Remove connected components smaller than min_area.

    Uses scipy only here. If scipy is unavailable, fail with
    an explicit dependency error rather than silently changing
    the evaluation.
    """
    try:
        from scipy import ndimage
    except ImportError as exc:
        raise RuntimeError(
            "V4 VALIDATION CALIBRATION FAILED: scipy is required "
            "for min-area filtering.\n"
            "Install it with: pip install scipy"
        ) from exc

    if min_area <= 1:
        return mask.astype(bool)

    structure = np.ones((3, 3), dtype=np.uint8)

    labeled, num = ndimage.label(
        mask.astype(np.uint8),
        structure=structure,
    )

    if num == 0:
        return np.zeros_like(mask, dtype=bool)

    sizes = np.bincount(labeled.ravel())

    keep = sizes >= min_area
    keep[0] = False

    return keep[labeled]


# ============================================================
# MAIN
# ============================================================

def main():

    parser = argparse.ArgumentParser(
        description="PS26143 V4 validation-only threshold calibration"
    )

    parser.add_argument(
        "--checkpoint",
        type=Path,
        required=True,
    )

    parser.add_argument(
        "--val-manifest",
        type=Path,
        required=True,
    )

    parser.add_argument(
        "--output",
        type=Path,
        required=True,
    )

    parser.add_argument(
        "--thresholds",
        type=float,
        nargs="+",
        default=[
            0.40,
            0.45,
            0.50,
            0.55,
            0.60,
            0.65,
        ],
    )

    parser.add_argument(
        "--min-areas",
        type=int,
        nargs="+",
        default=[
            0,
            25,
            50,
            100,
        ],
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=16,
    )

    args = parser.parse_args()

    print("=" * 70)
    print("PS26143 — V4 VALIDATION CALIBRATION")
    print("=" * 70)

    print()
    print("Checkpoint :", args.checkpoint)
    print("Validation :", args.val_manifest)
    print("Output     :", args.output)

    # --------------------------------------------------------
    # DEVICE
    # --------------------------------------------------------

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    print()
    print("Device:", device)

    if device.type == "cuda":
        print(
            "GPU:",
            torch.cuda.get_device_name(0),
        )

    # --------------------------------------------------------
    # INPUT CHECKS
    # --------------------------------------------------------

    if not args.checkpoint.exists():
        raise FileNotFoundError(
            f"V4 checkpoint not found: {args.checkpoint}"
        )

    if not args.val_manifest.exists():
        raise FileNotFoundError(
            f"Validation manifest not found: {args.val_manifest}"
        )

    # --------------------------------------------------------
    # MANIFEST
    # --------------------------------------------------------

    df = pd.read_csv(args.val_manifest)

    if len(df) != 135:
        raise RuntimeError(
            "VALIDATION SET INTEGRITY FAILURE: "
            f"expected exactly 135 validation scenes, found {len(df)}"
        )

    required_columns = {
        "global_id",
        "dataset",
        "image",
        "mask",
    }

    missing = required_columns - set(df.columns)

    if missing:
        raise RuntimeError(
            "VALIDATION MANIFEST MISSING REQUIRED COLUMNS: "
            f"{sorted(missing)}"
        )

    print()
    print("Validation samples:", len(df))

    print()
    print("Dataset distribution:")
    print(df["dataset"].value_counts())

    # --------------------------------------------------------
    # PATH CHECK
    # --------------------------------------------------------

    for _, row in df.iterrows():

        image_path = Path(row["image"])
        mask_path = Path(row["mask"])

        if not image_path.exists():
            raise FileNotFoundError(
                f"Validation image missing: {image_path}"
            )

        if not mask_path.exists():
            raise FileNotFoundError(
                f"Validation mask missing: {mask_path}"
            )

    # --------------------------------------------------------
    # MODEL
    # --------------------------------------------------------

    print()
    print("=" * 70)
    print("BUILDING V4 MODEL")
    print("=" * 70)

    model = build_model(
        architecture="unet",
        encoder="resnet34",
        encoder_weights=None,
        in_channels=2,
        classes=1,
    )

    model = model.to(device)

    print("Architecture : U-Net")
    print("Encoder      : ResNet34")
    print("Input        : VV + VH")
    print("Resolution   : 512 x 512")
    print("Checkpoint   :", args.checkpoint)

    checkpoint = load_checkpoint(
        model,
        args.checkpoint,
        device,
    )

    if isinstance(checkpoint, dict):

        if "epoch" in checkpoint:
            print("Checkpoint epoch:", checkpoint["epoch"])

        if "best_val_dice" in checkpoint:
            print(
                "Checkpoint best validation Dice:",
                checkpoint["best_val_dice"],
            )

    # --------------------------------------------------------
    # DATASET
    # --------------------------------------------------------

    dataset = OilSegmentationDataset(
        df.to_dict("records"),
        augment=False,
    )

    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=(device.type == "cuda"),
    )

    # --------------------------------------------------------
    # COLLECT RAW PROBABILITY MAPS
    # --------------------------------------------------------

    model.eval()

    samples = []

    print()
    print("=" * 70)
    print("GENERATING VALIDATION PROBABILITIES")
    print("=" * 70)

    with torch.no_grad():

        processed = 0

        for batch in loader:

            images = batch["image"].to(
                device,
                non_blocking=True,
            )

            masks = batch["mask"].to(
                device,
                non_blocking=True,
            )

            if device.type == "cuda":
                with torch.autocast(
                    device_type="cuda",
                    dtype=torch.float16,
                ):
                    logits = model(images)
            else:
                logits = model(images)

            probabilities = torch.sigmoid(
                logits.float()
            )

            probabilities_np = (
                probabilities
                .cpu()
                .numpy()
            )

            masks_np = (
                masks
                .cpu()
                .numpy()
            )

            for i in range(len(images)):

                samples.append(
                    {
                        "global_id": str(
                            batch["global_id"][i]
                        ),
                        "dataset": str(
                            batch["dataset"][i]
                        ),
                        "probability": probabilities_np[
                            i, 0
                        ],
                        "target": (
                            masks_np[i, 0] > 0.5
                        ),
                    }
                )

                processed += 1

        print(
            f"Processed: {processed}/{len(df)}"
        )

    if len(samples) != 135:
        raise RuntimeError(
            "VALIDATION PROCESSING FAILURE: "
            f"expected 135 samples, got {len(samples)}"
        )

    # --------------------------------------------------------
    # SWEEP
    # --------------------------------------------------------

    results = []

    print()
    print("=" * 70)
    print("RUNNING THRESHOLD × MIN-AREA SWEEP")
    print("=" * 70)

    for threshold in args.thresholds:

        for min_area in args.min_areas:

            total_tp = 0
            total_fp = 0
            total_fn = 0
            total_tn = 0

            oil_tp = 0
            oil_fp = 0
            oil_fn = 0
            oil_tn = 0

            oil_detected = 0
            oil_total = 0

            lookalike_false_alarms = 0
            lookalike_total = 0

            no_oil_false_alarms = 0
            no_oil_total = 0

            predicted_positive_pixels = 0
            false_positive_pixels = 0

            for sample in samples:

                raw_prediction = (
                    sample["probability"]
                    >= threshold
                )

                prediction = remove_small_components(
                    raw_prediction,
                    min_area,
                )

                target = sample["target"]

                tp, fp, fn, tn = segmentation_counts(
                    prediction,
                    target,
                )

                total_tp += tp
                total_fp += fp
                total_fn += fn
                total_tn += tn

                predicted_positive_pixels += int(
                    prediction.sum()
                )

                false_positive_pixels += fp

                dataset_name = sample["dataset"]

                scene_positive = bool(
                    prediction.any()
                )

                if dataset_name == "oil":

                    oil_total += 1

                    if scene_positive:
                        oil_detected += 1

                    oil_tp += tp
                    oil_fp += fp
                    oil_fn += fn
                    oil_tn += tn

                elif dataset_name == "lookalike":

                    lookalike_total += 1

                    if scene_positive:
                        lookalike_false_alarms += 1

                elif dataset_name == "no_oil":

                    no_oil_total += 1

                    if scene_positive:
                        no_oil_false_alarms += 1

            overall = metrics_from_counts(
                total_tp,
                total_fp,
                total_fn,
                total_tn,
            )

            oil_metrics = metrics_from_counts(
                oil_tp,
                oil_fp,
                oil_fn,
                oil_tn,
            )

            oil_detection_rate = safe_divide(
                oil_detected,
                oil_total,
            )

            lookalike_far = safe_divide(
                lookalike_false_alarms,
                lookalike_total,
            )

            no_oil_far = safe_divide(
                no_oil_false_alarms,
                no_oil_total,
            )

            # ------------------------------------------------
            # DECISION SCORE
            #
            # Primary objective:
            #   maximize oil segmentation quality
            #
            # Strong penalty for lookalike/no-oil alarms.
            # ------------------------------------------------

            score = (
                oil_metrics["dice"]
                - 0.50 * lookalike_far
                - 0.50 * no_oil_far
            )

            results.append(
                {
                    "threshold": float(threshold),
                    "min_area": int(min_area),

                    "score": float(score),

                    "oil_detection_rate":
                        oil_detection_rate,

                    "lookalike_false_alarm_rate":
                        lookalike_far,

                    "no_oil_false_alarm_rate":
                        no_oil_far,

                    "oil_dice":
                        oil_metrics["dice"],

                    "oil_iou":
                        oil_metrics["iou"],

                    "oil_precision":
                        oil_metrics["precision"],

                    "oil_recall":
                        oil_metrics["recall"],

                    "overall_dice":
                        overall["dice"],

                    "overall_iou":
                        overall["iou"],

                    "overall_precision":
                        overall["precision"],

                    "overall_recall":
                        overall["recall"],

                    "tp": total_tp,
                    "fp": total_fp,
                    "fn": total_fn,
                    "tn": total_tn,

                    "predicted_positive_pixels":
                        predicted_positive_pixels,

                    "false_positive_pixels":
                        false_positive_pixels,
                }
            )

    results_df = pd.DataFrame(results)

    # --------------------------------------------------------
    # SORT
    # --------------------------------------------------------

    results_df = results_df.sort_values(
        by=[
            "score",
            "oil_dice",
            "oil_iou",
        ],
        ascending=False,
    ).reset_index(drop=True)

    # --------------------------------------------------------
    # OUTPUT
    # --------------------------------------------------------

    args.output.mkdir(
        parents=True,
        exist_ok=True,
    )

    sweep_path = (
        args.output
        / "v4_validation_calibration_sweep.csv"
    )

    selected_path = (
        args.output
        / "v4_selected_config.json"
    )

    results_df.to_csv(
        sweep_path,
        index=False,
    )

    best = results_df.iloc[0].to_dict()

    selected = {
        "experiment": "oil-seg-v4",
        "source": "validation_only",

        "checkpoint": str(
            args.checkpoint
        ),

        "validation_manifest": str(
            args.val_manifest
        ),

        "threshold": float(
            best["threshold"]
        ),

        "min_area": int(
            best["min_area"]
        ),

        "score": float(
            best["score"]
        ),

        "validation_metrics": {
            "oil_detection_rate":
                float(
                    best[
                        "oil_detection_rate"
                    ]
                ),

            "lookalike_false_alarm_rate":
                float(
                    best[
                        "lookalike_false_alarm_rate"
                    ]
                ),

            "no_oil_false_alarm_rate":
                float(
                    best[
                        "no_oil_false_alarm_rate"
                    ]
                ),

            "oil_dice":
                float(
                    best["oil_dice"]
                ),

            "oil_iou":
                float(
                    best["oil_iou"]
                ),

            "oil_precision":
                float(
                    best["oil_precision"]
                ),

            "oil_recall":
                float(
                    best["oil_recall"]
                ),
        },

        "safety": {
            "training_used": False,
            "test_used": False,
            "validation_samples": 135,
        },
    }

    with selected_path.open(
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            selected,
            f,
            indent=2,
        )

    # --------------------------------------------------------
    # PRINT
    # --------------------------------------------------------

    print()
    print("=" * 70)
    print("V4 VALIDATION CALIBRATION RESULTS")
    print("=" * 70)

    print()
    print("SELECTED CONFIGURATION")

    print(
        json.dumps(
            selected,
            indent=2,
        )
    )

    print()
    print("=" * 70)
    print("TOP 10 CONFIGURATIONS")
    print("=" * 70)

    print(
        results_df[
            [
                "score",
                "threshold",
                "min_area",
                "oil_detection_rate",
                "lookalike_false_alarm_rate",
                "no_oil_false_alarm_rate",
                "oil_dice",
                "oil_iou",
                "oil_precision",
                "oil_recall",
            ]
        ]
        .head(10)
        .to_string(index=False)
    )

    print()
    print("=" * 70)
    print("ARTIFACTS")
    print("=" * 70)

    print("Selected config:")
    print(selected_path)

    print()
    print("Sweep CSV:")
    print(sweep_path)

    print()
    print("=" * 70)
    print("✅ V4 VALIDATION CALIBRATION COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()

