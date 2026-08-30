
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
# HELPERS
# ============================================================

def safe_divide(a, b):
    if b == 0:
        return 0.0
    return float(a / b)


def counts(pred, target):
    pred = pred.astype(bool)
    target = target.astype(bool)

    tp = int(np.logical_and(pred, target).sum())
    fp = int(np.logical_and(pred, ~target).sum())
    fn = int(np.logical_and(~pred, target).sum())
    tn = int(np.logical_and(~pred, ~target).sum())

    return tp, fp, fn, tn


def metrics(tp, fp, fn, tn):
    return {
        "dice": safe_divide(
            2 * tp,
            2 * tp + fp + fn,
        ),
        "iou": safe_divide(
            tp,
            tp + fp + fn,
        ),
        "precision": safe_divide(
            tp,
            tp + fp,
        ),
        "recall": safe_divide(
            tp,
            tp + fn,
        ),
        "accuracy": safe_divide(
            tp + tn,
            tp + tn + fp + fn,
        ),
    }


def remove_small_components(mask, min_area):
    try:
        from scipy import ndimage
    except ImportError as exc:
        raise RuntimeError(
            "FINAL V4 EVALUATION FAILED.\n"
            "Reason: scipy is required for min-area filtering.\n"
            "Install with: pip install scipy"
        ) from exc

    mask = mask.astype(bool)

    if min_area <= 1:
        return mask

    structure = np.ones(
        (3, 3),
        dtype=np.uint8,
    )

    labels, count = ndimage.label(
        mask.astype(np.uint8),
        structure=structure,
    )

    if count == 0:
        return np.zeros_like(
            mask,
            dtype=bool,
        )

    sizes = np.bincount(
        labels.ravel()
    )

    keep = sizes >= min_area
    keep[0] = False

    return keep[labels]


def mask_to_polygon(mask):
    """
    Convert binary mask to polygon contours.

    OpenCV is used only for output geometry.
    """
    try:
        import cv2
    except ImportError as exc:
        raise RuntimeError(
            "FINAL V4 EVALUATION FAILED.\n"
            "Reason: opencv-python is required "
            "for polygon generation.\n"
            "Install with: pip install opencv-python"
        ) from exc

    binary = (
        mask.astype(np.uint8) * 255
    )

    contours, _ = cv2.findContours(
        binary,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE,
    )

    polygons = []

    for contour in contours:

        if len(contour) < 3:
            continue

        points = contour.reshape(
            -1,
            2,
        )

        polygons.append(
            points.tolist()
        )

    return polygons


# ============================================================
# CHECKPOINT
# ============================================================

def load_checkpoint(
    model,
    checkpoint_path,
    device,
):
    checkpoint = torch.load(
        checkpoint_path,
        map_location=device,
        weights_only=False,
    )

    if isinstance(checkpoint, dict):

        if "model_state_dict" in checkpoint:
            state_dict = checkpoint[
                "model_state_dict"
            ]

        elif "state_dict" in checkpoint:
            state_dict = checkpoint[
                "state_dict"
            ]

        else:
            state_dict = checkpoint

    else:
        state_dict = checkpoint

    if any(
        key.startswith("module.")
        for key in state_dict.keys()
    ):
        state_dict = {
            key.removeprefix("module."):
            value
            for key, value
            in state_dict.items()
        }

    model.load_state_dict(
        state_dict,
        strict=True,
    )

    return checkpoint


# ============================================================
# MAIN
# ============================================================

def main():

    parser = argparse.ArgumentParser(
        description=(
            "PS26143 final V4 untouched-test "
            "evaluation"
        )
    )

    parser.add_argument(
        "--checkpoint",
        type=Path,
        required=True,
    )

    parser.add_argument(
        "--test-manifest",
        type=Path,
        required=True,
    )

    parser.add_argument(
        "--output",
        type=Path,
        required=True,
    )

    parser.add_argument(
        "--threshold",
        type=float,
        required=True,
    )

    parser.add_argument(
        "--min-area",
        type=int,
        required=True,
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=16,
    )

    args = parser.parse_args()

    print("=" * 70)
    print("PS26143 — V4 FINAL TEST EVALUATION")
    print("=" * 70)

    print()
    print("Checkpoint :", args.checkpoint)
    print("Test       :", args.test_manifest)
    print("Threshold  :", args.threshold)
    print("Min area   :", args.min_area)

    # --------------------------------------------------------
    # DEVICE
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
            "GPU:",
            torch.cuda.get_device_name(0),
        )

    # --------------------------------------------------------
    # SAFETY CHECKS
    # --------------------------------------------------------

    if not args.checkpoint.exists():
        raise FileNotFoundError(
            "V4 checkpoint not found:\n"
            + str(args.checkpoint)
        )

    if not args.test_manifest.exists():
        raise FileNotFoundError(
            "Test manifest not found:\n"
            + str(args.test_manifest)
        )

    if not (
        0.0 < args.threshold < 1.0
    ):
        raise ValueError(
            "Threshold must be between 0 and 1."
        )

    if args.min_area < 0:
        raise ValueError(
            "Minimum area cannot be negative."
        )

    # --------------------------------------------------------
    # TEST MANIFEST
    # --------------------------------------------------------

    df = pd.read_csv(
        args.test_manifest
    )

    if len(df) != 135:
        raise RuntimeError(
            "TEST SET INTEGRITY FAILURE.\n"
            f"Expected exactly 135 test scenes, "
            f"found {len(df)}.\n"
            "STOPPING to protect the evaluation protocol."
        )

    required = {
        "global_id",
        "dataset",
        "image",
        "mask",
    }

    missing = required - set(df.columns)

    if missing:
        raise RuntimeError(
            "TEST MANIFEST MISSING REQUIRED COLUMNS:\n"
            + str(sorted(missing))
        )

    print()
    print("=" * 70)
    print("TEST DATASET")
    print("=" * 70)

    print("Samples:", len(df))
    print()
    print(
        df["dataset"].value_counts()
    )

    # --------------------------------------------------------
    # VERIFY ALL TEST FILES
    # --------------------------------------------------------

    for _, row in df.iterrows():

        image_path = Path(
            row["image"]
        )

        mask_path = Path(
            row["mask"]
        )

        if not image_path.exists():
            raise FileNotFoundError(
                "TEST IMAGE MISSING:\n"
                + str(image_path)
            )

        if not mask_path.exists():
            raise FileNotFoundError(
                "TEST MASK MISSING:\n"
                + str(mask_path)
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

    model.to(device)

    print("Architecture : U-Net")
    print("Encoder      : ResNet34")
    print("Input        : VV + VH")

    checkpoint = load_checkpoint(
        model,
        args.checkpoint,
        device,
    )

    if isinstance(
        checkpoint,
        dict,
    ):

        if "epoch" in checkpoint:
            print(
                "Checkpoint epoch:",
                checkpoint["epoch"],
            )

        if "best_val_dice" in checkpoint:
            print(
                "Best validation Dice:",
                checkpoint[
                    "best_val_dice"
                ],
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
        pin_memory=(
            device.type == "cuda"
        ),
    )

    # --------------------------------------------------------
    # EVALUATION
    # --------------------------------------------------------

    model.eval()

    rows = []

    total_tp = 0
    total_fp = 0
    total_fn = 0
    total_tn = 0

    oil_tp = 0
    oil_fp = 0
    oil_fn = 0
    oil_tn = 0

    oil_total = 0
    oil_detected = 0

    lookalike_total = 0
    lookalike_false_alarm = 0

    no_oil_total = 0
    no_oil_false_alarm = 0

    print()
    print("=" * 70)
    print("RUNNING FINAL V4 TEST")
    print("=" * 70)

    processed = 0

    with torch.no_grad():

        for batch in loader:

            images = batch[
                "image"
            ].to(
                device,
                non_blocking=True,
            )

            masks = batch[
                "mask"
            ].to(
                device,
                non_blocking=True,
            )

            if device.type == "cuda":

                with torch.autocast(
                    device_type="cuda",
                    dtype=torch.float16,
                ):
                    logits = model(
                        images
                    )

            else:
                logits = model(
                    images
                )

            probabilities = torch.sigmoid(
                logits.float()
            )

            probs_np = (
                probabilities
                .cpu()
                .numpy()
            )

            masks_np = (
                masks
                .cpu()
                .numpy()
            )

            for i in range(
                len(images)
            ):

                global_id = str(
                    batch["global_id"][i]
                )

                dataset_name = str(
                    batch["dataset"][i]
                )

                probability = probs_np[
                    i,
                    0,
                ]

                target = (
                    masks_np[
                        i,
                        0,
                    ] > 0.5
                )

                raw_prediction = (
                    probability
                    >= args.threshold
                )

                prediction = (
                    remove_small_components(
                        raw_prediction,
                        args.min_area,
                    )
                )

                tp, fp, fn, tn = counts(
                    prediction,
                    target,
                )

                total_tp += tp
                total_fp += fp
                total_fn += fn
                total_tn += tn

                scene_positive = bool(
                    prediction.any()
                )

                predicted_pixels = int(
                    prediction.sum()
                )

                target_pixels = int(
                    target.sum()
                )

                polygons = mask_to_polygon(
                    prediction
                )

                row = {
                    "global_id":
                        global_id,

                    "dataset":
                        dataset_name,

                    "predicted_pixels":
                        predicted_pixels,

                    "target_pixels":
                        target_pixels,

                    "tp": tp,
                    "fp": fp,
                    "fn": fn,
                    "tn": tn,

                    "dice":
                        metrics(
                            tp,
                            fp,
                            fn,
                            tn,
                        )["dice"],

                    "iou":
                        metrics(
                            tp,
                            fp,
                            fn,
                            tn,
                        )["iou"],

                    "precision":
                        metrics(
                            tp,
                            fp,
                            fn,
                            tn,
                        )["precision"],

                    "recall":
                        metrics(
                            tp,
                            fp,
                            fn,
                            tn,
                        )["recall"],

                    "max_probability":
                        float(
                            probability.max()
                        ),

                    "mean_probability":
                        float(
                            probability.mean()
                        ),

                    "scene_positive":
                        scene_positive,

                    "polygon_count":
                        len(polygons),
                }

                rows.append(row)

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
                        lookalike_false_alarm += 1

                elif dataset_name == "no_oil":

                    no_oil_total += 1

                    if scene_positive:
                        no_oil_false_alarm += 1

                processed += 1

    if processed != 135:
        raise RuntimeError(
            "FINAL TEST PROCESSING FAILURE.\n"
            f"Expected 135 samples, "
            f"processed {processed}."
        )

    # --------------------------------------------------------
    # METRICS
    # --------------------------------------------------------

    overall = metrics(
        total_tp,
        total_fp,
        total_fn,
        total_tn,
    )

    oil = metrics(
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
        lookalike_false_alarm,
        lookalike_total,
    )

    no_oil_far = safe_divide(
        no_oil_false_alarm,
        no_oil_total,
    )

    # --------------------------------------------------------
    # OUTPUT
    # --------------------------------------------------------

    args.output.mkdir(
        parents=True,
        exist_ok=True,
    )

    predictions_path = (
        args.output
        / "test_predictions.csv"
    )

    summary_path = (
        args.output
        / "test_summary.json"
    )

    polygons_dir = (
        args.output
        / "polygons"
    )

    polygons_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    predictions_df = pd.DataFrame(
        rows
    )

    predictions_df.to_csv(
        predictions_path,
        index=False,
    )

    summary = {
        "experiment":
            "oil-seg-v4",

        "source":
            "untouched_test",

        "checkpoint":
            str(args.checkpoint),

        "test_manifest":
            str(args.test_manifest),

        "threshold":
            float(args.threshold),

        "min_area":
            int(args.min_area),

        "test_samples":
            135,

        "overall": overall,

        "oil": {
            "samples":
                oil_total,

            **oil,

            "detection_rate":
                oil_detection_rate,
        },

        "lookalike": {
            "samples":
                lookalike_total,

            "false_alarm_scenes":
                lookalike_false_alarm,

            "false_alarm_rate":
                lookalike_far,
        },

        "no_oil": {
            "samples":
                no_oil_total,

            "false_alarm_scenes":
                no_oil_false_alarm,

            "false_alarm_rate":
                no_oil_far,
        },

        "false_positive_pixels": {
            "overall":
                total_fp,

            "oil":
                oil_fp,
        },

        "safety": {
            "training_used":
                False,

            "validation_used":
                False,

            "test_samples":
                135,

            "test_modified":
                False,

            "calibration_locked":
                True,
        },
    }

    with summary_path.open(
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            summary,
            f,
            indent=2,
        )

    # --------------------------------------------------------
    # FINAL REPORT
    # --------------------------------------------------------

    print()
    print("=" * 70)
    print("V4 FINAL TEST RESULTS")
    print("=" * 70)

    print()
    print("OVERALL")
    print(
        json.dumps(
            overall,
            indent=2,
        )
    )

    print()
    print("OIL")
    print(
        json.dumps(
            {
                **oil,
                "detection_rate":
                    oil_detection_rate,
            },
            indent=2,
        )
    )

    print()
    print("LOOKALIKE")
    print(
        "False-alarm scenes:",
        lookalike_false_alarm,
        "/",
        lookalike_total,
    )

    print(
        "False-alarm rate:",
        f"{lookalike_far:.6f}",
    )

    print()
    print("NO-OIL")
    print(
        "False-alarm scenes:",
        no_oil_false_alarm,
        "/",
        no_oil_total,
    )

    print(
        "False-alarm rate:",
        f"{no_oil_far:.6f}",
    )

    print()
    print("=" * 70)
    print("ARTIFACTS")
    print("=" * 70)

    print(
        "CSV:",
        predictions_path,
    )

    print(
        "JSON:",
        summary_path,
    )

    print()
    print("=" * 70)
    print("SAFETY")
    print("=" * 70)

    print("Training : NOT USED")
    print("Validation: NOT USED")
    print("Test     : 135 scenes")
    print("Test     : UNTOUCHED")
    print("Threshold:", args.threshold)
    print("Min area :", args.min_area)

    print()
    print("=" * 70)
    print("✅ V4 FINAL TEST EVALUATION COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()

