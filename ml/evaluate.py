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
from src.training.losses import V2SegmentationLoss


# ============================================================
# DEFAULTS
# ============================================================

DRIVE_ROOT = Path(
    "/content/drive/MyDrive/PS26143"
)

DEFAULT_TEST_MANIFEST = (
    DRIVE_ROOT
    / "data/processed/test/manifest.csv"
)

DEFAULT_V1_CHECKPOINT = (
    DRIVE_ROOT
    / "checkpoints/oil_seg_v1_best.pt"
)

DEFAULT_V2_CHECKPOINT = (
    DRIVE_ROOT
    / "checkpoints/oil_seg_v2_best.pt"
)


# ============================================================
# METRICS
# ============================================================

def safe_divide(a, b):
    if b == 0:
        return 0.0

    return float(a / b)


def segmentation_metrics(
    prediction,
    target,
):
    prediction = prediction.astype(bool)
    target = target.astype(bool)

    tp = int(
        np.logical_and(
            prediction,
            target,
        ).sum()
    )

    fp = int(
        np.logical_and(
            prediction,
            ~target,
        ).sum()
    )

    fn = int(
        np.logical_and(
            ~prediction,
            target,
        ).sum()
    )

    tn = int(
        np.logical_and(
            ~prediction,
            ~target,
        ).sum()
    )

    dice = safe_divide(
        2.0 * tp,
        2.0 * tp + fp + fn,
    )

    iou = safe_divide(
        tp,
        tp + fp + fn,
    )

    precision = safe_divide(
        tp,
        tp + fp,
    )

    recall = safe_divide(
        tp,
        tp + fn,
    )

    f1 = safe_divide(
        2.0 * precision * recall,
        precision + recall,
    )

    accuracy = safe_divide(
        tp + tn,
        tp + tn + fp + fn,
    )

    return {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
        "dice": dice,
        "iou": iou,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "accuracy": accuracy,
        "target_positive_pixels": int(
            target.sum()
        ),
        "predicted_positive_pixels": int(
            prediction.sum()
        ),
    }


# ============================================================
# CHECKPOINT LOADING
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

    # Handle checkpoints saved through
    # DataParallel/DDP if ever encountered.
    if any(
        key.startswith("module.")
        for key in state_dict.keys()
    ):
        state_dict = {
            key.removeprefix("module."): value
            for key, value in state_dict.items()
        }

    model.load_state_dict(
        state_dict,
        strict=True,
    )

    return checkpoint

    if isinstance(
        checkpoint,
        dict,
    ):
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

    model.load_state_dict(
        state_dict,
        strict=True,
    )

    return checkpoint


# ============================================================
# MAIN
# ============================================================

def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--checkpoint",
        type=Path,
        required=True,
    )

    parser.add_argument(
        "--test-manifest",
        type=Path,
        default=DEFAULT_TEST_MANIFEST,
    )

    parser.add_argument(
        "--experiment",
        type=str,
        required=True,
    )

    parser.add_argument(
        "--threshold",
        type=float,
        default=0.5,
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=16,
    )

    args = parser.parse_args()

    print("=" * 70)
    print(
        f"PS26143 — {args.experiment.upper()} TEST EVALUATION"
    )
    print("=" * 70)

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
    # INPUT CHECKS
    # --------------------------------------------------------

    if not args.checkpoint.exists():
        raise FileNotFoundError(
            args.checkpoint
        )

    if not args.test_manifest.exists():
        raise FileNotFoundError(
            args.test_manifest
        )

    # --------------------------------------------------------
    # MANIFEST
    # --------------------------------------------------------

    df = pd.read_csv(
        args.test_manifest
    )

    if len(df) != 135:
        raise RuntimeError(
            f"Expected 135 test samples, "
            f"found {len(df)}"
        )

    required_columns = {
        "global_id",
        "dataset",
        "image",
        "mask",
    }

    missing_columns = (
        required_columns
        - set(df.columns)
    )

    if missing_columns:
        raise RuntimeError(
            f"Manifest missing columns: "
            f"{sorted(missing_columns)}"
        )

    print()
    print(
        "Test samples:",
        len(df),
    )

    print()
    print(
        "Dataset distribution:"
    )

    print(
        df["dataset"].value_counts()
    )

    # --------------------------------------------------------
    # PATH VALIDATION
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
                image_path
            )

        if not mask_path.exists():
            raise FileNotFoundError(
                mask_path
            )

    print()
    print(
        "All test image/mask paths verified."
    )

    # --------------------------------------------------------
    # MODEL
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

    model = model.to(device)

    print(
        "Architecture : U-Net"
    )

    print(
        "Encoder      : ResNet34"
    )

    print(
        "Input        : 2 channels"
    )

    print(
        "Output       : 1 channel"
    )

    # --------------------------------------------------------
    # CHECKPOINT
    # --------------------------------------------------------

    print()
    print(
        "=" * 70
    )

    print(
        "LOADING CHECKPOINT"
    )

    print(
        "=" * 70
    )

    checkpoint = load_checkpoint(
        model,
        args.checkpoint,
        device,
    )

    print(
        "Checkpoint:",
        args.checkpoint,
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

    records = df.to_dict(
        "records"
    )

    dataset = OilSegmentationDataset(
        records,
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
    # PREDICTION
    # --------------------------------------------------------

    model.eval()

    per_sample = []

    total_tp = 0
    total_fp = 0
    total_fn = 0
    total_tn = 0

    dataset_accumulator = {}

    print()
    print("=" * 70)
    print("GENERATING TEST PREDICTIONS")
    print("=" * 70)

    with torch.no_grad():

        processed = 0

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

            # ------------------------------------------------
            # AMP inference
            # ------------------------------------------------

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

            predictions = (
                probabilities
                >= args.threshold
            )

            images_np = (
                predictions
                .cpu()
                .numpy()
            )

            masks_np = (
                masks
                .cpu()
                .numpy()
            )

            probabilities_np = (
                probabilities
                .cpu()
                .numpy()
            )

            for i in range(
                len(images_np)
            ):

                prediction = (
                    images_np[i, 0]
                )

                target = (
                    masks_np[i, 0]
                    > 0.5
                )

                probability = (
                    probabilities_np[
                        i, 0
                    ]
                )

                metrics = (
                    segmentation_metrics(
                        prediction,
                        target,
                    )
                )

                global_id = str(
                    batch[
                        "global_id"
                    ][i]
                )

                dataset_name = str(
                    batch[
                        "dataset"
                    ][i]
                )

                predicted_area = int(
                    prediction.sum()
                )

                target_area = int(
                    target.sum()
                )

                max_probability = float(
                    probability.max()
                )

                scene_positive = (
                    predicted_area > 0
                )

                target_positive = (
                    target_area > 0
                )

                record = {
                    "global_id": global_id,
                    "dataset": dataset_name,
                    "tp": metrics["tp"],
                    "fp": metrics["fp"],
                    "fn": metrics["fn"],
                    "tn": metrics["tn"],
                    "dice": metrics["dice"],
                    "iou": metrics["iou"],
                    "precision": metrics[
                        "precision"
                    ],
                    "recall": metrics[
                        "recall"
                    ],
                    "f1": metrics["f1"],
                    "accuracy": metrics[
                        "accuracy"
                    ],
                    "target_positive_pixels":
                        target_area,
                    "predicted_positive_pixels":
                        predicted_area,
                    "max_probability":
                        max_probability,
                    "target_positive_scene":
                        target_positive,
                    "predicted_positive_scene":
                        scene_positive,
                }

                per_sample.append(
                    record
                )

                total_tp += metrics[
                    "tp"
                ]

                total_fp += metrics[
                    "fp"
                ]

                total_fn += metrics[
                    "fn"
                ]

                total_tn += metrics[
                    "tn"
                ]

                if dataset_name not in (
                    dataset_accumulator
                ):
                    dataset_accumulator[
                        dataset_name
                    ] = []

                dataset_accumulator[
                    dataset_name
                ].append(
                    record
                )

                processed += 1

            if (
                processed == 1
                or processed % 25 == 0
                or processed == len(df)
            ):
                print(
                    f"Processed: "
                    f"{processed}/{len(df)}"
                )

    # --------------------------------------------------------
    # OVERALL MICRO
    # --------------------------------------------------------

    micro_dice = safe_divide(
        2 * total_tp,
        2 * total_tp
        + total_fp
        + total_fn,
    )

    micro_iou = safe_divide(
        total_tp,
        total_tp
        + total_fp
        + total_fn,
    )

    micro_precision = safe_divide(
        total_tp,
        total_tp + total_fp,
    )

    micro_recall = safe_divide(
        total_tp,
        total_tp + total_fn,
    )

    micro_f1 = safe_divide(
        2
        * micro_precision
        * micro_recall,
        micro_precision
        + micro_recall,
    )

    micro_accuracy = safe_divide(
        total_tp + total_tn,
        total_tp
        + total_tn
        + total_fp
        + total_fn,
    )

    # --------------------------------------------------------
    # MACRO
    # --------------------------------------------------------

    sample_df = pd.DataFrame(
        per_sample
    )

    macro_metrics = {
        metric: float(
            sample_df[metric].mean()
        )
        for metric in [
            "dice",
            "iou",
            "precision",
            "recall",
            "f1",
            "accuracy",
        ]
    }

    # --------------------------------------------------------
    # DATASET METRICS
    # --------------------------------------------------------

    by_dataset = {}

    for dataset_name, rows in (
        dataset_accumulator.items()
    ):

        ddf = pd.DataFrame(rows)

        tp = int(
            ddf["tp"].sum()
        )

        fp = int(
            ddf["fp"].sum()
        )

        fn = int(
            ddf["fn"].sum()
        )

        tn = int(
            ddf["tn"].sum()
        )

        dice = safe_divide(
            2 * tp,
            2 * tp + fp + fn,
        )

        iou = safe_divide(
            tp,
            tp + fp + fn,
        )

        precision = safe_divide(
            tp,
            tp + fp,
        )

        recall = safe_divide(
            tp,
            tp + fn,
        )

        f1 = safe_divide(
            2 * precision * recall,
            precision + recall,
        )

        accuracy = safe_divide(
            tp + tn,
            tp + tn + fp + fn,
        )

        by_dataset[
            dataset_name
        ] = {
            "samples": len(ddf),
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "tn": tn,
            "dice": dice,
            "iou": iou,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "accuracy": accuracy,
            "target_positive_pixels":
                int(
                    ddf[
                        "target_positive_pixels"
                    ].sum()
                ),
            "predicted_positive_pixels":
                int(
                    ddf[
                        "predicted_positive_pixels"
                    ].sum()
                ),
        }

    # --------------------------------------------------------
    # SCENE LEVEL
    # --------------------------------------------------------

    oil_df = sample_df[
        sample_df["dataset"] == "oil"
    ]

    lookalike_df = sample_df[
        sample_df["dataset"]
        == "lookalike"
    ]

    no_oil_df = sample_df[
        sample_df["dataset"]
        == "no_oil"
    ]

    oil_detection_rate = safe_divide(
        int(
            oil_df[
                "predicted_positive_scene"
            ].sum()
        ),
        len(oil_df),
    )

    lookalike_false_alarm_rate = safe_divide(
        int(
            lookalike_df[
                "predicted_positive_scene"
            ].sum()
        ),
        len(lookalike_df),
    )

    no_oil_false_alarm_rate = safe_divide(
        int(
            no_oil_df[
                "predicted_positive_scene"
            ].sum()
        ),
        len(no_oil_df),
    )

    # --------------------------------------------------------
    # SUMMARY
    # --------------------------------------------------------

    summary = {
        "experiment": args.experiment,
        "checkpoint": str(
            args.checkpoint
        ),
        "test_manifest": str(
            args.test_manifest
        ),
        "test_samples": len(df),
        "threshold": args.threshold,
        "device": str(device),
        "gpu": (
            torch.cuda.get_device_name(0)
            if device.type == "cuda"
            else "CPU"
        ),
        "architecture": "U-Net",
        "encoder": "ResNet34",
        "in_channels": 2,
        "classes": 1,
        "image_size": 512,
        "overall_micro": {
            "samples": len(df),
            "tp": total_tp,
            "fp": total_fp,
            "fn": total_fn,
            "tn": total_tn,
            "dice": micro_dice,
            "iou": micro_iou,
            "precision": micro_precision,
            "recall": micro_recall,
            "f1": micro_f1,
            "accuracy": micro_accuracy,
            "target_positive_pixels": int(
                sample_df[
                    "target_positive_pixels"
                ].sum()
            ),
            "predicted_positive_pixels":
                int(
                    sample_df[
                        "predicted_positive_pixels"
                    ].sum()
                ),
        },
        "overall_macro": macro_metrics,
        "by_dataset": by_dataset,
        "scene_level": {
            "oil_detection_rate":
                oil_detection_rate,
            "lookalike_false_alarm_rate":
                lookalike_false_alarm_rate,
            "no_oil_false_alarm_rate":
                no_oil_false_alarm_rate,
        },
        "oil_scene_detection": {
            "oil_samples": len(oil_df),
            "oil_samples_predicted_positive":
                int(
                    oil_df[
                        "predicted_positive_scene"
                    ].sum()
                ),
            "oil_samples_predicted_empty":
                int(
                    (
                        ~oil_df[
                            "predicted_positive_scene"
                        ]
                    ).sum()
                ),
        },
    }

    # --------------------------------------------------------
    # OUTPUT
    # --------------------------------------------------------

    output_dir = (
        DRIVE_ROOT
        / "evaluation"
        / args.experiment
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    csv_path = (
        output_dir
        / "test_predictions.csv"
    )

    json_path = (
        output_dir
        / "test_summary.json"
    )

    sample_df.to_csv(
        csv_path,
        index=False,
    )

    with json_path.open(
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            summary,
            f,
            indent=2,
        )

    # --------------------------------------------------------
    # PRINT RESULTS
    # --------------------------------------------------------

    print()
    print("=" * 70)
    print("V2 TEST RESULTS")
    print("=" * 70)

    print()
    print(
        "Overall micro Dice :",
        f"{micro_dice:.6f}",
    )

    print(
        "Overall micro IoU  :",
        f"{micro_iou:.6f}",
    )

    print(
        "Precision           :",
        f"{micro_precision:.6f}",
    )

    print(
        "Recall              :",
        f"{micro_recall:.6f}",
    )

    print()
    print("BY DATASET")

    for name, metrics in (
        by_dataset.items()
    ):

        print()
        print(
            name.upper()
        )

        print(
            "  Dice      :",
            f"{metrics['dice']:.6f}",
        )

        print(
            "  IoU       :",
            f"{metrics['iou']:.6f}",
        )

        print(
            "  Precision :",
            f"{metrics['precision']:.6f}",
        )

        print(
            "  Recall    :",
            f"{metrics['recall']:.6f}",
        )

    print()
    print("SCENE LEVEL")

    print(
        "  Oil detection:",
        f"{oil_detection_rate:.2%}",
    )

    print(
        "  Lookalike false alarm:",
        f"{lookalike_false_alarm_rate:.2%}",
    )

    print(
        "  No-oil false alarm:",
        f"{no_oil_false_alarm_rate:.2%}",
    )

    print()
    print("CSV:")
    print(csv_path)

    print()
    print("JSON:")
    print(json_path)

    print()
    print("=" * 70)
    print("TEST EVALUATION COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()