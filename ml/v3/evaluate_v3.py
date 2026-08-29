from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

# ------------------------------------------------------------
# Make ml/src importable regardless of launch directory
# ------------------------------------------------------------

HERE = Path(__file__).resolve()
ML_ROOT = HERE.parent.parent
SRC_ROOT = ML_ROOT / "src"

if str(ML_ROOT) not in sys.path:
    sys.path.insert(0, str(ML_ROOT))

if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from datasets.oil_dataset import OilSegmentationDataset
from models.segmentation_model import build_model

from candidate_dataset import CandidateDataset
from candidate_model import CandidateClassifier


# ============================================================
# CONSTANTS
# ============================================================

IMAGE_SIZE = 512
V1_THRESHOLD = 0.50
DEFAULT_V3_THRESHOLD = 0.50


# ============================================================
# CHECKPOINT LOADING
# ============================================================

def load_checkpoint(model, path, device):

    checkpoint = torch.load(
        path,
        map_location=device,
        weights_only=False,
    )

    if "model_state_dict" in checkpoint:
        state = checkpoint["model_state_dict"]
    elif "state_dict" in checkpoint:
        state = checkpoint["state_dict"]
    else:
        state = checkpoint

    model.load_state_dict(
        state,
        strict=True,
    )

    return checkpoint


# ============================================================
# V1 MODEL
# ============================================================

def build_v1(device):

    model = build_model(
        architecture="unet",
        encoder="resnet34",
        encoder_weights=None,
        in_channels=2,
        classes=1,
    )

    return model.to(device)


# ============================================================
# V3 MODEL
# ============================================================

def build_v3(device):

    model = CandidateClassifier(
        feature_dim=8,
        pretrained=False,
    )

    return model.to(device)


# ============================================================
# FEATURE CONSTRUCTION
# ============================================================

def build_features(
    area,
    width,
    height,
    centroid_x,
    centroid_y,
    mean_probability,
    p95_probability,
    max_probability,
):

    width = max(float(width), 1.0)
    height = max(float(height), 1.0)

    aspect_ratio = width / height

    normalized_area = (
        float(area)
        / float(IMAGE_SIZE * IMAGE_SIZE)
    )

    log_area = (
        np.log1p(float(area))
        / np.log1p(
            float(IMAGE_SIZE * IMAGE_SIZE)
        )
    )

    return np.array(
        [
            normalized_area,
            log_area,
            aspect_ratio / 10.0,
            float(centroid_x) / IMAGE_SIZE,
            float(centroid_y) / IMAGE_SIZE,
            float(mean_probability),
            float(p95_probability),
            float(max_probability),
        ],
        dtype=np.float32,
    )


# ============================================================
# COMPONENT EXTRACTION
# ============================================================

def extract_candidates(
    image,
    probability,
    threshold=V1_THRESHOLD,
    min_area=20,
    crop_size=128,
    padding=16,
):

    binary = (
        probability >= threshold
    ).astype(np.uint8)

    num_labels, labels, stats, centroids = (
        cv2.connectedComponentsWithStats(
            binary,
            connectivity=8,
        )
    )

    candidates = []

    half = crop_size // 2

    for component_id in range(
        1,
        num_labels,
    ):

        area = int(
            stats[
                component_id,
                cv2.CC_STAT_AREA,
            ]
        )

        if area < min_area:
            continue

        x = int(
            stats[
                component_id,
                cv2.CC_STAT_LEFT,
            ]
        )

        y = int(
            stats[
                component_id,
                cv2.CC_STAT_TOP,
            ]
        )

        w = int(
            stats[
                component_id,
                cv2.CC_STAT_WIDTH,
            ]
        )

        h = int(
            stats[
                component_id,
                cv2.CC_STAT_HEIGHT,
            ]
        )

        cx, cy = centroids[
            component_id
        ]

        x0 = max(
            0,
            x - padding,
        )

        y0 = max(
            0,
            y - padding,
        )

        x1 = min(
            IMAGE_SIZE,
            x + w + padding,
        )

        y1 = min(
            IMAGE_SIZE,
            y + h + padding,
        )

        # Centered crop with fixed 128×128 output.
        center_x = int(round(cx))
        center_y = int(round(cy))

        sx0 = center_x - half
        sy0 = center_y - half
        sx1 = sx0 + crop_size
        sy1 = sy0 + crop_size

        pad_left = max(
            0,
            -sx0,
        )

        pad_top = max(
            0,
            -sy0,
        )

        pad_right = max(
            0,
            sx1 - IMAGE_SIZE,
        )

        pad_bottom = max(
            0,
            sy1 - IMAGE_SIZE,
        )

        sx0 = max(
            0,
            sx0,
        )

        sy0 = max(
            0,
            sy0,
        )

        sx1 = min(
            IMAGE_SIZE,
            sx1,
        )

        sy1 = min(
            IMAGE_SIZE,
            sy1,
        )

        crop_image = image[
            :,
            sy0:sy1,
            sx0:sx1,
        ]

        component_mask = (
            labels[
                sy0:sy1,
                sx0:sx1,
            ]
            == component_id
        ).astype(np.float32)

        if (
            pad_left
            or pad_top
            or pad_right
            or pad_bottom
        ):

            crop_image = np.pad(
                crop_image,
                (
                    (
                        0,
                        0,
                    ),
                    (
                        pad_top,
                        pad_bottom,
                    ),
                    (
                        pad_left,
                        pad_right,
                    ),
                ),
                mode="constant",
            )

            component_mask = np.pad(
                component_mask,
                (
                    (
                        pad_top,
                        pad_bottom,
                    ),
                    (
                        pad_left,
                        pad_right,
                    ),
                ),
                mode="constant",
            )

        crop_image = crop_image[
            :,
            :crop_size,
            :crop_size,
        ]

        component_mask = component_mask[
            :crop_size,
            :crop_size,
        ]

        # Safety resize if boundary arithmetic produced
        # anything other than exactly 128×128.
        if (
            crop_image.shape[1] != crop_size
            or crop_image.shape[2] != crop_size
        ):

            resized = []

            for channel in crop_image:

                resized.append(
                    cv2.resize(
                        channel,
                        (
                            crop_size,
                            crop_size,
                        ),
                        interpolation=cv2.INTER_LINEAR,
                    )
                )

            crop_image = np.stack(
                resized,
                axis=0,
            )

        if (
            component_mask.shape[0] != crop_size
            or component_mask.shape[1] != crop_size
        ):

            component_mask = cv2.resize(
                component_mask,
                (
                    crop_size,
                    crop_size,
                ),
                interpolation=cv2.INTER_NEAREST,
            )

        component_probability = (
            probability[
                sy0:sy1,
                sx0:sx1,
            ]
        )

        if component_probability.size == 0:
            continue

        mean_probability = float(
            component_probability[
                component_mask > 0.5
            ].mean()
        )

        p95_probability = float(
            np.percentile(
                component_probability[
                    component_mask > 0.5
                ],
                95,
            )
        )

        max_probability = float(
            component_probability.max()
        )

        features = build_features(
            area=area,
            width=w,
            height=h,
            centroid_x=cx,
            centroid_y=cy,
            mean_probability=mean_probability,
            p95_probability=p95_probability,
            max_probability=max_probability,
        )

        candidates.append(
            {
                "component_id": component_id,
                "area": area,
                "x": x,
                "y": y,
                "w": w,
                "h": h,
                "centroid_x": float(cx),
                "centroid_y": float(cy),
                "crop_image": crop_image.astype(
                    np.float32
                ),
                "crop_mask": component_mask.astype(
                    np.float32
                ),
                "features": features,
                "mean_probability": mean_probability,
                "p95_probability": p95_probability,
                "max_probability": max_probability,
            }
        )

    return candidates


# ============================================================
# NORMALIZATION
# ============================================================

def normalize_image(image):

    image = image.astype(
        np.float32
    )

    # Match the numerical representation expected by
    # the already-trained SAR model without altering
    # the candidate geometry.
    for c in range(image.shape[0]):

        channel = image[c]

        lo = np.percentile(
            channel,
            1,
        )

        hi = np.percentile(
            channel,
            99,
        )

        if hi > lo:

            channel = (
                channel - lo
            ) / (
                hi - lo
            )

        else:

            channel = np.zeros_like(
                channel
            )

        image[c] = np.clip(
            channel,
            0.0,
            1.0,
        )

    return image


# ============================================================
# V1 INFERENCE
# ============================================================

def predict_v1(
    model,
    image,
    device,
):

    tensor = torch.from_numpy(
        image
    ).unsqueeze(0).to(
        device,
        non_blocking=True,
    )

    with torch.inference_mode():

        with torch.autocast(
            device_type="cuda",
            dtype=torch.float16,
            enabled=(
                device.type == "cuda"
            ),
        ):

            logits = model(
                tensor
            )

            probability = torch.sigmoid(
                logits
            )[0, 0]

    return (
        probability
        .float()
        .cpu()
        .numpy()
    )


# ============================================================
# V3 CANDIDATE INFERENCE
# ============================================================

def predict_v3(
    model,
    candidates,
    device,
    threshold,
):

    if not candidates:
        return []

    images = []

    features = []

    for candidate in candidates:

        x = np.concatenate(
            [
                candidate[
                    "crop_image"
                ],
                candidate[
                    "crop_mask"
                ][None, ...],
            ],
            axis=0,
        )

        images.append(x)

        features.append(
            candidate["features"]
        )

    images = torch.from_numpy(
        np.stack(images)
    ).to(
        device,
        non_blocking=True,
    )

    features = torch.from_numpy(
        np.stack(features)
    ).to(
        device,
        non_blocking=True,
    )

    with torch.inference_mode():

        with torch.autocast(
            device_type="cuda",
            dtype=torch.float16,
            enabled=(
                device.type == "cuda"
            ),
        ):

            logits = model(
                images,
                features,
            )

            probabilities = (
                torch.sigmoid(
                    logits
                )
                .flatten()
                .float()
                .cpu()
                .numpy()
            )

    accepted = []

    for candidate, probability in zip(
        candidates,
        probabilities,
    ):

        candidate = dict(
            candidate
        )

        candidate[
            "v3_probability"
        ] = float(probability)

        candidate[
            "accepted"
        ] = bool(
            probability >= threshold
        )

        accepted.append(
            candidate
        )

    return accepted


# ============================================================
# MASK RECONSTRUCTION
# ============================================================

def reconstruct_mask(
    probability,
    candidates,
):

    mask = np.zeros(
        probability.shape,
        dtype=np.uint8,
    )

    accepted_count = 0

    for candidate in candidates:

        if not candidate[
            "accepted"
        ]:
            continue

        accepted_count += 1

        component_id = candidate[
            "component_id"
        ]

        threshold_mask = (
            probability >= V1_THRESHOLD
        ).astype(np.uint8)

        num_labels, labels = (
            cv2.connectedComponents(
                threshold_mask,
                connectivity=8,
            )
        )

        if component_id >= num_labels:
            continue

        mask[
            labels == component_id
        ] = 1

    return mask, accepted_count


# ============================================================
# METRICS
# ============================================================

def binary_metrics(
    prediction,
    target,
):

    prediction = (
        prediction > 0
    )

    target = (
        target > 0
    )

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

    dice = (
        2.0 * tp
        / max(
            2 * tp + fp + fn,
            1,
        )
    )

    iou = (
        tp
        / max(
            tp + fp + fn,
            1,
        )
    )

    precision = (
        tp
        / max(
            tp + fp,
            1,
        )
    )

    recall = (
        tp
        / max(
            tp + fn,
            1,
        )
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
        "f1": dice,
        "predicted_pixels": int(
            prediction.sum()
        ),
        "target_pixels": int(
            target.sum()
        ),
    }


# ============================================================
# DATASET RECORD EXTRACTION
# ============================================================

def get_record(dataset, idx):

    if hasattr(
        dataset,
        "records",
    ):

        return dataset.records[idx]

    if hasattr(
        dataset,
        "df",
    ):

        return dataset.df.iloc[idx].to_dict()

    raise RuntimeError(
        "Cannot access dataset record."
    )


# ============================================================
# MAIN
# ============================================================

def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--v1-checkpoint",
        type=Path,
        required=True,
    )

    parser.add_argument(
        "--v3-checkpoint",
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
        default=DEFAULT_V3_THRESHOLD,
    )

    args = parser.parse_args()

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print("=" * 70)
    print(
        "PS26143 — V3 TEST INFERENCE"
    )
    print("=" * 70)

    print("Device:", device)

    if device.type == "cuda":

        print(
            "GPU:",
            torch.cuda.get_device_name(0),
        )

    # --------------------------------------------------------
    # Test dataset
    # --------------------------------------------------------

    test_dataset = (
        OilSegmentationDataset(
            manifest_path=str(
                args.test_manifest
            ),
            image_size=IMAGE_SIZE,
            augment=False,
        )
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=0,
    )

    print(
        "Test samples:",
        len(test_dataset),
    )

    # --------------------------------------------------------
    # Models
    # --------------------------------------------------------

    print()
    print(
        "Loading frozen V1..."
    )

    v1 = build_v1(
        device
    )

    load_checkpoint(
        v1,
        args.v1_checkpoint,
        device,
    )

    v1.eval()

    print(
        "V1 loaded."
    )

    print()
    print(
        "Loading frozen V3..."
    )

    v3 = build_v3(
        device
    )

    load_checkpoint(
        v3,
        args.v3_checkpoint,
        device,
    )

    v3.eval()

    print(
        "V3 loaded."
    )

    # --------------------------------------------------------
    # Output directories
    # --------------------------------------------------------

    args.output.mkdir(
        parents=True,
        exist_ok=True,
    )

    mask_dir = (
        args.output
        / "masks"
    )

    polygon_dir = (
        args.output
        / "polygons"
    )

    mask_dir.mkdir(
        exist_ok=True
    )

    polygon_dir.mkdir(
        exist_ok=True
    )

    rows = []

    dataset_totals = {}

    # --------------------------------------------------------
    # Inference
    # --------------------------------------------------------

    print()
    print(
        "=" * 70
    )
    print(
        "RUNNING V3 ON UNTOUCHED TEST"
    )
    print(
        "=" * 70
    )

    for idx in range(
        len(test_dataset)
    ):

        sample = test_dataset[idx]

        # Dataset tensors are assumed C×H×W.
        image = sample[
            "image"
        ]

        target = sample[
            "mask"
        ]

        if torch.is_tensor(image):
            image = (
                image.cpu()
                .numpy()
            )

        if torch.is_tensor(target):
            target = (
                target.cpu()
                .numpy()
            )

        image = image.astype(
            np.float32
        )

        target = (
            target > 0.5
        ).astype(np.uint8)

        if target.ndim == 3:
            target = target.squeeze(0)

        # V1 dataset preprocessing has already been applied.
        probability = predict_v1(
            v1,
            image,
            device,
        )

        candidates = extract_candidates(
            image,
            probability,
            threshold=V1_THRESHOLD,
        )

        candidates = predict_v3(
            v3,
            candidates,
            device,
            args.threshold,
        )

        prediction, accepted_count = (
            reconstruct_mask(
                probability,
                candidates,
            )
        )

        metrics = binary_metrics(
            prediction,
            target,
        )

        record = get_record(
            test_dataset,
            idx,
        )

        global_id = str(
            record.get(
                "global_id",
                record.get(
                    "id",
                    f"sample_{idx:04d}",
                ),
            )
        )

        dataset_name = str(
            record.get(
                "dataset",
                "unknown",
            )
        )

        # ----------------------------------------------------
        # Save mask
        # ----------------------------------------------------

        np.save(
            mask_dir
            / f"{global_id}.npy",
            prediction.astype(
                np.uint8
            ),
        )

        # ----------------------------------------------------
        # Polygon extraction
        # ----------------------------------------------------

        contours, _ = cv2.findContours(
            prediction.astype(
                np.uint8
            ),
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE,
        )

        polygon_data = []

        for contour in contours:

            area = float(
                cv2.contourArea(
                    contour
                )
            )

            if area <= 0:
                continue

            polygon_data.append(
                contour.reshape(
                    -1,
                    2,
                ).tolist()
            )

        with open(
            polygon_dir
            / f"{global_id}.json",
            "w",
            encoding="utf-8",
        ) as f:

            json.dump(
                {
                    "global_id": global_id,
                    "polygons": polygon_data,
                },
                f,
            )

        row = {
            "global_id": global_id,
            "dataset": dataset_name,
            "v3_threshold": args.threshold,
            "v1_candidate_count": len(
                candidates
            ),
            "v3_accepted_candidates": accepted_count,
            **metrics,
        }

        rows.append(row)

        print(
            f"{idx + 1:3d}/"
            f"{len(test_dataset)} "
            f"{global_id:20s} "
            f"dataset={dataset_name:10s} "
            f"candidates={len(candidates):3d} "
            f"accepted={accepted_count:3d} "
            f"Dice={metrics['dice']:.4f}"
        )

    # --------------------------------------------------------
    # Results
    # --------------------------------------------------------

    results = pd.DataFrame(
        rows
    )

    csv_path = (
        args.output
        / "test_predictions.csv"
    )

    json_path = (
        args.output
        / "test_summary.json"
    )

    results.to_csv(
        csv_path,
        index=False,
    )

    summary = {
        "experiment": "oil-seg-v3",
        "v1_checkpoint": str(
            args.v1_checkpoint
        ),
        "v3_checkpoint": str(
            args.v3_checkpoint
        ),
        "test_manifest": str(
            args.test_manifest
        ),
        "test_samples": int(
            len(results)
        ),
        "v1_threshold": V1_THRESHOLD,
        "v3_threshold": args.threshold,
        "device": str(device),
        "gpu": (
            torch.cuda.get_device_name(0)
            if device.type == "cuda"
            else None
        ),
        "overall": {
            "dice": float(
                results["dice"].mean()
            ),
            "iou": float(
                results["iou"].mean()
            ),
            "precision": float(
                results["precision"].mean()
            ),
            "recall": float(
                results["recall"].mean()
            ),
        },
        "by_dataset": {},
        "scene_level": {},
    }

    for dataset_name in [
        "oil",
        "lookalike",
        "no_oil",
    ]:

        subset = results[
            results["dataset"]
            == dataset_name
        ]

        if len(subset) == 0:
            continue

        summary[
            "by_dataset"
        ][dataset_name] = {
            "samples": int(
                len(subset)
            ),
            "dice": float(
                subset["dice"].mean()
            ),
            "iou": float(
                subset["iou"].mean()
            ),
            "precision": float(
                subset[
                    "precision"
                ].mean()
            ),
            "recall": float(
                subset[
                    "recall"
                ].mean()
            ),
            "false_positive_scenes": int(
                (
                    subset["fp"]
                    > 0
                ).sum()
            ),
            "false_positive_rate": float(
                (
                    subset["fp"]
                    > 0
                ).mean()
            ),
            "false_positive_pixels": int(
                subset["fp"].sum()
            ),
        }

    oil = results[
        results["dataset"] == "oil"
    ]

    lookalike = results[
        results["dataset"]
        == "lookalike"
    ]

    no_oil = results[
        results["dataset"]
        == "no_oil"
    ]

    if len(oil):

        summary[
            "scene_level"
        ][
            "oil_detection_rate"
        ] = float(
            (
                oil["predicted_pixels"]
                > 0
            ).mean()
        )

    if len(lookalike):

        summary[
            "scene_level"
        ][
            "lookalike_false_alarm_rate"
        ] = float(
            (
                lookalike[
                    "predicted_pixels"
                ]
                > 0
            ).mean()
        )

    if len(no_oil):

        summary[
            "scene_level"
        ][
            "no_oil_false_alarm_rate"
        ] = float(
            (
                no_oil[
                    "predicted_pixels"
                ]
                > 0
            ).mean()
        )

    with open(
        json_path,
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            summary,
            f,
            indent=2,
        )

    print()
    print("=" * 70)
    print(
        "✅ V3 TEST INFERENCE COMPLETE"
    )
    print("=" * 70)

    print(
        "CSV:",
        csv_path,
    )

    print(
        "JSON:",
        json_path,
    )

    print(
        "Masks:",
        mask_dir,
    )

    print(
        "Polygons:",
        polygon_dir,
    )


if __name__ == "__main__":
    main()