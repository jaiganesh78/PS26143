from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset


# ============================================================
# PATH SETUP
# ============================================================

HERE = Path(__file__).resolve()

ML_ROOT = HERE.parent.parent
REPO_ROOT = ML_ROOT.parent
SRC_ROOT = ML_ROOT / "src"

for path in [REPO_ROOT, ML_ROOT, SRC_ROOT]:
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))


from models.segmentation_model import build_model
from candidate_model import CandidateClassifier


# ============================================================
# CONSTANTS
# ============================================================

IMAGE_SIZE = 512
V1_THRESHOLD = 0.50


# ============================================================
# SIMPLE MANIFEST DATASET
# ============================================================

class TestManifestDataset(Dataset):

    def __init__(self, manifest_path):

        self.df = pd.read_csv(
            manifest_path
        )

        required = {
            "global_id",
            "dataset",
            "image",
            "mask",
        }

        missing = (
            required
            - set(self.df.columns)
        )

        if missing:
            raise RuntimeError(
                "Test manifest is missing "
                f"columns: {sorted(missing)}"
            )

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):

        row = self.df.iloc[idx]

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

        image = np.load(
            image_path
        ).astype(np.float32)

        mask = np.load(
            mask_path
        ).astype(np.float32)

        if image.shape != (
            2,
            IMAGE_SIZE,
            IMAGE_SIZE,
        ):
            raise ValueError(
                f"Unexpected image shape "
                f"{image.shape} for "
                f"{row['global_id']}"
            )

        if mask.shape != (
            IMAGE_SIZE,
            IMAGE_SIZE,
        ):
            raise ValueError(
                f"Unexpected mask shape "
                f"{mask.shape} for "
                f"{row['global_id']}"
            )

        return {
            "image": torch.from_numpy(
                image
            ),
            "mask": torch.from_numpy(
                mask
            ),
            "global_id": str(
                row["global_id"]
            ),
            "dataset": str(
                row["dataset"]
            ),
        }


# ============================================================
# CHECKPOINT LOADER
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

    if (
        isinstance(checkpoint, dict)
        and "model_state_dict"
        in checkpoint
    ):

        state_dict = checkpoint[
            "model_state_dict"
        ]

    elif (
        isinstance(checkpoint, dict)
        and "state_dict"
        in checkpoint
    ):

        state_dict = checkpoint[
            "state_dict"
        ]

    else:

        state_dict = checkpoint

    model.load_state_dict(
        state_dict,
        strict=True,
    )

    return checkpoint


# ============================================================
# BUILD V1
# ============================================================

def build_v1(device):

    model = build_model(
        architecture="unet",
        encoder="resnet34",
        encoder_weights=None,
        in_channels=2,
        classes=1,
    )

    model = model.to(device)

    model.eval()

    return model


# ============================================================
# BUILD V3
# ============================================================

def build_v3(device):

    model = CandidateClassifier(
        feature_dim=8,
        pretrained=False,
    )

    model = model.to(device)

    model.eval()

    return model


# ============================================================
# V1 PREDICTION
# ============================================================

def predict_v1(
    model,
    image,
    device,
):

    tensor = (
        torch.from_numpy(
            image
        )
        .unsqueeze(0)
        .to(device)
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
            )

    return (
        probability[
            0,
            0,
        ]
        .float()
        .cpu()
        .numpy()
    )


# ============================================================
# CANDIDATE EXTRACTION
# ============================================================

def extract_candidates(
    image,
    probability,
    threshold=0.50,
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

        width = int(
            stats[
                component_id,
                cv2.CC_STAT_WIDTH,
            ]
        )

        height = int(
            stats[
                component_id,
                cv2.CC_STAT_HEIGHT,
            ]
        )

        cx, cy = centroids[
            component_id
        ]

        center_x = int(
            round(cx)
        )

        center_y = int(
            round(cy)
        )

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

        crop_mask = (
            labels[
                sy0:sy1,
                sx0:sx1,
            ]
            == component_id
        ).astype(np.float32)

        if any([
            pad_left,
            pad_top,
            pad_right,
            pad_bottom,
        ]):

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

            crop_mask = np.pad(
                crop_mask,
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

        crop_mask = crop_mask[
            :crop_size,
            :crop_size,
        ]

        if (
            crop_image.shape[1]
            != crop_size
            or crop_image.shape[2]
            != crop_size
        ):

            channels = []

            for channel in crop_image:

                channels.append(
                    cv2.resize(
                        channel,
                        (
                            crop_size,
                            crop_size,
                        ),
                        interpolation=(
                            cv2.INTER_LINEAR
                        ),
                    )
                )

            crop_image = np.stack(
                channels,
                axis=0,
            )

        if (
            crop_mask.shape[0]
            != crop_size
            or crop_mask.shape[1]
            != crop_size
        ):

            crop_mask = cv2.resize(
                crop_mask,
                (
                    crop_size,
                    crop_size,
                ),
                interpolation=(
                    cv2.INTER_NEAREST
                ),
            )

        component_pixels = (
            labels
            == component_id
        )

        component_probs = (
            probability[
                component_pixels
            ]
        )

        if component_probs.size == 0:
            continue

        mean_probability = float(
            component_probs.mean()
        )

        p95_probability = float(
            np.percentile(
                component_probs,
                95,
            )
        )

        max_probability = float(
            component_probs.max()
        )

        normalized_area = (
            area
            / float(
                IMAGE_SIZE
                * IMAGE_SIZE
            )
        )

        log_area = (
            np.log1p(area)
            / np.log1p(
                IMAGE_SIZE
                * IMAGE_SIZE
            )
        )

        aspect_ratio = (
            width
            / max(
                height,
                1,
            )
        )

        features = np.array(
            [
                normalized_area,
                log_area,
                aspect_ratio / 10.0,
                cx / IMAGE_SIZE,
                cy / IMAGE_SIZE,
                mean_probability,
                p95_probability,
                max_probability,
            ],
            dtype=np.float32,
        )

        candidates.append(
            {
                "component_id":
                    component_id,

                "area":
                    area,

                "width":
                    width,

                "height":
                    height,

                "centroid_x":
                    float(cx),

                "centroid_y":
                    float(cy),

                "crop_image":
                    crop_image.astype(
                        np.float32
                    ),

                "crop_mask":
                    crop_mask.astype(
                        np.float32
                    ),

                "features":
                    features,

                "mean_probability":
                    mean_probability,

                "p95_probability":
                    p95_probability,

                "max_probability":
                    max_probability,
            }
        )

    return candidates


# ============================================================
# V3 PREDICTION
# ============================================================

def classify_candidates(
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
            candidate[
                "features"
            ]
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

    results = []

    for candidate, probability in zip(
        candidates,
        probabilities,
    ):

        item = dict(
            candidate
        )

        item[
            "v3_probability"
        ] = float(
            probability
        )

        item[
            "accepted"
        ] = bool(
            probability
            >= threshold
        )

        results.append(item)

    return results


# ============================================================
# MASK RECONSTRUCTION
# ============================================================

def reconstruct_mask(
    probability,
    candidates,
):

    base_binary = (
        probability >= V1_THRESHOLD
    ).astype(np.uint8)

    num_labels, labels = (
        cv2.connectedComponents(
            base_binary,
            connectivity=8,
        )
    )

    output = np.zeros_like(
        base_binary,
        dtype=np.uint8,
    )

    accepted = 0

    for candidate in candidates:

        if not candidate[
            "accepted"
        ]:
            continue

        component_id = candidate[
            "component_id"
        ]

        if component_id >= num_labels:
            continue

        output[
            labels == component_id
        ] = 1

        accepted += 1

    return output, accepted


# ============================================================
# PIXEL METRICS
# ============================================================

def calculate_metrics(
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
# POLYGON EXPORT
# ============================================================

def save_polygons(
    mask,
    path,
):

    contours, _ = cv2.findContours(
        mask.astype(np.uint8),
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE,
    )

    polygons = []

    for contour in contours:

        area = float(
            cv2.contourArea(
                contour
            )
        )

        if area <= 0:
            continue

        polygons.append(
            contour.reshape(
                -1,
                2,
            ).tolist()
        )

    with open(
        path,
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            {
                "polygons": polygons
            },
            f,
            indent=2,
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
        default=0.50,
    )

    args = parser.parse_args()

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print("=" * 70)
    print(
        "PS26143 — V3 FINAL TEST EVALUATION"
    )
    print("=" * 70)

    print(
        "Device:",
        device,
    )

    if device.type == "cuda":

        print(
            "GPU:",
            torch.cuda.get_device_name(0),
        )

    # --------------------------------------------------------
    # Test data
    # --------------------------------------------------------

    dataset = (
        TestManifestDataset(
            args.test_manifest
        )
    )

    print(
        "Test samples:",
        len(dataset),
    )

    # --------------------------------------------------------
    # V1
    # --------------------------------------------------------

    print()
    print(
        "Loading frozen V1..."
    )

    v1 = build_v1(
        device
    )

    checkpoint_v1 = (
        load_checkpoint(
            v1,
            args.v1_checkpoint,
            device,
        )
    )

    print(
        "V1 checkpoint loaded."
    )

    if isinstance(
        checkpoint_v1,
        dict,
    ):

        print(
            "V1 epoch:",
            checkpoint_v1.get(
                "epoch",
                "unknown",
            ),
        )

    # --------------------------------------------------------
    # V3
    # --------------------------------------------------------

    print()
    print(
        "Loading frozen V3..."
    )

    v3 = build_v3(
        device
    )

    checkpoint_v3 = (
        load_checkpoint(
            v3,
            args.v3_checkpoint,
            device,
        )
    )

    print(
        "V3 checkpoint loaded."
    )

    if isinstance(
        checkpoint_v3,
        dict,
    ):

        print(
            "V3 epoch:",
            checkpoint_v3.get(
                "epoch",
                "unknown",
            ),
        )

    # --------------------------------------------------------
    # Output
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

    # --------------------------------------------------------
    # Test inference
    # --------------------------------------------------------

    print()
    print("=" * 70)
    print(
        "RUNNING V3 ON 135 UNTOUCHED TEST SCENES"
    )
    print("=" * 70)

    for idx in range(
        len(dataset)
    ):

        sample = dataset[idx]

        image = (
            sample["image"]
            .numpy()
            .astype(np.float32)
        )

        target = (
            sample["mask"]
            .numpy()
            > 0.5
        ).astype(np.uint8)

        global_id = sample[
            "global_id"
        ]

        dataset_name = sample[
            "dataset"
        ]

        # ----------------------------------------------------
        # V1
        # ----------------------------------------------------

        probability = predict_v1(
            v1,
            image,
            device,
        )

        # ----------------------------------------------------
        # V1 candidates
        # ----------------------------------------------------

        candidates = extract_candidates(
            image,
            probability,
            threshold=V1_THRESHOLD,
        )

        original_candidate_count = (
            len(candidates)
        )

        # ----------------------------------------------------
        # V3 rejection
        # ----------------------------------------------------

        candidates = classify_candidates(
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

        metrics = calculate_metrics(
            prediction,
            target,
        )

        # ----------------------------------------------------
        # Candidate statistics
        # ----------------------------------------------------

        rejected_count = (
            original_candidate_count
            - accepted_count
        )

        v3_probabilities = [
            x["v3_probability"]
            for x in candidates
        ]

        max_v3_probability = (
            max(v3_probabilities)
            if v3_probabilities
            else 0.0
        )

        # ----------------------------------------------------
        # Save prediction mask
        # ----------------------------------------------------

        np.save(
            mask_dir
            / f"{global_id}.npy",
            prediction.astype(
                np.uint8
            ),
        )

        # ----------------------------------------------------
        # Save polygons
        # ----------------------------------------------------

        save_polygons(
            prediction,
            polygon_dir
            / f"{global_id}.json",
        )

        # ----------------------------------------------------
        # Row
        # ----------------------------------------------------

        row = {
            "global_id":
                global_id,

            "dataset":
                dataset_name,

            "v1_threshold":
                V1_THRESHOLD,

            "v3_threshold":
                args.threshold,

            "v1_candidates":
                original_candidate_count,

            "v3_accepted":
                accepted_count,

            "v3_rejected":
                rejected_count,

            "max_v3_probability":
                max_v3_probability,

            **metrics,
        }

        rows.append(row)

        print(
            f"{idx + 1:3d}/"
            f"{len(dataset)} "
            f"{global_id:20s} "
            f"{dataset_name:10s} "
            f"cand={original_candidate_count:3d} "
            f"accepted={accepted_count:3d} "
            f"Dice={metrics['dice']:.4f}"
        )

    # --------------------------------------------------------
    # DataFrame
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

    # --------------------------------------------------------
    # Summary
    # --------------------------------------------------------

    summary = {
        "experiment":
            "oil-seg-v3",

        "v1_checkpoint":
            str(args.v1_checkpoint),

        "v3_checkpoint":
            str(args.v3_checkpoint),

        "test_manifest":
            str(args.test_manifest),

        "test_samples":
            int(len(results)),

        "v1_threshold":
            V1_THRESHOLD,

        "v3_threshold":
            args.threshold,

        "device":
            str(device),

        "gpu":
            (
                torch.cuda.get_device_name(0)
                if device.type == "cuda"
                else None
            ),

        "overall_macro": {
            "dice":
                float(
                    results["dice"].mean()
                ),

            "iou":
                float(
                    results["iou"].mean()
                ),

            "precision":
                float(
                    results[
                        "precision"
                    ].mean()
                ),

            "recall":
                float(
                    results[
                        "recall"
                    ].mean()
                ),
        },

        "overall_micro": {
            "tp":
                int(
                    results["tp"].sum()
                ),

            "fp":
                int(
                    results["fp"].sum()
                ),

            "fn":
                int(
                    results["fn"].sum()
                ),

            "tn":
                int(
                    results["tn"].sum()
                ),
        },

        "by_dataset": {},

        "scene_level": {},
    }

    # --------------------------------------------------------
    # Dataset metrics
    # --------------------------------------------------------

    for name in [
        "oil",
        "lookalike",
        "no_oil",
    ]:

        subset = results[
            results["dataset"]
            == name
        ]

        if subset.empty:
            continue

        summary[
            "by_dataset"
        ][name] = {
            "samples":
                int(len(subset)),

            "dice":
                float(
                    subset["dice"].mean()
                ),

            "iou":
                float(
                    subset["iou"].mean()
                ),

            "precision":
                float(
                    subset[
                        "precision"
                    ].mean()
                ),

            "recall":
                float(
                    subset[
                        "recall"
                    ].mean()
                ),

            "false_positive_scenes":
                int(
                    (
                        subset["fp"]
                        > 0
                    ).sum()
                ),

            "false_positive_pixels":
                int(
                    subset["fp"].sum()
                ),
        }

    # --------------------------------------------------------
    # Scene-level metrics
    # --------------------------------------------------------

    oil = results[
        results["dataset"]
        == "oil"
    ]

    lookalike = results[
        results["dataset"]
        == "lookalike"
    ]

    no_oil = results[
        results["dataset"]
        == "no_oil"
    ]

    if not oil.empty:

        summary[
            "scene_level"
        ][
            "oil_detection_rate"
        ] = float(
            (
                oil[
                    "predicted_pixels"
                ]
                > 0
            ).mean()
        )

    if not lookalike.empty:

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

    if not no_oil.empty:

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
        "✅ V3 TEST EVALUATION COMPLETE"
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