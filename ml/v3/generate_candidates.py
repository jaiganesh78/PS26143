from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch


# ============================================================
# REPOSITORY IMPORT PATH
# ============================================================

REPO_ROOT = Path(__file__).resolve().parents[2]

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from ml.src.models.segmentation_model import build_model


# ============================================================
# DEFAULT PATHS
# ============================================================

DEFAULT_CHECKPOINT = Path(
    "/content/drive/MyDrive/PS26143/"
    "checkpoints/oil_seg_v1_best.pt"
)

DEFAULT_MANIFEST = Path(
    "/content/drive/MyDrive/PS26143/"
    "data/processed/train/manifest.csv"
)

DEFAULT_OUTPUT = Path(
    "/content/drive/MyDrive/PS26143/"
    "evaluation/oil_seg_v3/candidates"
)

DEFAULT_THRESHOLD = 0.50
DEFAULT_MIN_AREA = 20
DEFAULT_MATCH_IOU = 0.30

SEED = 42


# ============================================================
# REPRODUCIBILITY
# ============================================================

def seed_everything(seed=SEED):

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ============================================================
# CHECKPOINT
# ============================================================

def load_checkpoint(
    model,
    checkpoint_path,
    device,
):

    try:

        checkpoint = torch.load(
            checkpoint_path,
            map_location=device,
            weights_only=False,
        )

    except TypeError:

        checkpoint = torch.load(
            checkpoint_path,
            map_location=device,
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

    cleaned = {}

    for key, value in state_dict.items():

        if key.startswith("module."):

            key = key[len("module."):]

        cleaned[key] = value

    model.load_state_dict(
        cleaned,
        strict=True,
    )

    metadata = {}

    if isinstance(checkpoint, dict):

        metadata = {
            "epoch": checkpoint.get("epoch"),
            "best_val_dice": checkpoint.get(
                "best_val_dice"
            ),
        }

    return metadata


# ============================================================
# MODEL
# ============================================================

def build_v1_model(device):

    model = build_model(
        architecture="unet",
        encoder="resnet34",
        encoder_weights=None,
        in_channels=2,
        classes=1,
    )

    model.to(device)
    model.eval()

    return model


# ============================================================
# COMPONENT EXTRACTION
# ============================================================

def extract_components(
    probability,
    threshold,
    min_area,
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

    components = []

    for label_id in range(
        1,
        num_labels,
    ):

        area = int(
            stats[
                label_id,
                cv2.CC_STAT_AREA,
            ]
        )

        if area < min_area:
            continue

        x = int(
            stats[
                label_id,
                cv2.CC_STAT_LEFT,
            ]
        )

        y = int(
            stats[
                label_id,
                cv2.CC_STAT_TOP,
            ]
        )

        width = int(
            stats[
                label_id,
                cv2.CC_STAT_WIDTH,
            ]
        )

        height = int(
            stats[
                label_id,
                cv2.CC_STAT_HEIGHT,
            ]
        )

        component_mask = (
            labels == label_id
        )

        pixels = probability[
            component_mask
        ]

        components.append(
            {
                "label_id": label_id,
                "area": area,
                "x": x,
                "y": y,
                "width": width,
                "height": height,
                "cx": float(
                    centroids[label_id][0]
                ),
                "cy": float(
                    centroids[label_id][1]
                ),
                "mean_probability": float(
                    pixels.mean()
                ),
                "p95_probability": float(
                    np.percentile(
                        pixels,
                        95,
                    )
                ),
                "max_probability": float(
                    pixels.max()
                ),
                "mask": component_mask,
            }
        )

    return components


# ============================================================
# IOU
# ============================================================

def calculate_iou(
    candidate_mask,
    gt_mask,
):

    intersection = np.logical_and(
        candidate_mask,
        gt_mask,
    ).sum()

    union = np.logical_or(
        candidate_mask,
        gt_mask,
    ).sum()

    if union == 0:

        return 0.0

    return float(
        intersection / union
    )


# ============================================================
# CROP
# ============================================================

def extract_candidate_crop(
    image,
    candidate_mask,
    bbox,
    padding,
):

    h, w = image.shape[1:]

    x = bbox["x"]
    y = bbox["y"]

    width = bbox["width"]
    height = bbox["height"]

    x1 = max(
        0,
        x - padding,
    )

    y1 = max(
        0,
        y - padding,
    )

    x2 = min(
        w,
        x + width + padding,
    )

    y2 = min(
        h,
        y + height + padding,
    )

    image_crop = image[
        :,
        y1:y2,
        x1:x2,
    ]

    mask_crop = candidate_mask[
        y1:y2,
        x1:x2,
    ]

    return (
        image_crop.astype(
            np.float32
        ),
        mask_crop.astype(
            np.uint8
        ),
    )


# ============================================================
# RESIZE
# ============================================================

def resize_candidate(
    image,
    mask,
    size,
):

    bands = []

    for band in image:

        band = cv2.resize(
            band,
            (size, size),
            interpolation=cv2.INTER_LINEAR,
        )

        bands.append(
            band.astype(
                np.float32
            )
        )

    image_out = np.stack(
        bands,
        axis=0,
    )

    mask_out = cv2.resize(
        mask,
        (size, size),
        interpolation=cv2.INTER_NEAREST,
    )

    return (
        image_out.astype(
            np.float32
        ),
        mask_out.astype(
            np.uint8
        ),
    )


# ============================================================
# SAVE
# ============================================================

def save_npy(
    path,
    array,
):

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    np.save(
        path,
        array,
    )


# ============================================================
# MAIN
# ============================================================

def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=DEFAULT_CHECKPOINT,
    )

    parser.add_argument(
        "--manifest",
        type=Path,
        default=DEFAULT_MANIFEST,
    )

    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
    )

    parser.add_argument(
        "--threshold",
        type=float,
        default=DEFAULT_THRESHOLD,
    )

    parser.add_argument(
        "--min-area",
        type=int,
        default=DEFAULT_MIN_AREA,
    )

    parser.add_argument(
        "--match-iou",
        type=float,
        default=DEFAULT_MATCH_IOU,
    )

    parser.add_argument(
        "--crop-size",
        type=int,
        default=128,
    )

    parser.add_argument(
        "--padding",
        type=int,
        default=16,
    )

    args = parser.parse_args()

    seed_everything()

    # ========================================================
    # SAFETY
    # ========================================================

    manifest_lower = str(
        args.manifest
    ).lower()

    if (
        "/test/" in manifest_lower
        or "test_manifest" in manifest_lower
    ):

        raise RuntimeError(
            "FATAL: test manifest detected. "
            "V3 candidate generation is training-only."
        )

    if not args.checkpoint.exists():

        raise FileNotFoundError(
            f"Missing checkpoint: "
            f"{args.checkpoint}"
        )

    if not args.manifest.exists():

        raise FileNotFoundError(
            f"Missing manifest: "
            f"{args.manifest}"
        )

    # ========================================================
    # DEVICE
    # ========================================================

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print("=" * 70)
    print(
        "PS26143 — V3 CANDIDATE GENERATION"
    )
    print("=" * 70)

    print(
        "Repository:",
        REPO_ROOT,
    )

    print(
        "Device:",
        device,
    )

    if device.type == "cuda":

        print(
            "GPU:",
            torch.cuda.get_device_name(0),
        )

    print()
    print(
        "Checkpoint:",
        args.checkpoint,
    )

    print(
        "Manifest:",
        args.manifest,
    )

    print(
        "Threshold:",
        args.threshold,
    )

    print(
        "Minimum area:",
        args.min_area,
    )

    print(
        "Positive IoU threshold:",
        args.match_iou,
    )

    # ========================================================
    # MANIFEST
    # ========================================================

    df = pd.read_csv(
        args.manifest
    )

    required = {
        "global_id",
        "dataset",
        "image",
        "mask",
    }

    missing = (
        required
        - set(df.columns)
    )

    if missing:

        raise RuntimeError(
            "Manifest missing columns: "
            f"{sorted(missing)}"
        )

    if len(df) != 630:

        print(
            "WARNING:"
            f" Expected 630 training scenes, "
            f"found {len(df)}."
        )

    # ========================================================
    # PATH CHECK
    # ========================================================

    for row in df.itertuples(
        index=False
    ):

        image_path = Path(
            row.image
        )

        mask_path = Path(
            row.mask
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
        "All training image/mask paths verified."
    )

    # ========================================================
    # MODEL
    # ========================================================

    print()
    print("=" * 70)
    print(
        "LOADING FROZEN V1 MODEL"
    )
    print("=" * 70)

    model = build_v1_model(
        device
    )

    metadata = load_checkpoint(
        model,
        args.checkpoint,
        device,
    )

    print(
        "Checkpoint loaded."
    )

    print(
        "Checkpoint epoch:",
        metadata.get("epoch"),
    )

    print(
        "Best validation Dice:",
        metadata.get(
            "best_val_dice"
        ),
    )

    # ========================================================
    # OUTPUT
    # ========================================================

    args.output.mkdir(
        parents=True,
        exist_ok=True,
    )

    positive_dir = (
        args.output
        / "patches"
        / "positive"
    )

    negative_dir = (
        args.output
        / "patches"
        / "hard_negative"
    )

    positive_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    negative_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    records = []

    total_candidates = 0
    positive_candidates = 0
    negative_candidates = 0

    positive_scenes = 0
    negative_scenes = 0

    oil_scenes = 0
    oil_scenes_with_candidate = 0

    # ========================================================
    # INFERENCE
    # ========================================================

    print()
    print("=" * 70)
    print(
        "GENERATING V1 CANDIDATES"
    )
    print("=" * 70)

    for index, row in enumerate(
        df.itertuples(index=False),
        start=1,
    ):

        global_id = str(
            row.global_id
        )

        dataset = str(
            row.dataset
        )

        image = np.load(
            Path(row.image)
        ).astype(
            np.float32
        )

        gt_mask = np.load(
            Path(row.mask)
        ).astype(
            np.float32
        )

        if image.shape[0] != 2:

            raise ValueError(
                f"{global_id}: "
                f"expected [2,H,W], "
                f"got {image.shape}"
            )

        gt_binary = (
            gt_mask > 0.5
        )

        if dataset == "oil":

            oil_scenes += 1

        tensor = torch.from_numpy(
            image
        ).unsqueeze(0).to(
            device,
            non_blocking=True,
        )

        with torch.inference_mode():

            if device.type == "cuda":

                with torch.autocast(
                    device_type="cuda",
                    dtype=torch.float16,
                ):

                    logits = model(
                        tensor
                    )

            else:

                logits = model(
                    tensor
                )

            probability = (
                torch.sigmoid(
                    logits
                )[0, 0]
                .float()
                .cpu()
                .numpy()
            )

        components = extract_components(
            probability,
            args.threshold,
            args.min_area,
        )

        if (
            dataset == "oil"
            and len(components) > 0
        ):

            oil_scenes_with_candidate += 1

        scene_positive = False
        scene_negative = False

        for candidate_index, component in enumerate(
            components,
            start=1,
        ):

            candidate_mask = component[
                "mask"
            ]

            iou = calculate_iou(
                candidate_mask,
                gt_binary,
            )

            if iou >= args.match_iou:

                label = "positive"

                positive_candidates += 1
                scene_positive = True

                patch_dir = positive_dir

            else:

                label = "hard_negative"

                negative_candidates += 1
                scene_negative = True

                patch_dir = negative_dir

            total_candidates += 1

            crop_image, crop_mask = (
                extract_candidate_crop(
                    image,
                    candidate_mask,
                    component,
                    args.padding,
                )
            )

            crop_image, crop_mask = (
                resize_candidate(
                    crop_image,
                    crop_mask,
                    args.crop_size,
                )
            )

            candidate_id = (
                f"{global_id}"
                f"_candidate_"
                f"{candidate_index:03d}"
            )

            image_path = (
                patch_dir
                / f"{candidate_id}.npy"
            )

            mask_path = (
                patch_dir
                / f"{candidate_id}_mask.npy"
            )

            save_npy(
                image_path,
                crop_image,
            )

            save_npy(
                mask_path,
                crop_mask,
            )

            records.append(
                {
                    "global_id": global_id,
                    "dataset": dataset,
                    "candidate_id": candidate_id,
                    "candidate_index": candidate_index,
                    "label": label,
                    "gt_iou": float(iou),
                    "area": component["area"],
                    "x": component["x"],
                    "y": component["y"],
                    "width": component["width"],
                    "height": component["height"],
                    "centroid_x": component["cx"],
                    "centroid_y": component["cy"],
                    "mean_probability": component[
                        "mean_probability"
                    ],
                    "p95_probability": component[
                        "p95_probability"
                    ],
                    "max_probability": component[
                        "max_probability"
                    ],
                    "crop_image": str(
                        image_path
                    ),
                    "crop_mask": str(
                        mask_path
                    ),
                }
            )

        if scene_positive:

            positive_scenes += 1

        if scene_negative:

            negative_scenes += 1

        if (
            index == 1
            or index % 25 == 0
            or index == len(df)
        ):

            print(
                f"[{index:3d}/{len(df)}] "
                f"{global_id:20s} "
                f"candidates="
                f"{len(components):3d}"
            )

    # ========================================================
    # SAVE CSV
    # ========================================================

    result_df = pd.DataFrame(
        records
    )

    csv_path = (
        args.output
        / "train_candidates.csv"
    )

    result_df.to_csv(
        csv_path,
        index=False,
    )

    # ========================================================
    # STATISTICS
    # ========================================================

    if total_candidates:

        positive_fraction = (
            positive_candidates
            / total_candidates
        )

        negative_fraction = (
            negative_candidates
            / total_candidates
        )

    else:

        positive_fraction = 0.0
        negative_fraction = 0.0

    stats = {
        "experiment": "oil-seg-v3",
        "stage": "candidate_generation",
        "source_model": "oil-seg-v1",
        "checkpoint": str(
            args.checkpoint
        ),
        "manifest": str(
            args.manifest
        ),
        "training_scenes": int(
            len(df)
        ),
        "threshold": float(
            args.threshold
        ),
        "minimum_area": int(
            args.min_area
        ),
        "positive_iou_threshold": float(
            args.match_iou
        ),
        "crop_size": int(
            args.crop_size
        ),
        "padding": int(
            args.padding
        ),
        "total_candidates": int(
            total_candidates
        ),
        "positive_candidates": int(
            positive_candidates
        ),
        "hard_negative_candidates": int(
            negative_candidates
        ),
        "positive_fraction": float(
            positive_fraction
        ),
        "hard_negative_fraction": float(
            negative_fraction
        ),
        "positive_scenes": int(
            positive_scenes
        ),
        "negative_scenes": int(
            negative_scenes
        ),
        "oil_scenes": int(
            oil_scenes
        ),
        "oil_scenes_with_candidate": int(
            oil_scenes_with_candidate
        ),
    }

    stats_path = (
        args.output
        / "train_candidate_stats.json"
    )

    with open(
        stats_path,
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            stats,
            f,
            indent=2,
        )

    # ========================================================
    # FINAL REPORT
    # ========================================================

    print()
    print("=" * 70)
    print(
        "V3 CANDIDATE GENERATION COMPLETE"
    )
    print("=" * 70)

    print(
        "Training scenes:",
        len(df),
    )

    print(
        "Total candidates:",
        total_candidates,
    )

    print(
        "Positive candidates:",
        positive_candidates,
    )

    print(
        "Hard-negative candidates:",
        negative_candidates,
    )

    print(
        "Positive fraction:",
        f"{positive_fraction:.4f}",
    )

    print(
        "Hard-negative fraction:",
        f"{negative_fraction:.4f}",
    )

    print(
        "Positive scenes:",
        positive_scenes,
    )

    print(
        "Oil scenes:",
        oil_scenes,
    )

    print(
        "Oil scenes with candidates:",
        oil_scenes_with_candidate,
    )

    print()
    print(
        "CSV:",
        csv_path,
    )

    print(
        "Statistics:",
        stats_path,
    )

    print()
    print(
        "Positive patches:",
        positive_dir,
    )

    print(
        "Hard-negative patches:",
        negative_dir,
    )

    print()
    print(
        "Only the 630 training scenes were used."
    )

    print(
        "Validation/test sets were not accessed."
    )

    print(
        "V1 checkpoint was not modified."
    )


if __name__ == "__main__":
    main()