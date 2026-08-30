from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
from scipy import ndimage
from tqdm import tqdm


# ============================================================
# PATH SETUP
# ============================================================

SCRIPT_DIR = Path(__file__).resolve().parent
ML_ROOT = SCRIPT_DIR.parent

if str(ML_ROOT) not in sys.path:
    sys.path.insert(0, str(ML_ROOT))


from src.datasets.oil_dataset import OilSegmentationDataset
from src.models.segmentation_model import build_model


# ============================================================
# UTILITIES
# ============================================================

def load_checkpoint(model, checkpoint_path, device):
    """
    Load a trusted training checkpoint.

    PyTorch >= 2.6 defaults to weights_only=True.
    Our PS26143 checkpoints contain normal training metadata,
    so explicitly use weights_only=False for our own trusted
    checkpoint files.
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
            # Handle a raw state_dict checkpoint.
            state_dict = checkpoint

    else:
        raise RuntimeError(
            f"Unsupported checkpoint format: "
            f"{type(checkpoint)}"
        )

    model.load_state_dict(
        state_dict,
        strict=True,
    )

    return checkpoint


def get_checkpoint_epoch(checkpoint):
    if not isinstance(checkpoint, dict):
        return None

    for key in (
        "epoch",
        "last_epoch",
    ):
        if key in checkpoint:
            return checkpoint[key]

    return None


def normalize_array(x):
    x = np.asarray(x, dtype=np.float32)

    finite = np.isfinite(x)

    if not finite.any():
        return np.zeros_like(x, dtype=np.float32)

    values = x[finite]

    lo = np.percentile(values, 1)
    hi = np.percentile(values, 99)

    if hi <= lo:
        return np.zeros_like(x, dtype=np.float32)

    x = (x - lo) / (hi - lo)

    return np.clip(
        x,
        0.0,
        1.0,
    )


def load_image(path):
    image = np.load(path).astype(np.float32)

    if image.ndim != 3:
        raise ValueError(
            f"Expected image shape [C,H,W], got "
            f"{image.shape} from {path}"
        )

    if image.shape[0] != 2:
        raise ValueError(
            f"Expected VV+VH (2 channels), got "
            f"{image.shape} from {path}"
        )

    return image


def load_mask(path):
    mask = np.load(path).astype(np.float32)

    if mask.ndim == 3:
        mask = np.squeeze(mask)

    if mask.ndim != 2:
        raise ValueError(
            f"Expected mask shape [H,W], got "
            f"{mask.shape} from {path}"
        )

    return mask > 0.5


# ============================================================
# SPATIAL PROPOSAL GENERATION
# ============================================================

def threshold_probability_map(
    probability,
    threshold,
):
    return probability >= threshold


def remove_small_components(
    binary,
    min_area,
):
    """
    Remove connected components smaller than min_area.
    """
    binary = binary.astype(np.uint8)

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
        binary,
        connectivity=8,
    )

    output = np.zeros_like(binary)

    for label_id in range(1, num_labels):

        area = stats[label_id, cv2.CC_STAT_AREA]

        if area >= min_area:
            output[labels == label_id] = 1

    return output.astype(bool)


def dilate_mask(
    mask,
    radius,
):
    if radius <= 0:
        return mask.astype(bool)

    size = 2 * radius + 1

    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (size, size),
    )

    result = cv2.dilate(
        mask.astype(np.uint8),
        kernel,
        iterations=1,
    )

    return result.astype(bool)


def merge_nearby_components(
    binary,
    merge_radius,
):
    """
    Merge components whose dilated regions touch.

    This is the key V3.2 spatial recovery mechanism.

    V3.1 mostly changed proposal thresholds.
    V3.2 additionally allows fragmented pieces of the
    same oil slick to become one candidate.
    """
    if merge_radius <= 0:
        return binary.astype(bool)

    dilated = dilate_mask(
        binary,
        merge_radius,
    )

    num_labels, labels = cv2.connectedComponents(
        dilated.astype(np.uint8),
        connectivity=8,
    )

    if num_labels <= 1:
        return binary.astype(bool)

    output = np.zeros_like(binary, dtype=bool)

    for label_id in range(1, num_labels):

        region = labels == label_id

        # Recover original pixels belonging to this merged region.
        output |= (
            binary
            & region
        )

    return output


def fill_small_holes(
    binary,
    max_hole_area,
):
    """
    Fill small holes inside candidate components.
    """
    if max_hole_area <= 0:
        return binary.astype(bool)

    binary_u8 = binary.astype(np.uint8)

    inverse = 1 - binary_u8

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
        inverse,
        connectivity=8,
    )

    output = binary_u8.copy()

    h, w = binary.shape

    for label_id in range(1, num_labels):

        area = stats[label_id, cv2.CC_STAT_AREA]

        ys, xs = np.where(
            labels == label_id
        )

        if len(xs) == 0:
            continue

        touches_border = (
            xs.min() == 0
            or ys.min() == 0
            or xs.max() == w - 1
            or ys.max() == h - 1
        )

        if (
            not touches_border
            and area <= max_hole_area
        ):
            output[labels == label_id] = 1

    return output.astype(bool)


# ============================================================
# COMPONENT EXTRACTION
# ============================================================

def extract_components(
    probability,
    proposal_masks,
    min_area,
):
    """
    Extract connected components from multiple proposal masks.

    Duplicate components across thresholds are later merged
    using IoU / containment.
    """
    candidates = []

    h, w = probability.shape

    for threshold, binary in proposal_masks:

        binary = remove_small_components(
            binary,
            min_area,
        )

        num_labels, labels, stats, centroids = (
            cv2.connectedComponentsWithStats(
                binary.astype(np.uint8),
                connectivity=8,
            )
        )

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

            component_mask = (
                labels == component_id
            )

            values = probability[
                component_mask
            ]

            if values.size == 0:
                continue

            centroid_x = float(
                centroids[
                    component_id,
                    0,
                ]
            )

            centroid_y = float(
                centroids[
                    component_id,
                    1,
                ]
            )

            candidates.append(
                {
                    "threshold": float(threshold),
                    "mask": component_mask,
                    "area": area,
                    "x": x,
                    "y": y,
                    "width": width,
                    "height": height,
                    "centroid_x": centroid_x,
                    "centroid_y": centroid_y,
                    "mean_probability": float(
                        values.mean()
                    ),
                    "p95_probability": float(
                        np.percentile(
                            values,
                            95,
                        )
                    ),
                    "max_probability": float(
                        values.max()
                    ),
                }
            )

    return candidates


# ============================================================
# DUPLICATE / OVERLAP MERGING
# ============================================================

def mask_iou(mask_a, mask_b):
    intersection = np.logical_and(
        mask_a,
        mask_b,
    ).sum()

    union = np.logical_or(
        mask_a,
        mask_b,
    ).sum()

    if union == 0:
        return 0.0

    return float(
        intersection / union
    )


def containment(mask_a, mask_b):
    """
    Return overlap relative to the smaller component.
    """
    intersection = np.logical_and(
        mask_a,
        mask_b,
    ).sum()

    smaller = min(
        mask_a.sum(),
        mask_b.sum(),
    )

    if smaller == 0:
        return 0.0

    return float(
        intersection / smaller
    )


def merge_candidates(
    candidates,
    duplicate_iou,
):
    """
    Merge duplicate / highly overlapping proposals.

    The largest spatial support is retained while confidence
    statistics are recalculated over the merged mask.
    """
    if not candidates:
        return []

    ordered = sorted(
        candidates,
        key=lambda c: (
            c["area"],
            c["mean_probability"],
        ),
        reverse=True,
    )

    merged = []

    for candidate in ordered:

        assigned = False

        for existing in merged:

            iou = mask_iou(
                candidate["mask"],
                existing["mask"],
            )

            cont = containment(
                candidate["mask"],
                existing["mask"],
            )

            if (
                iou >= duplicate_iou
                or cont >= duplicate_iou
            ):
                existing["mask"] = (
                    existing["mask"]
                    | candidate["mask"]
                )

                existing["area"] = int(
                    existing["mask"].sum()
                )

                assigned = True
                break

        if not assigned:
            merged.append(
                candidate.copy()
            )

    return merged


# ============================================================
# GT MATCHING
# ============================================================

def calculate_gt_iou(
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


def match_candidate_to_gt(
    candidate_mask,
    gt_mask,
):
    return calculate_gt_iou(
        candidate_mask,
        gt_mask,
    )


# ============================================================
# CROP EXTRACTION
# ============================================================

def extract_crop(
    image,
    candidate_mask,
    padding,
    crop_size,
):
    ys, xs = np.where(
        candidate_mask
    )

    if len(xs) == 0:
        return None, None, None

    h, w = candidate_mask.shape

    x1 = max(
        0,
        int(xs.min()) - padding,
    )

    y1 = max(
        0,
        int(ys.min()) - padding,
    )

    x2 = min(
        w,
        int(xs.max()) + padding + 1,
    )

    y2 = min(
        h,
        int(ys.max()) + padding + 1,
    )

    crop_image = image[
        :,
        y1:y2,
        x1:x2,
    ]

    crop_mask = candidate_mask[
        y1:y2,
        x1:x2,
    ]

    # Resize spatial dimensions while keeping VV/VH channels.
    crop_image = np.stack(
        [
            cv2.resize(
                crop_image[c],
                (
                    crop_size,
                    crop_size,
                ),
                interpolation=cv2.INTER_LINEAR,
            )
            for c in range(
                crop_image.shape[0]
            )
        ],
        axis=0,
    )

    crop_mask = cv2.resize(
        crop_mask.astype(np.uint8),
        (
            crop_size,
            crop_size,
        ),
        interpolation=cv2.INTER_NEAREST,
    )

    return (
        crop_image.astype(np.float32),
        crop_mask.astype(np.uint8),
        (
            x1,
            y1,
            x2,
            y2,
        ),
    )


# ============================================================
# MAIN
# ============================================================

def main():

    parser = argparse.ArgumentParser(
        description=(
            "PS26143 V3.2 spatial candidate generator"
        )
    )

    parser.add_argument(
        "--checkpoint",
        required=True,
    )

    parser.add_argument(
        "--manifest",
        required=True,
    )

    parser.add_argument(
        "--output",
        required=True,
    )

    parser.add_argument(
        "--thresholds",
        nargs="+",
        type=float,
        default=[
            0.20,
            0.30,
            0.40,
            0.50,
        ],
    )

    parser.add_argument(
        "--min-area",
        type=int,
        default=20,
    )

    parser.add_argument(
        "--merge-radius",
        type=int,
        default=12,
    )

    parser.add_argument(
        "--duplicate-iou",
        type=float,
        default=0.70,
    )

    parser.add_argument(
        "--match-iou",
        type=float,
        default=0.30,
    )

    parser.add_argument(
        "--hole-area",
        type=int,
        default=100,
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

    parser.add_argument(
        "--batch-size",
        type=int,
        default=16,
    )

    args = parser.parse_args()

    output_dir = Path(
        args.output
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    patch_dir = output_dir / "patches"

    patch_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    positive_dir = (
        patch_dir / "positive"
    )

    negative_dir = (
        patch_dir / "hard_negative"
    )

    positive_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    negative_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    # --------------------------------------------------------
    # DEVICE
    # --------------------------------------------------------

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print("=" * 70)
    print(
        "PS26143 — V3.2 SPATIAL CANDIDATE GENERATION"
    )
    print("=" * 70)

    print()
    print("Device:", device)

    if device.type == "cuda":
        print(
            "GPU:",
            torch.cuda.get_device_name(0),
        )

    # --------------------------------------------------------
    # MANIFEST
    # --------------------------------------------------------

    manifest_path = Path(
        args.manifest
    )

    if not manifest_path.exists():
        raise FileNotFoundError(
            f"Manifest not found:\n{manifest_path}"
        )

    df = pd.read_csv(
        manifest_path
    )

    required_columns = {
        "global_id",
        "dataset",
        "image",
        "mask",
    }

    missing = (
        required_columns
        - set(df.columns)
    )

    if missing:
        raise ValueError(
            "Manifest missing required columns: "
            f"{sorted(missing)}"
        )

    print()
    print(
        "Manifest:",
        manifest_path,
    )

    print(
        "Scenes:",
        len(df),
    )

    print(
        "Dataset distribution:"
    )
    print(
        df["dataset"].value_counts()
    )

    # --------------------------------------------------------
    # SAFETY
    # --------------------------------------------------------

    if set(
        df["dataset"].astype(str)
    ) - {
        "oil",
        "lookalike",
        "no_oil",
    }:
        raise ValueError(
            "Unexpected dataset labels found."
        )

    # This generator is intended for TRAIN ONLY.
    if len(df) > 0 and "split" in df.columns:
        if not (
            df["split"]
            .astype(str)
            .str.lower()
            .eq("train")
            .all()
        ):
            raise RuntimeError(
                "V3.2 candidate generation must use "
                "TRAIN split only."
            )

    # --------------------------------------------------------
    # MODEL
    # --------------------------------------------------------

    print()
    print("=" * 70)
    print("LOADING V1 FROZEN MODEL")
    print("=" * 70)

    model = build_model(
        architecture="unet",
        encoder="resnet34",
        encoder_weights=None,
        in_channels=2,
        classes=1,
    )

    checkpoint = load_checkpoint(
        model,
        args.checkpoint,
        device,
    )

    model.to(device)
    model.eval()

    print(
        "Checkpoint:",
        args.checkpoint,
    )

    epoch = get_checkpoint_epoch(
        checkpoint
    )

    if epoch is not None:
        print(
            "Checkpoint epoch:",
            epoch,
        )

    # --------------------------------------------------------
    # DATASET
    # --------------------------------------------------------

    print()
    print("=" * 70)
    print("GENERATING V3.2 PROPOSALS")
    print("=" * 70)

    print(
        "Thresholds:",
        args.thresholds,
    )

    print(
        "Minimum area:",
        args.min_area,
    )

    print(
        "Merge radius:",
        args.merge_radius,
    )

    print(
        "Duplicate IoU:",
        args.duplicate_iou,
    )

    print(
        "Match IoU:",
        args.match_iou,
    )

    rows = []

    total_candidates = 0
    positive_candidates = 0
    hard_negative_candidates = 0

    recovered_oil_scenes = set()

    # --------------------------------------------------------
    # SCENE LOOP
    # --------------------------------------------------------

    for _, row in tqdm(
        df.iterrows(),
        total=len(df),
        desc="Scenes",
    ):

        global_id = str(
            row["global_id"]
        )

        dataset_name = str(
            row["dataset"]
        )

        image_path = Path(
            row["image"]
        )

        mask_path = Path(
            row["mask"]
        )

        if not image_path.exists():
            raise FileNotFoundError(
                f"Image missing:\n{image_path}"
            )

        if not mask_path.exists():
            raise FileNotFoundError(
                f"Mask missing:\n{mask_path}"
            )

        image = load_image(
            image_path
        )

        gt_mask = load_mask(
            mask_path
        )

        if image.shape[1:] != gt_mask.shape:
            raise ValueError(
                f"Image/mask shape mismatch "
                f"for {global_id}: "
                f"{image.shape} vs "
                f"{gt_mask.shape}"
            )

        # ----------------------------------------------------
        # MODEL INFERENCE
        # ----------------------------------------------------

        tensor = torch.from_numpy(
            image
        ).unsqueeze(0).to(
            device,
            non_blocking=True,
        )

        with torch.no_grad():

            with torch.autocast(
                device_type="cuda",
                enabled=device.type == "cuda",
            ):
                logits = model(
                    tensor
                )

            probability = (
                torch.sigmoid(
                    logits.float()
                )
                .squeeze()
                .detach()
                .cpu()
                .numpy()
            )

        probability = np.asarray(
            probability,
            dtype=np.float32,
        )

        # ----------------------------------------------------
        # MULTI-THRESHOLD PROPOSALS
        # ----------------------------------------------------

        proposal_masks = []

        for threshold in args.thresholds:

            binary = (
                probability >= threshold
            )

            binary = remove_small_components(
                binary,
                args.min_area,
            )

            # Fill small holes so fragmented slicks
            # retain coherent spatial support.
            binary = fill_small_holes(
                binary,
                args.hole_area,
            )

            # V3.2 spatial merging.
            binary = merge_nearby_components(
                binary,
                args.merge_radius,
            )

            proposal_masks.append(
                (
                    threshold,
                    binary,
                )
            )

        candidates = extract_components(
            probability,
            proposal_masks,
            args.min_area,
        )

        candidates = merge_candidates(
            candidates,
            args.duplicate_iou,
        )

        # ----------------------------------------------------
        # CANDIDATE PROCESSING
        # ----------------------------------------------------

        scene_candidate_index = 0

        for candidate in candidates:

            candidate_mask = candidate[
                "mask"
            ]

            area = int(
                candidate_mask.sum()
            )

            if area < args.min_area:
                continue

            gt_iou = match_candidate_to_gt(
                candidate_mask,
                gt_mask,
            )

            label = (
                "positive"
                if gt_iou >= args.match_iou
                else "hard_negative"
            )

            if label == "positive":
                positive_candidates += 1

                if dataset_name == "oil":
                    recovered_oil_scenes.add(
                        global_id
                    )

            else:
                hard_negative_candidates += 1

            candidate_id = (
                f"{global_id}_candidate_"
                f"{scene_candidate_index:03d}"
            )

            crop_result = extract_crop(
                image,
                candidate_mask,
                args.padding,
                args.crop_size,
            )

            if crop_result[0] is None:
                continue

            crop_image, crop_mask, bbox = (
                crop_result
            )

            crop_image_path = (
                positive_dir
                if label == "positive"
                else negative_dir
            ) / (
                f"{candidate_id}_image.npy"
            )

            crop_mask_path = (
                positive_dir
                if label == "positive"
                else negative_dir
            ) / (
                f"{candidate_id}_mask.npy"
            )

            np.save(
                crop_image_path,
                crop_image,
            )

            np.save(
                crop_mask_path,
                crop_mask,
            )

            total_candidates += 1

            rows.append(
                {
                    "global_id": global_id,
                    "dataset": dataset_name,
                    "candidate_id": candidate_id,
                    "crop_image": str(
                        crop_image_path
                    ),
                    "crop_mask": str(
                        crop_mask_path
                    ),
                    "x": int(
                        candidate["x"]
                    ),
                    "y": int(
                        candidate["y"]
                    ),
                    "width": int(
                        candidate["width"]
                    ),
                    "height": int(
                        candidate["height"]
                    ),
                    "area": area,
                    "centroid_x": float(
                        candidate[
                            "centroid_x"
                        ]
                    ),
                    "centroid_y": float(
                        candidate[
                            "centroid_y"
                        ]
                    ),
                    "mean_probability": float(
                        candidate[
                            "mean_probability"
                        ]
                    ),
                    "p95_probability": float(
                        candidate[
                            "p95_probability"
                        ]
                    ),
                    "max_probability": float(
                        candidate[
                            "max_probability"
                        ]
                    ),
                    "proposal_threshold": float(
                        candidate[
                            "threshold"
                        ]
                    ),
                    "gt_iou": float(
                        gt_iou
                    ),
                    "label": label,
                    "bbox_x1": int(
                        bbox[0]
                    ),
                    "bbox_y1": int(
                        bbox[1]
                    ),
                    "bbox_x2": int(
                        bbox[2]
                    ),
                    "bbox_y2": int(
                        bbox[3]
                    ),
                }
            )

            scene_candidate_index += 1

    # --------------------------------------------------------
    # SAVE CSV
    # --------------------------------------------------------

    output_df = pd.DataFrame(
        rows
    )

    csv_path = (
        output_dir
        / "train_candidates.csv"
    )

    output_df.to_csv(
        csv_path,
        index=False,
    )

    # --------------------------------------------------------
    # STATISTICS
    # --------------------------------------------------------

    stats = {
        "experiment": "oil-seg-v3.2",
        "source": "train_only",
        "num_scenes": int(len(df)),
        "num_candidates": int(
            len(output_df)
        ),
        "positive_candidates": int(
            positive_candidates
        ),
        "hard_negative_candidates": int(
            hard_negative_candidates
        ),
        "positive_scene_count": int(
            len(recovered_oil_scenes)
        ),
        "thresholds": [
            float(x)
            for x in args.thresholds
        ],
        "min_area": int(
            args.min_area
        ),
        "merge_radius": int(
            args.merge_radius
        ),
        "duplicate_iou": float(
            args.duplicate_iou
        ),
        "match_iou": float(
            args.match_iou
        ),
        "hole_area": int(
            args.hole_area
        ),
        "crop_size": int(
            args.crop_size
        ),
        "padding": int(
            args.padding
        ),
        "checkpoint": str(
            args.checkpoint
        ),
    }

    if len(output_df):

        stats["dataset_distribution"] = (
            output_df[
                "dataset"
            ]
            .value_counts()
            .to_dict()
        )

        stats["label_distribution"] = (
            output_df[
                "label"
            ]
            .value_counts()
            .to_dict()
        )

        stats["positive_gt_iou_mean"] = (
            float(
                output_df.loc[
                    output_df["label"]
                    == "positive",
                    "gt_iou",
                ].mean()
            )
            if (
                output_df["label"]
                == "positive"
            ).any()
            else 0.0
        )

        stats["positive_gt_iou_median"] = (
            float(
                output_df.loc[
                    output_df["label"]
                    == "positive",
                    "gt_iou",
                ].median()
            )
            if (
                output_df["label"]
                == "positive"
            ).any()
            else 0.0
        )

    stats_path = (
        output_dir
        / "train_candidate_stats.json"
    )

    with open(
        stats_path,
        "w",
    ) as f:
        json.dump(
            stats,
            f,
            indent=2,
        )

    # --------------------------------------------------------
    # FINAL OUTPUT
    # --------------------------------------------------------

    print()
    print("=" * 70)
    print("V3.2 CANDIDATE GENERATION COMPLETE")
    print("=" * 70)

    print(
        "Scenes:",
        len(df),
    )

    print(
        "Candidates:",
        len(output_df),
    )

    print(
        "Positive:",
        positive_candidates,
    )

    print(
        "Hard negative:",
        hard_negative_candidates,
    )

    print(
        "Oil scenes with positive candidates:",
        len(recovered_oil_scenes),
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

    print("=" * 70)


if __name__ == "__main__":
    main()