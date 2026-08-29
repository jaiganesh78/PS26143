from __future__ import annotations

import json
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image, ImageDraw
from torch.utils.data import DataLoader

from src.datasets.oil_dataset import OilSegmentationDataset
from src.models.segmentation_model import build_model


# ============================================================
# PS26143 — CONSOLIDATED TEST ERROR ANALYSIS
# ============================================================
#
# PURPOSE
# -------
# Analyze the already-frozen 135-sample test set using the
# already-trained BEST checkpoint.
#
# THIS SCRIPT DOES NOT:
#   - train
#   - modify the model
#   - modify the test set
#   - tune the threshold
#   - create a checkpoint
#
# THIS SCRIPT DOES:
#   - regenerate test predictions
#   - calculate per-scene metrics
#   - identify oil segmentation failures
#   - identify lookalike false positives
#   - identify no-oil false positives
#   - measure boundary disagreement
#   - generate compact diagnostic galleries
#   - save machine-readable analysis artifacts
# ============================================================


SEED = 42

DRIVE_ROOT = Path(
    "/content/drive/MyDrive/PS26143"
)

PROCESSED_ROOT = (
    DRIVE_ROOT / "data/processed"
)

TEST_MANIFEST = (
    PROCESSED_ROOT / "test/manifest.csv"
)

CHECKPOINT = (
    DRIVE_ROOT /
    "checkpoints/oil_seg_v1_best.pt"
)

OUTPUT_DIR = (
    DRIVE_ROOT /
    "evaluation/oil_seg_v1/error_analysis"
)

RESULTS_CSV = (
    OUTPUT_DIR /
    "error_analysis.csv"
)

SUMMARY_JSON = (
    OUTPUT_DIR /
    "error_analysis_summary.json"
)

GALLERY_DIR = (
    OUTPUT_DIR /
    "galleries"
)

BATCH_SIZE = 16
NUM_WORKERS = 0

# IMPORTANT:
# This remains the frozen baseline threshold.
# We are NOT tuning it during this analysis.
THRESHOLD = 0.5

GALLERY_COUNT = 12


# ============================================================
# REPRODUCIBILITY
# ============================================================

def set_seed(seed: int = SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ============================================================
# CHECKPOINT LOADING
# ============================================================

def load_checkpoint(
    model,
    checkpoint_path: Path,
    device,
):
    checkpoint = torch.load(
        checkpoint_path,
        map_location=device,
        weights_only=False,
    )

    if not isinstance(checkpoint, dict):
        raise RuntimeError(
            "Unsupported checkpoint format."
        )

    if "model_state_dict" in checkpoint:
        state_dict = checkpoint["model_state_dict"]

    elif "state_dict" in checkpoint:
        state_dict = checkpoint["state_dict"]

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

    return checkpoint


# ============================================================
# BASIC METRICS
# ============================================================

def calculate_metrics(
    prediction,
    target,
):
    prediction = prediction.astype(bool)
    target = target.astype(bool)

    tp = np.logical_and(
        prediction,
        target,
    ).sum()

    fp = np.logical_and(
        prediction,
        ~target,
    ).sum()

    fn = np.logical_and(
        ~prediction,
        target,
    ).sum()

    tn = np.logical_and(
        ~prediction,
        ~target,
    ).sum()

    dice = (
        2.0 * tp /
        max(
            2.0 * tp + fp + fn,
            1,
        )
    )

    iou = (
        tp /
        max(
            tp + fp + fn,
            1,
        )
    )

    precision = (
        tp /
        max(tp + fp, 1)
    )

    recall = (
        tp /
        max(tp + fn, 1)
    )

    target_area = int(
        target.sum()
    )

    prediction_area = int(
        prediction.sum()
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
        "target_area": target_area,
        "prediction_area": prediction_area,
        "fp_area": int(fp),
        "fn_area": int(fn),
    }


# ============================================================
# BOUNDARY ANALYSIS
# ============================================================

def binary_boundary(mask):
    """
    Approximate boundary pixels using 4-neighbour erosion.
    """

    mask = mask.astype(bool)

    if not mask.any():
        return np.zeros_like(
            mask,
            dtype=bool,
        )

    padded = np.pad(
        mask,
        1,
        mode="constant",
        constant_values=False,
    )

    center = padded[1:-1, 1:-1]

    up = padded[:-2, 1:-1]
    down = padded[2:, 1:-1]
    left = padded[1:-1, :-2]
    right = padded[1:-1, 2:]

    erosion = (
        center
        & up
        & down
        & left
        & right
    )

    return center & ~erosion


def boundary_error(
    prediction,
    target,
):
    """
    Boundary disagreement indicator.

    This is intentionally NOT presented as a formal
    Hausdorff distance.
    """

    pred_boundary = binary_boundary(
        prediction
    )

    target_boundary = binary_boundary(
        target
    )

    pred_boundary_count = int(
        pred_boundary.sum()
    )

    target_boundary_count = int(
        target_boundary.sum()
    )

    if (
        pred_boundary_count == 0
        and target_boundary_count == 0
    ):
        return {
            "boundary_error": 0.0,
            "boundary_disagreement": 0.0,
            "pred_boundary_pixels": 0,
            "target_boundary_pixels": 0,
        }

    padded_target = np.pad(
        target_boundary,
        1,
        mode="constant",
        constant_values=False,
    )

    target_neighbourhood = (
        padded_target[:-2, :-2]
        | padded_target[:-2, 1:-1]
        | padded_target[:-2, 2:]
        | padded_target[1:-1, :-2]
        | padded_target[1:-1, 1:-1]
        | padded_target[1:-1, 2:]
        | padded_target[2:, :-2]
        | padded_target[2:, 1:-1]
        | padded_target[2:, 2:]
    )

    padded_pred = np.pad(
        pred_boundary,
        1,
        mode="constant",
        constant_values=False,
    )

    pred_neighbourhood = (
        padded_pred[:-2, :-2]
        | padded_pred[:-2, 1:-1]
        | padded_pred[:-2, 2:]
        | padded_pred[1:-1, :-2]
        | padded_pred[1:-1, 1:-1]
        | padded_pred[1:-1, 2:]
        | padded_pred[2:, :-2]
        | padded_pred[2:, 1:-1]
        | padded_pred[2:, 2:]
    )

    pred_match = (
        pred_boundary
        & target_neighbourhood
    )

    target_match = (
        target_boundary
        & pred_neighbourhood
    )

    pred_boundary_accuracy = (
        pred_match.sum()
        / max(pred_boundary_count, 1)
    )

    target_boundary_accuracy = (
        target_match.sum()
        / max(target_boundary_count, 1)
    )

    disagreement = 1.0 - (
        0.5 *
        (
            pred_boundary_accuracy
            + target_boundary_accuracy
        )
    )

    return {
        "boundary_error": float(disagreement),
        "boundary_disagreement": float(disagreement),
        "pred_boundary_pixels": pred_boundary_count,
        "target_boundary_pixels": target_boundary_count,
    }


# ============================================================
# DISPLAY HELPERS
# ============================================================

def normalize_display_band(band):

    band = band.astype(np.float32)

    finite = np.isfinite(band)

    if not finite.any():
        return np.zeros(
            band.shape,
            dtype=np.uint8,
        )

    values = band[finite]

    low = np.percentile(
        values,
        2,
    )

    high = np.percentile(
        values,
        98,
    )

    if high <= low:
        return np.zeros(
            band.shape,
            dtype=np.uint8,
        )

    band = np.nan_to_num(
        band,
        nan=low,
        posinf=high,
        neginf=low,
    )

    band = np.clip(
        band,
        low,
        high,
    )

    band = (
        (band - low)
        / (high - low)
    )

    return (
        band * 255
    ).astype(np.uint8)


def make_base_image(image):

    vv = normalize_display_band(
        image[0]
    )

    vh = normalize_display_band(
        image[1]
    )

    mean = (
        (
            vv.astype(np.float32)
            + vh.astype(np.float32)
        )
        / 2.0
    ).astype(np.uint8)

    return np.stack(
        [vv, vh, mean],
        axis=-1,
    )


def overlay_mask(
    base,
    mask,
    mode,
):
    """
    mode:
        target     = red
        prediction = green
        fp         = blue
        fn         = yellow
    """

    out = base.copy()

    mask = mask.astype(bool)

    if mode == "target":

        out[mask] = (
            0.55 * out[mask]
            + 0.45 * np.array(
                [255, 0, 0],
                dtype=np.float32,
            )
        ).astype(np.uint8)

    elif mode == "prediction":

        out[mask] = (
            0.55 * out[mask]
            + 0.45 * np.array(
                [0, 255, 0],
                dtype=np.float32,
            )
        ).astype(np.uint8)

    elif mode == "fp":

        out[mask] = (
            0.45 * out[mask]
            + 0.55 * np.array(
                [0, 80, 255],
                dtype=np.float32,
            )
        ).astype(np.uint8)

    elif mode == "fn":

        out[mask] = (
            0.45 * out[mask]
            + 0.55 * np.array(
                [255, 255, 0],
                dtype=np.float32,
            )
        ).astype(np.uint8)

    return out


# ============================================================
# GALLERY
# ============================================================

def make_panel(
    image,
    title,
    width=512,
    height=560,
):

    image = Image.fromarray(
        image
    ).resize(
        (width, width)
    )

    canvas = Image.new(
        "RGB",
        (width, height),
        "black",
    )

    canvas.paste(
        image,
        (0, 48),
    )

    draw = ImageDraw.Draw(
        canvas
    )

    draw.text(
        (10, 15),
        title,
        fill="white",
    )

    return canvas


def save_case_gallery(
    case,
    output_path,
):
    """
    Robustly save one diagnostic case.

    IMPORTANT:
    Metadata lives inside case["record"].
    This function therefore reads global_id/dataset from
    the record instead of assuming they exist at the
    top level of case.
    """

    record = case.get(
        "record",
        {}
    )

    global_id = str(
        record.get(
            "global_id",
            "unknown",
        )
    )

    dataset_name = str(
        record.get(
            "dataset",
            "unknown",
        )
    )

    base = make_base_image(
        case["image"]
    )

    target = overlay_mask(
        base,
        case["target"],
        "target",
    )

    prediction = overlay_mask(
        base,
        case["prediction"],
        "prediction",
    )

    fp = (
        case["prediction"]
        & ~case["target"]
    )

    fn = (
        case["target"]
        & ~case["prediction"]
    )

    errors = base.copy()

    errors = overlay_mask(
        errors,
        fp,
        "fp",
    )

    errors = overlay_mask(
        errors,
        fn,
        "fn",
    )

    title_base = (
        f"{global_id} | "
        f"{dataset_name}"
    )

    panels = [
        make_panel(
            base,
            "SAR VV/VH",
        ),
        make_panel(
            target,
            "Ground Truth",
        ),
        make_panel(
            prediction,
            "Prediction",
        ),
        make_panel(
            errors,
            "Errors: blue=FP yellow=FN",
        ),
    ]

    total_width = sum(
        p.width for p in panels
    )

    canvas = Image.new(
        "RGB",
        (
            total_width,
            panels[0].height + 40,
        ),
        "black",
    )

    draw = ImageDraw.Draw(
        canvas
    )

    draw.text(
        (10, 10),
        title_base,
        fill="white",
    )

    x = 0

    for panel in panels:

        canvas.paste(
            panel,
            (x, 40),
        )

        x += panel.width

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    canvas.save(
        output_path,
        quality=95,
    )


# ============================================================
# CASE SELECTION
# ============================================================

def choose_gallery_cases(
    results_df,
):

    galleries = {}

    oil = results_df[
        results_df["dataset"] == "oil"
    ].copy()

    lookalike = results_df[
        results_df["dataset"] == "lookalike"
    ].copy()

    no_oil = results_df[
        results_df["dataset"] == "no_oil"
    ].copy()

    galleries["worst_oil_dice"] = (
        oil.sort_values(
            ["dice", "fn_area"],
            ascending=[True, False],
        )
        .head(GALLERY_COUNT)
        ["global_id"]
        .tolist()
    )

    galleries["largest_oil_fn"] = (
        oil.sort_values(
            "fn_area",
            ascending=False,
        )
        .head(GALLERY_COUNT)
        ["global_id"]
        .tolist()
    )

    galleries["worst_oil_boundary"] = (
        oil.sort_values(
            "boundary_error",
            ascending=False,
        )
        .head(GALLERY_COUNT)
        ["global_id"]
        .tolist()
    )

    galleries["lookalike_false_positives"] = (
        lookalike.sort_values(
            "fp_area",
            ascending=False,
        )
        .head(GALLERY_COUNT)
        ["global_id"]
        .tolist()
    )

    galleries["lookalike_predicted_area"] = (
        lookalike.sort_values(
            "prediction_area",
            ascending=False,
        )
        .head(GALLERY_COUNT)
        ["global_id"]
        .tolist()
    )

    galleries["no_oil_false_positives"] = (
        no_oil.sort_values(
            "fp_area",
            ascending=False,
        )
        .head(GALLERY_COUNT)
        ["global_id"]
        .tolist()
    )

    galleries["no_oil_predicted_area"] = (
        no_oil.sort_values(
            "prediction_area",
            ascending=False,
        )
        .head(GALLERY_COUNT)
        ["global_id"]
        .tolist()
    )

    return galleries


# ============================================================
# MAIN
# ============================================================

def main():

    set_seed()

    print("=" * 70)
    print(
        "PS26143 — CONSOLIDATED TEST ERROR ANALYSIS"
    )
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
            "GPU:",
            torch.cuda.get_device_name(0),
        )

    # --------------------------------------------------------
    # Paths
    # --------------------------------------------------------

    print()
    print("Test manifest:")
    print(TEST_MANIFEST)

    print()
    print("Best checkpoint:")
    print(CHECKPOINT)

    if not TEST_MANIFEST.exists():
        raise FileNotFoundError(
            TEST_MANIFEST
        )

    if not CHECKPOINT.exists():
        raise FileNotFoundError(
            CHECKPOINT
        )

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    GALLERY_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    # --------------------------------------------------------
    # Manifest
    # --------------------------------------------------------

    df = pd.read_csv(
        TEST_MANIFEST
    )

    if len(df) != 135:
        raise RuntimeError(
            f"Expected 135 test samples, "
            f"found {len(df)}."
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
            "Test manifest missing columns: "
            f"{sorted(missing_columns)}"
        )

    if df["global_id"].duplicated().any():
        raise RuntimeError(
            "Duplicate global IDs in test manifest."
        )

    print()
    print("Test samples:", len(df))

    print()
    print("Dataset distribution:")
    print(
        df["dataset"]
        .value_counts()
        .sort_index()
    )

    # --------------------------------------------------------
    # Verify paths
    # --------------------------------------------------------

    missing_images = [
        p
        for p in df["image"]
        if not Path(p).exists()
    ]

    missing_masks = [
        p
        for p in df["mask"]
        if not Path(p).exists()
    ]

    if missing_images:
        raise FileNotFoundError(
            "Missing test images: "
            f"{missing_images[:5]}"
        )

    if missing_masks:
        raise FileNotFoundError(
            "Missing test masks: "
            f"{missing_masks[:5]}"
        )

    print()
    print(
        "All test image/mask paths verified."
    )

    # --------------------------------------------------------
    # Dataset / loader
    # --------------------------------------------------------

    records = df.to_dict(
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
    print("LOADING BEST MODEL")
    print("=" * 70)

    model = build_model(
        architecture="unet",
        encoder="resnet34",
        encoder_weights=None,
        in_channels=2,
        classes=1,
    )

    model.to(device)

    checkpoint = load_checkpoint(
        model,
        CHECKPOINT,
        device,
    )

    model.eval()

    print(
        "Checkpoint loaded:",
        CHECKPOINT,
    )

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
    # Prediction pass
    # --------------------------------------------------------

    print()
    print("=" * 70)
    print("GENERATING TEST PREDICTIONS")
    print("=" * 70)

    cases = {}

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
                logits
            )

            predictions = (
                probabilities >= THRESHOLD
            )

            for i in range(
                images.shape[0]
            ):

                image_np = (
                    images[i]
                    .detach()
                    .cpu()
                    .numpy()
                )

                target_np = (
                    masks[i, 0]
                    .detach()
                    .cpu()
                    .numpy()
                    >= 0.5
                )

                probability_np = (
                    probabilities[i, 0]
                    .detach()
                    .cpu()
                    .numpy()
                )

                prediction_np = (
                    predictions[i, 0]
                    .detach()
                    .cpu()
                    .numpy()
                    .astype(bool)
                )

                global_id = str(
                    batch["global_id"][i]
                )

                dataset_name = str(
                    batch["dataset"][i]
                )

                metrics = calculate_metrics(
                    prediction_np,
                    target_np,
                )

                boundary = boundary_error(
                    prediction_np,
                    target_np,
                )

                scene_positive = bool(
                    prediction_np.any()
                )

                target_positive = bool(
                    target_np.any()
                )

                probability_mean_positive = (
                    float(
                        probability_np[
                            prediction_np
                        ].mean()
                    )
                    if prediction_np.any()
                    else 0.0
                )

                probability_max = float(
                    probability_np.max()
                )

                record = {
                    "global_id":
                        global_id,

                    "dataset":
                        dataset_name,

                    "dice":
                        metrics["dice"],

                    "iou":
                        metrics["iou"],

                    "precision":
                        metrics["precision"],

                    "recall":
                        metrics["recall"],

                    "tp":
                        metrics["tp"],

                    "fp":
                        metrics["fp"],

                    "fn":
                        metrics["fn"],

                    "tn":
                        metrics["tn"],

                    "target_area":
                        metrics["target_area"],

                    "prediction_area":
                        metrics[
                            "prediction_area"
                        ],

                    "fp_area":
                        metrics["fp_area"],

                    "fn_area":
                        metrics["fn_area"],

                    "target_fraction":
                        metrics["target_area"]
                        / target_np.size,

                    "prediction_fraction":
                        metrics["prediction_area"]
                        / prediction_np.size,

                    "fp_fraction":
                        metrics["fp_area"]
                        / prediction_np.size,

                    "fn_fraction":
                        metrics["fn_area"]
                        / max(target_np.size, 1),

                    "boundary_error":
                        boundary[
                            "boundary_error"
                        ],

                    "boundary_disagreement":
                        boundary[
                            "boundary_disagreement"
                        ],

                    "pred_boundary_pixels":
                        boundary[
                            "pred_boundary_pixels"
                        ],

                    "target_boundary_pixels":
                        boundary[
                            "target_boundary_pixels"
                        ],

                    "scene_prediction_positive":
                        scene_positive,

                    "scene_target_positive":
                        target_positive,

                    "mean_probability_on_prediction":
                        probability_mean_positive,

                    "max_probability":
                        probability_max,
                }

                cases[global_id] = {
                    "image":
                        image_np,

                    "target":
                        target_np,

                    "prediction":
                        prediction_np,

                    "probability":
                        probability_np,

                    # IMPORTANT:
                    # Metadata is kept inside record.
                    "record":
                        record,
                }

                processed += 1

        print(
            f"Processed: "
            f"{processed}/{len(dataset)}"
        )

    if processed != 135:
        raise RuntimeError(
            f"Expected 135 predictions, "
            f"got {processed}."
        )

    # --------------------------------------------------------
    # Results dataframe
    # --------------------------------------------------------

    results_df = pd.DataFrame(
        [
            cases[g]["record"]
            for g in cases
        ]
    )

    results_df = (
        results_df
        .sort_values(
            [
                "dataset",
                "global_id",
            ]
        )
        .reset_index(drop=True)
    )

    results_df.to_csv(
        RESULTS_CSV,
        index=False,
    )

    # --------------------------------------------------------
    # Aggregate summary
    # --------------------------------------------------------

    summary = {
        "experiment":
            "oil-seg-v1",

        "checkpoint":
            str(CHECKPOINT),

        "test_manifest":
            str(TEST_MANIFEST),

        "test_samples":
            int(len(results_df)),

        "threshold":
            THRESHOLD,

        "device":
            str(device),

        "gpu":
            (
                torch.cuda.get_device_name(0)
                if device.type == "cuda"
                else None
            ),

        "architecture":
            "U-Net",

        "encoder":
            "ResNet34",

        "in_channels":
            2,

        "classes":
            1,

        "image_size":
            512,
    }

    # --------------------------------------------------------
    # Dataset summaries
    # --------------------------------------------------------

    dataset_summary = {}

    for name in [
        "oil",
        "lookalike",
        "no_oil",
    ]:

        subset = results_df[
            results_df["dataset"] == name
        ]

        if len(subset) == 0:
            continue

        dataset_summary[name] = {
            "samples":
                int(len(subset)),

            "mean_dice":
                float(subset["dice"].mean()),

            "median_dice":
                float(subset["dice"].median()),

            "mean_iou":
                float(subset["iou"].mean()),

            "mean_precision":
                float(
                    subset["precision"].mean()
                ),

            "mean_recall":
                float(
                    subset["recall"].mean()
                ),

            "mean_fp_area":
                float(
                    subset["fp_area"].mean()
                ),

            "mean_fn_area":
                float(
                    subset["fn_area"].mean()
                ),

            "mean_boundary_error":
                float(
                    subset[
                        "boundary_error"
                    ].mean()
                ),

            "scene_positive_count":
                int(
                    subset[
                        "scene_prediction_positive"
                    ].sum()
                ),
        }

    summary["by_dataset"] = dataset_summary

    # --------------------------------------------------------
    # Oil analysis
    # --------------------------------------------------------

    oil = results_df[
        results_df["dataset"] == "oil"
    ]

    oil_detected = int(
        oil[
            "scene_prediction_positive"
        ].sum()
    )

    summary["oil_analysis"] = {
        "samples":
            int(len(oil)),

        "scene_detection_count":
            oil_detected,

        "scene_detection_rate":
            float(
                oil_detected
                / max(len(oil), 1)
            ),

        "mean_dice":
            float(oil["dice"].mean()),

        "mean_iou":
            float(oil["iou"].mean()),

        "mean_recall":
            float(oil["recall"].mean()),

        "mean_precision":
            float(
                oil["precision"].mean()
            ),

        "mean_fn_area":
            float(
                oil["fn_area"].mean()
            ),

        "mean_boundary_error":
            float(
                oil[
                    "boundary_error"
                ].mean()
            ),
    }

    # --------------------------------------------------------
    # Lookalike analysis
    # --------------------------------------------------------

    lookalike = results_df[
        results_df["dataset"]
        == "lookalike"
    ]

    lookalike_fp = int(
        lookalike[
            "scene_prediction_positive"
        ].sum()
    )

    summary["lookalike_rejection"] = {
        "samples":
            int(len(lookalike)),

        "false_positive_scenes":
            lookalike_fp,

        "false_alarm_rate":
            float(
                lookalike_fp
                / max(len(lookalike), 1)
            ),

        "mean_predicted_area":
            float(
                lookalike[
                    "prediction_area"
                ].mean()
            ),

        "max_predicted_area":
            int(
                lookalike[
                    "prediction_area"
                ].max()
            ),

        "total_false_positive_pixels":
            int(
                lookalike[
                    "fp_area"
                ].sum()
            ),
    }

    # --------------------------------------------------------
    # No-oil analysis
    # --------------------------------------------------------

    no_oil = results_df[
        results_df["dataset"]
        == "no_oil"
    ]

    no_oil_fp = int(
        no_oil[
            "scene_prediction_positive"
        ].sum()
    )

    summary["no_oil_rejection"] = {
        "samples":
            int(len(no_oil)),

        "false_positive_scenes":
            no_oil_fp,

        "false_alarm_rate":
            float(
                no_oil_fp
                / max(len(no_oil), 1)
            ),

        "mean_predicted_area":
            float(
                no_oil[
                    "prediction_area"
                ].mean()
            ),

        "max_predicted_area":
            int(
                no_oil[
                    "prediction_area"
                ].max()
            ),

        "total_false_positive_pixels":
            int(
                no_oil[
                    "fp_area"
                ].sum()
            ),
    }

    # --------------------------------------------------------
    # Worst cases
    # --------------------------------------------------------

    summary["worst_cases"] = {
        "oil_lowest_dice":
            oil.sort_values(
                "dice"
            ).head(10)[
                [
                    "global_id",
                    "dice",
                    "iou",
                    "recall",
                    "precision",
                    "fn_area",
                    "boundary_error",
                ]
            ].to_dict(
                orient="records"
            ),

        "oil_largest_fn":
            oil.sort_values(
                "fn_area",
                ascending=False,
            ).head(10)[
                [
                    "global_id",
                    "dice",
                    "iou",
                    "recall",
                    "fn_area",
                    "boundary_error",
                ]
            ].to_dict(
                orient="records"
            ),

        "lookalike_largest_fp":
            lookalike.sort_values(
                "fp_area",
                ascending=False,
            ).head(10)[
                [
                    "global_id",
                    "fp_area",
                    "prediction_area",
                    "max_probability",
                ]
            ].to_dict(
                orient="records"
            ),

        "no_oil_largest_fp":
            no_oil.sort_values(
                "fp_area",
                ascending=False,
            ).head(10)[
                [
                    "global_id",
                    "fp_area",
                    "prediction_area",
                    "max_probability",
                ]
            ].to_dict(
                orient="records"
            ),

        "oil_worst_boundary":
            oil.sort_values(
                "boundary_error",
                ascending=False,
            ).head(10)[
                [
                    "global_id",
                    "dice",
                    "iou",
                    "boundary_error",
                    "fn_area",
                ]
            ].to_dict(
                orient="records"
            ),
    }

    # --------------------------------------------------------
    # Save initial summary
    # --------------------------------------------------------

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
    # Galleries
    # --------------------------------------------------------

    print()
    print("=" * 70)
    print("BUILDING ERROR GALLERIES")
    print("=" * 70)

    gallery_cases = choose_gallery_cases(
        results_df
    )

    gallery_summary = {}

    for gallery_name, global_ids in (
        gallery_cases.items()
    ):

        gallery_dir = (
            GALLERY_DIR /
            gallery_name
        )

        gallery_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        saved = []

        for rank, global_id in enumerate(
            global_ids,
            start=1,
        ):

            if global_id not in cases:
                continue

            output_path = (
                gallery_dir /
                f"{rank:02d}_{global_id}.jpg"
            )

            save_case_gallery(
                cases[global_id],
                output_path,
            )

            saved.append(
                str(output_path)
            )

        gallery_summary[
            gallery_name
        ] = {
            "count":
                len(saved),

            "files":
                saved,
        }

        print(
            f"{gallery_name:35s}: "
            f"{len(saved)}"
        )

    summary["galleries"] = (
        gallery_summary
    )

    # Rewrite summary with gallery info.
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
    # Human-readable report
    # --------------------------------------------------------

    print()
    print("=" * 70)
    print("ERROR ANALYSIS SUMMARY")
    print("=" * 70)

    print()
    print("OIL SCENES:")

    print(
        f"  Detection: "
        f"{oil_detected}/{len(oil)} "
        f"("
        f"{100.0 * oil_detected / max(len(oil), 1):.2f}%"
        f")"
    )

    print(
        f"  Mean Dice: "
        f"{oil['dice'].mean():.4f}"
    )

    print(
        f"  Mean IoU: "
        f"{oil['iou'].mean():.4f}"
    )

    print(
        f"  Mean Precision: "
        f"{oil['precision'].mean():.4f}"
    )

    print(
        f"  Mean Recall: "
        f"{oil['recall'].mean():.4f}"
    )

    print(
        f"  Mean Boundary Error: "
        f"{oil['boundary_error'].mean():.4f}"
    )

    print()

    print("LOOKALIKE:")

    print(
        f"  False-positive scenes: "
        f"{lookalike_fp}/{len(lookalike)}"
    )

    print(
        f"  False-alarm rate: "
        f"{100.0 * lookalike_fp / max(len(lookalike), 1):.2f}%"
    )

    print(
        f"  Mean predicted area: "
        f"{lookalike['prediction_area'].mean():.1f} pixels"
    )

    print()

    print("NO-OIL:")

    print(
        f"  False-positive scenes: "
        f"{no_oil_fp}/{len(no_oil)}"
    )

    print(
        f"  False-alarm rate: "
        f"{100.0 * no_oil_fp / max(len(no_oil), 1):.2f}%"
    )

    print(
        f"  Mean predicted area: "
        f"{no_oil['prediction_area'].mean():.1f} pixels"
    )

    print()

    print("Worst oil Dice cases:")

    for row in summary[
        "worst_cases"
    ][
        "oil_lowest_dice"
    ][:5]:

        print(
            f"  {row['global_id']:20s} "
            f"Dice={row['dice']:.4f} "
            f"IoU={row['iou']:.4f} "
            f"Recall={row['recall']:.4f} "
            f"FN={row['fn_area']}"
        )

    print()

    print(
        "Largest lookalike false positives:"
    )

    for row in summary[
        "worst_cases"
    ][
        "lookalike_largest_fp"
    ][:5]:

        print(
            f"  {row['global_id']:20s} "
            f"FP={row['fp_area']} "
            f"Pred={row['prediction_area']} "
            f"MaxP={row['max_probability']:.4f}"
        )

    print()

    print(
        "Largest no-oil false positives:"
    )

    for row in summary[
        "worst_cases"
    ][
        "no_oil_largest_fp"
    ][:5]:

        print(
            f"  {row['global_id']:20s} "
            f"FP={row['fp_area']} "
            f"Pred={row['prediction_area']} "
            f"MaxP={row['max_probability']:.4f}"
        )

    print()
    print("=" * 70)
    print("ERROR ANALYSIS COMPLETE")
    print("=" * 70)

    print()
    print("CSV:")
    print(RESULTS_CSV)

    print()
    print("JSON:")
    print(SUMMARY_JSON)

    print()
    print("Galleries:")
    print(GALLERY_DIR)

    print()
    print(
        "IMPORTANT: "
        "No training or test-set modification was performed."
    )


if __name__ == "__main__":
    main()