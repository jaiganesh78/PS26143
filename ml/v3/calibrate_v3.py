from __future__ import annotations

import argparse
import json
import sys
import traceback
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


# ============================================================
# PATH SETUP
# ============================================================

HERE = Path(__file__).resolve()
ML_ROOT = HERE.parent.parent
REPO_ROOT = ML_ROOT.parent
SRC_ROOT = ML_ROOT / "src"

for p in [REPO_ROOT, ML_ROOT, SRC_ROOT]:
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))


# Existing project components.
from models.segmentation_model import build_model
from candidate_model import CandidateClassifier


# ============================================================
# CONSTANTS
# ============================================================

IMAGE_SIZE = 512
V1_THRESHOLD = 0.50

# Thresholds are deliberately broad enough to find the
# operating point rather than assuming 0.50 is optimal.
V3_THRESHOLDS = [
    0.50,
    0.55,
    0.60,
    0.65,
    0.70,
    0.75,
    0.80,
    0.85,
    0.90,
    0.95,
]

MIN_AREAS = [
    10,
    20,
    50,
    100,
]

# Keep this conservative. We are calibrating the candidate
# rejection system, not destroying the V1 geometry.
MORPHOLOGY_OPTIONS = [
    "none",
    "opening",
    "closing",
    "open_close",
]


# ============================================================
# LOGGING
# ============================================================

def banner(text):
    print()
    print("=" * 70)
    print(text)
    print("=" * 70)


# ============================================================
# DATASET
# ============================================================

class ManifestDataset(Dataset):

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

        missing = required - set(
            self.df.columns
        )

        if missing:
            raise RuntimeError(
                "Manifest schema error.\n"
                f"Missing columns: {sorted(missing)}\n"
                f"Available columns: "
                f"{list(self.df.columns)}"
            )

        if len(self.df) == 0:
            raise RuntimeError(
                f"Manifest contains zero rows: "
                f"{manifest_path}"
            )

        # Verify every path before starting inference.
        missing_files = []

        for _, row in self.df.iterrows():

            image = Path(row["image"])
            mask = Path(row["mask"])

            if not image.exists():
                missing_files.append(
                    str(image)
                )

            if not mask.exists():
                missing_files.append(
                    str(mask)
                )

        if missing_files:
            raise FileNotFoundError(
                "Manifest contains missing files.\n"
                + "\n".join(
                    missing_files[:20]
                )
            )

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):

        row = self.df.iloc[idx]

        image = np.load(
            Path(row["image"])
        ).astype(np.float32)

        mask = np.load(
            Path(row["mask"])
        ).astype(np.float32)

        if image.shape != (
            2,
            IMAGE_SIZE,
            IMAGE_SIZE,
        ):
            raise ValueError(
                f"{row['global_id']}: "
                f"expected image shape "
                f"(2,512,512), got "
                f"{image.shape}"
            )

        if mask.shape != (
            IMAGE_SIZE,
            IMAGE_SIZE,
        ):
            raise ValueError(
                f"{row['global_id']}: "
                f"expected mask shape "
                f"(512,512), got "
                f"{mask.shape}"
            )

        return (
            image,
            mask,
            str(row["global_id"]),
            str(row["dataset"]),
        )


# ============================================================
# CHECKPOINT LOADING
# ============================================================

def load_checkpoint(
    model,
    checkpoint_path,
    device,
):

    checkpoint_path = Path(
        checkpoint_path
    )

    if not checkpoint_path.exists():
        raise FileNotFoundError(
            f"Checkpoint does not exist:\n"
            f"{checkpoint_path}"
        )

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

    try:
        model.load_state_dict(
            state_dict,
            strict=True,
        )
    except Exception as exc:

        raise RuntimeError(
            f"Checkpoint architecture mismatch.\n"
            f"Checkpoint: {checkpoint_path}\n"
            f"Model: {type(model).__name__}\n"
            f"Original error: {exc}"
        ) from exc

    return checkpoint


# ============================================================
# MODEL BUILDING
# ============================================================

def build_v1(device):

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


def build_v3(device):

    model = CandidateClassifier(
        feature_dim=8,
        pretrained=False,
    )

    model.to(device)
    model.eval()

    return model


# ============================================================
# V1 INFERENCE
# ============================================================

@torch.inference_mode()
def predict_v1(
    model,
    image,
    device,
):

    x = torch.from_numpy(
        image
    ).unsqueeze(0).to(device)

    with torch.autocast(
        device_type="cuda",
        dtype=torch.float16,
        enabled=device.type == "cuda",
    ):

        logits = model(x)
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
):

    binary = (
        probability >= V1_THRESHOLD
    ).astype(np.uint8)

    n_labels, labels, stats, centroids = (
        cv2.connectedComponentsWithStats(
            binary,
            connectivity=8,
        )
    )

    candidates = []

    for component_id in range(
        1,
        n_labels,
    ):

        area = int(
            stats[
                component_id,
                cv2.CC_STAT_AREA,
            ]
        )

        if area < 10:
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

        component_pixels = (
            labels == component_id
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

        # Candidate crop.
        crop_size = 128
        half = crop_size // 2

        center_x = int(round(cx))
        center_y = int(round(cy))

        x0 = center_x - half
        y0 = center_y - half
        x1 = x0 + crop_size
        y1 = y0 + crop_size

        pad_left = max(0, -x0)
        pad_top = max(0, -y0)
        pad_right = max(
            0,
            x1 - IMAGE_SIZE,
        )
        pad_bottom = max(
            0,
            y1 - IMAGE_SIZE,
        )

        x0 = max(0, x0)
        y0 = max(0, y0)
        x1 = min(IMAGE_SIZE, x1)
        y1 = min(IMAGE_SIZE, y1)

        crop_image = image[
            :,
            y0:y1,
            x0:x1,
        ]

        crop_mask = (
            labels[
                y0:y1,
                x0:x1,
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
            crop_mask.shape
            != (
                crop_size,
                crop_size,
            )
        ):

            crop_mask = cv2.resize(
                crop_mask,
                (
                    crop_size,
                    crop_size,
                ),
                interpolation=cv2.INTER_NEAREST,
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
            / max(height, 1)
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

                "crop_image":
                    crop_image,

                "crop_mask":
                    crop_mask,

                "features":
                    features,
            }
        )

    return candidates


# ============================================================
# V3 CLASSIFIER
# ============================================================

@torch.inference_mode()
def classify_candidates(
    model,
    candidates,
    device,
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

    with torch.autocast(
        device_type="cuda",
        dtype=torch.float16,
        enabled=device.type == "cuda",
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

    output = []

    for candidate, probability in zip(
        candidates,
        probabilities,
    ):

        item = dict(candidate)

        item[
            "v3_probability"
        ] = float(probability)

        output.append(item)

    return output


# ============================================================
# MASK RECONSTRUCTION
# ============================================================

def reconstruct_mask(
    probability,
    candidates,
    v3_threshold,
    min_area,
    morphology,
):

    base = (
        probability >= V1_THRESHOLD
    ).astype(np.uint8)

    n_labels, labels, stats, _ = (
        cv2.connectedComponentsWithStats(
            base,
            connectivity=8,
        )
    )

    output = np.zeros_like(
        base,
        dtype=np.uint8,
    )

    accepted = 0

    for candidate in candidates:

        component_id = candidate[
            "component_id"
        ]

        area = int(
            candidate["area"]
        )

        score = float(
            candidate[
                "v3_probability"
            ]
        )

        if area < min_area:
            continue

        if score < v3_threshold:
            continue

        if component_id >= n_labels:
            continue

        output[
            labels == component_id
        ] = 1

        accepted += 1

    # Small 3x3 morphological operations only.
    # These are deliberately conservative.
    kernel = np.ones(
        (3, 3),
        dtype=np.uint8,
    )

    if morphology == "opening":

        output = cv2.morphologyEx(
            output,
            cv2.MORPH_OPEN,
            kernel,
        )

    elif morphology == "closing":

        output = cv2.morphologyEx(
            output,
            cv2.MORPH_CLOSE,
            kernel,
        )

    elif morphology == "open_close":

        output = cv2.morphologyEx(
            output,
            cv2.MORPH_OPEN,
            kernel,
        )

        output = cv2.morphologyEx(
            output,
            cv2.MORPH_CLOSE,
            kernel,
        )

    return output, accepted


# ============================================================
# METRICS
# ============================================================

def metrics(
    prediction,
    target,
):

    p = prediction > 0
    t = target > 0

    tp = int(
        np.logical_and(
            p,
            t,
        ).sum()
    )

    fp = int(
        np.logical_and(
            p,
            ~t,
        ).sum()
    )

    fn = int(
        np.logical_and(
            ~p,
            t,
        ).sum()
    )

    tn = int(
        np.logical_and(
            ~p,
            ~t,
        ).sum()
    )

    dice = (
        2 * tp
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
        "predicted_pixels": int(
            p.sum()
        ),
    }


# ============================================================
# CALIBRATION OBJECTIVE
# ============================================================

def score_configuration(
    rows,
):

    df = pd.DataFrame(rows)

    oil = df[
        df["dataset"] == "oil"
    ]

    lookalike = df[
        df["dataset"] == "lookalike"
    ]

    no_oil = df[
        df["dataset"] == "no_oil"
    ]

    if oil.empty:
        raise RuntimeError(
            "Validation set contains no oil scenes."
        )

    oil_detection = float(
        (
            oil["predicted_pixels"] > 0
        ).mean()
    )

    lookalike_far = (
        float(
            (
                lookalike[
                    "predicted_pixels"
                ] > 0
            ).mean()
        )
        if not lookalike.empty
        else 0.0
    )

    no_oil_far = (
        float(
            (
                no_oil[
                    "predicted_pixels"
                ] > 0
            ).mean()
        )
        if not no_oil.empty
        else 0.0
    )

    oil_dice = float(
        oil["dice"].mean()
    )

    oil_iou = float(
        oil["iou"].mean()
    )

    oil_precision = float(
        oil["precision"].mean()
    )

    oil_recall = float(
        oil["recall"].mean()
    )

    negative_far = (
        0.6 * lookalike_far
        + 0.4 * no_oil_far
    )

    # Hard rejection requirements.
    #
    # We strongly prefer:
    #   oil detection >= 98%
    #   lookalike FAR <= 10%
    #   no-oil FAR <= 5%
    #
    # Configurations violating these are penalized heavily.
    penalty = 0.0

    if oil_detection < 0.98:
        penalty += (
            0.98 - oil_detection
        ) * 10.0

    if lookalike_far > 0.10:
        penalty += (
            lookalike_far - 0.10
        ) * 10.0

    if no_oil_far > 0.05:
        penalty += (
            no_oil_far - 0.05
        ) * 10.0

    # Main objective.
    #
    # Rejection is important, but segmentation must remain
    # useful for downstream polygon generation.
    score = (
        0.35 * oil_dice
        + 0.20 * oil_iou
        + 0.15 * oil_precision
        + 0.15 * oil_recall
        + 0.15 * oil_detection
        - 0.50 * negative_far
        - penalty
    )

    return {
        "score": float(score),
        "oil_detection_rate":
            oil_detection,
        "lookalike_false_alarm_rate":
            lookalike_far,
        "no_oil_false_alarm_rate":
            no_oil_far,
        "oil_dice":
            oil_dice,
        "oil_iou":
            oil_iou,
        "oil_precision":
            oil_precision,
        "oil_recall":
            oil_recall,
    }


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
        "--val-manifest",
        type=Path,
        required=True,
    )

    parser.add_argument(
        "--output",
        type=Path,
        required=True,
    )

    args = parser.parse_args()

    banner(
        "PS26143 — V3 VALIDATION CALIBRATION"
    )

    print(
        "IMPORTANT:"
    )

    print(
        "Validation set ONLY."
    )

    print(
        "Test set will NOT be accessed."
    )

    print(
        "V1 checkpoint remains frozen."
    )

    print(
        "V3 checkpoint remains frozen."
    )

    # --------------------------------------------------------
    # Device
    # --------------------------------------------------------

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print()
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
    # Data
    # --------------------------------------------------------

    banner(
        "LOADING VALIDATION DATA"
    )

    dataset = ManifestDataset(
        args.val_manifest
    )

    print(
        "Validation samples:",
        len(dataset),
    )

    print(
        dataset.df[
            "dataset"
        ].value_counts()
    )

    # --------------------------------------------------------
    # Models
    # --------------------------------------------------------

    banner(
        "LOADING FROZEN V1 MODEL"
    )

    v1 = build_v1(
        device
    )

    checkpoint_v1 = load_checkpoint(
        v1,
        args.v1_checkpoint,
        device,
    )

    if isinstance(
        checkpoint_v1,
        dict,
    ):

        print(
            "V1 checkpoint epoch:",
            checkpoint_v1.get(
                "epoch",
                "unknown",
            ),
        )

    print(
        "V1 loaded successfully."
    )

    banner(
        "LOADING FROZEN V3 CLASSIFIER"
    )

    v3 = build_v3(
        device
    )

    checkpoint_v3 = load_checkpoint(
        v3,
        args.v3_checkpoint,
        device,
    )

    if isinstance(
        checkpoint_v3,
        dict,
    ):

        print(
            "V3 checkpoint epoch:",
            checkpoint_v3.get(
                "epoch",
                "unknown",
            ),
        )

    print(
        "V3 loaded successfully."
    )

    # --------------------------------------------------------
    # Generate base predictions ONCE.
    # --------------------------------------------------------

    banner(
        "GENERATING VALIDATION CANDIDATES"
    )

    cached = []

    for idx in range(
        len(dataset)
    ):

        image, target, global_id, dataset_name = (
            dataset[idx]
        )

        probability = predict_v1(
            v1,
            image,
            device,
        )

        candidates = extract_candidates(
            image,
            probability,
        )

        candidates = classify_candidates(
            v3,
            candidates,
            device,
        )

        cached.append(
            {
                "image":
                    image,

                "target":
                    target,

                "global_id":
                    global_id,

                "dataset":
                    dataset_name,

                "probability":
                    probability,

                "candidates":
                    candidates,
            }
        )

        if (
            idx + 1
        ) % 25 == 0 or (
            idx + 1
        ) == len(dataset):

            print(
                f"Processed "
                f"{idx + 1}/"
                f"{len(dataset)}"
            )

    # --------------------------------------------------------
    # Sweep configurations.
    # --------------------------------------------------------

    banner(
        "SWEEPING V3 OPERATING POINT"
    )

    results = []

    total_configs = (
        len(V3_THRESHOLDS)
        * len(MIN_AREAS)
        * len(MORPHOLOGY_OPTIONS)
    )

    current = 0

    for threshold in V3_THRESHOLDS:

        for min_area in MIN_AREAS:

            for morphology in (
                MORPHOLOGY_OPTIONS
            ):

                current += 1

                rows = []

                for case in cached:

                    prediction, accepted = (
                        reconstruct_mask(
                            case[
                                "probability"
                            ],
                            case[
                                "candidates"
                            ],
                            threshold,
                            min_area,
                            morphology,
                        )
                    )

                    m = metrics(
                        prediction,
                        case["target"],
                    )

                    rows.append(
                        {
                            "global_id":
                                case[
                                    "global_id"
                                ],

                            "dataset":
                                case[
                                    "dataset"
                                ],

                            **m,
                        }
                    )

                summary = score_configuration(
                    rows
                )

                summary.update(
                    {
                        "v3_threshold":
                            threshold,

                        "min_area":
                            min_area,

                        "morphology":
                            morphology,
                    }
                )

                results.append(
                    summary
                )

                print(
                    f"[{current:3d}/"
                    f"{total_configs}] "
                    f"thr={threshold:.2f} "
                    f"area={min_area:3d} "
                    f"morph={morphology:10s} "
                    f"score="
                    f"{summary['score']:.5f} "
                    f"oilDice="
                    f"{summary['oil_dice']:.4f} "
                    f"lookFAR="
                    f"{summary['lookalike_false_alarm_rate']:.3f} "
                    f"noOilFAR="
                    f"{summary['no_oil_false_alarm_rate']:.3f}"
                )

    results_df = pd.DataFrame(
        results
    )

    results_df = results_df.sort_values(
        "score",
        ascending=False,
    ).reset_index(
        drop=True
    )

    # --------------------------------------------------------
    # Save sweep.
    # --------------------------------------------------------

    args.output.mkdir(
        parents=True,
        exist_ok=True,
    )

    sweep_path = (
        args.output
        / "validation_calibration_sweep.csv"
    )

    results_df.to_csv(
        sweep_path,
        index=False,
    )

    # --------------------------------------------------------
    # Select best VALIDATION configuration.
    # --------------------------------------------------------

    best = results_df.iloc[0].to_dict()

    selected = {
        "experiment":
            "oil-seg-v3",

        "source":
            "validation_only",

        "v1_threshold":
            V1_THRESHOLD,

        "v3_threshold":
            float(
                best["v3_threshold"]
            ),

        "min_area":
            int(
                best["min_area"]
            ),

        "morphology":
            str(
                best["morphology"]
            ),

        "validation_score":
            float(
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

        "checkpoint_v1":
            str(args.v1_checkpoint),

        "checkpoint_v3":
            str(args.v3_checkpoint),

        "validation_manifest":
            str(args.val_manifest),
    }

    config_path = (
        args.output
        / "selected_config.json"
    )

    with open(
        config_path,
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            selected,
            f,
            indent=2,
        )

    # --------------------------------------------------------
    # Display best configurations.
    # --------------------------------------------------------

    banner(
        "TOP VALIDATION CONFIGURATIONS"
    )

    display_columns = [
        "score",
        "v3_threshold",
        "min_area",
        "morphology",
        "oil_detection_rate",
        "lookalike_false_alarm_rate",
        "no_oil_false_alarm_rate",
        "oil_dice",
        "oil_iou",
        "oil_precision",
        "oil_recall",
    ]

    print(
        results_df[
            display_columns
        ].head(10).to_string(
            index=False
        )
    )

    banner(
        "SELECTED V3 CONFIGURATION"
    )

    print(
        json.dumps(
            selected,
            indent=2,
        )
    )

    print()
    print(
        "Sweep CSV:",
        sweep_path,
    )

    print(
        "Selected config:",
        config_path,
    )

    banner(
        "✅ V3 VALIDATION CALIBRATION COMPLETE"
    )

    print(
        "TEST SET WAS NOT ACCESSED."
    )


# ============================================================
# TOP LEVEL ERROR HANDLER
# ============================================================

if __name__ == "__main__":

    try:

        main()

    except Exception as exc:

        banner(
            "❌ V3 CALIBRATION FAILED"
        )

        print(
            "ERROR TYPE:",
            type(exc).__name__,
        )

        print(
            "ERROR:",
            str(exc),
        )

        print()
        print(
            "COMPLETE TRACEBACK:"
        )

        traceback.print_exc()

        sys.exit(1)