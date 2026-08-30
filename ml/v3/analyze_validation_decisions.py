from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

# ---------------------------------------------------------------------
# Repository import path
# ---------------------------------------------------------------------

SCRIPT_DIR = Path(__file__).resolve().parent
ML_ROOT = SCRIPT_DIR.parent

if str(ML_ROOT) not in sys.path:
    sys.path.insert(0, str(ML_ROOT))

from src.datasets.oil_dataset import OilSegmentationDataset
from src.models.segmentation_model import build_model

from candidate_dataset import CandidateDataset
from candidate_model import CandidateClassifier


# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------

def load_checkpoint(model, path, device):
    checkpoint = torch.load(
        path,
        map_location=device,
        weights_only=False,
    )

    if isinstance(checkpoint, dict):
        state = (
            checkpoint.get("model_state_dict")
            or checkpoint.get("state_dict")
            or checkpoint.get("model")
        )
    else:
        state = checkpoint

    if state is None:
        raise RuntimeError(
            f"Could not find model state in checkpoint: {path}"
        )

    # Remove DataParallel prefix if present.
    cleaned = {}
    for key, value in state.items():
        if key.startswith("module."):
            key = key[7:]
        cleaned[key] = value

    model.load_state_dict(cleaned, strict=True)
    model.to(device)
    model.eval()

    return checkpoint


def build_segmentation_model():
    """
    Build the same V1 segmentation architecture used by the
    candidate-generation pipeline.
    """
    model = build_model(
        architecture="unet",
        encoder="resnet34",
        encoder_weights=None,
        in_channels=2,
        classes=1,
    )
    return model


def predict_segmentation(
    model,
    image,
    device,
):
    if image.ndim == 3:
        image = image.unsqueeze(0)

    image = image.to(device, non_blocking=True)

    with torch.no_grad():
        logits = model(image)
        probs = torch.sigmoid(logits)

    return probs[0, 0].detach().cpu().numpy()


def connected_components(mask):
    """
    Simple 8-connected component extraction without requiring
    OpenCV/skimage.
    """
    h, w = mask.shape
    visited = np.zeros_like(mask, dtype=bool)
    components = []

    neighbors = [
        (-1, -1), (-1, 0), (-1, 1),
        (0, -1),           (0, 1),
        (1, -1),  (1, 0),  (1, 1),
    ]

    for y in range(h):
        for x in range(w):

            if not mask[y, x] or visited[y, x]:
                continue

            stack = [(y, x)]
            visited[y, x] = True
            pixels = []

            while stack:
                cy, cx = stack.pop()
                pixels.append((cy, cx))

                for dy, dx in neighbors:
                    ny = cy + dy
                    nx = cx + dx

                    if (
                        0 <= ny < h
                        and 0 <= nx < w
                        and mask[ny, nx]
                        and not visited[ny, nx]
                    ):
                        visited[ny, nx] = True
                        stack.append((ny, nx))

            components.append(pixels)

    return components


def component_statistics(
    probability,
    target,
    threshold=0.50,
    min_area=50,
):
    binary = probability >= threshold
    components = connected_components(binary)

    results = []

    for idx, pixels in enumerate(components, start=1):

        if len(pixels) < min_area:
            continue

        ys = np.array([p[0] for p in pixels])
        xs = np.array([p[1] for p in pixels])

        component_mask = np.zeros_like(binary, dtype=bool)
        component_mask[ys, xs] = True

        intersection = np.logical_and(
            component_mask,
            target,
        ).sum()

        union = np.logical_or(
            component_mask,
            target,
        ).sum()

        target_area = target.sum()

        gt_iou = (
            intersection / union
            if union > 0
            else 0.0
        )

        component_precision = (
            intersection / len(pixels)
            if len(pixels) > 0
            else 0.0
        )

        component_recall = (
            intersection / target_area
            if target_area > 0
            else 0.0
        )

        results.append({
            "component_index": idx,
            "area": int(len(pixels)),
            "mean_probability": float(
                probability[component_mask].mean()
            ),
            "p95_probability": float(
                np.percentile(
                    probability[component_mask],
                    95,
                )
            ),
            "max_probability": float(
                probability[component_mask].max()
            ),
            "gt_iou": float(gt_iou),
            "component_precision": float(
                component_precision
            ),
            "component_recall": float(
                component_recall
            ),
            "centroid_x": float(xs.mean()),
            "centroid_y": float(ys.mean()),
            "width": int(xs.max() - xs.min() + 1),
            "height": int(ys.max() - ys.min() + 1),
        })

    return results


def get_global_id(row):
    return str(row["global_id"])


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------

def main():

    parser = argparse.ArgumentParser(
        description=(
            "PS26143 V3 validation decision analysis. "
            "Validation only; never uses the test split."
        )
    )

    parser.add_argument(
        "--v1-checkpoint",
        required=True,
    )

    parser.add_argument(
        "--v3-checkpoint",
        required=True,
    )

    parser.add_argument(
        "--val-manifest",
        required=True,
    )

    parser.add_argument(
        "--candidate-csv",
        required=True,
    )

    parser.add_argument(
        "--output",
        required=True,
    )

    parser.add_argument(
        "--v1-threshold",
        type=float,
        default=0.50,
    )

    parser.add_argument(
        "--min-area",
        type=int,
        default=50,
    )

    parser.add_argument(
        "--v3-thresholds",
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

    args = parser.parse_args()

    output = Path(args.output)
    output.mkdir(
        parents=True,
        exist_ok=True,
    )

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print("=" * 70)
    print("PS26143 — V3 VALIDATION DECISION ANALYSIS")
    print("=" * 70)

    print("Device:", device)

    if torch.cuda.is_available():
        print(
            "GPU:",
            torch.cuda.get_device_name(0),
        )

    print()
    print("Validation manifest:")
    print(args.val_manifest)

    print()
    print("V1 checkpoint:")
    print(args.v1_checkpoint)

    print()
    print("V3 checkpoint:")
    print(args.v3_checkpoint)

    # -----------------------------------------------------------------
    # Validation manifest
    # -----------------------------------------------------------------

    manifest = pd.read_csv(
        args.val_manifest
    )

    required_columns = {
        "global_id",
        "image",
        "mask",
        "dataset",
    }

    missing = required_columns - set(
        manifest.columns
    )

    if missing:
        raise RuntimeError(
            "Validation manifest is missing columns: "
            + ", ".join(sorted(missing))
        )

    print()
    print("=" * 70)
    print("VALIDATION DATA")
    print("=" * 70)

    print("Samples:", len(manifest))
    print(
        manifest["dataset"]
        .value_counts()
        .to_string()
    )

    # -----------------------------------------------------------------
    # Candidate CSV
    # -----------------------------------------------------------------

    candidate_df = pd.read_csv(
        args.candidate_csv
    )

    required_candidate_columns = {
        "global_id",
        "candidate_id",
        "label",
        "area",
        "mean_probability",
        "p95_probability",
        "max_probability",
    }

    missing = (
        required_candidate_columns
        - set(candidate_df.columns)
    )

    if missing:
        raise RuntimeError(
            "Candidate CSV is missing columns: "
            + ", ".join(sorted(missing))
        )

    # -----------------------------------------------------------------
    # Load V3 candidate classifier
    # -----------------------------------------------------------------

    v3_model = CandidateClassifier(
        feature_dim=8,
        pretrained=False,
    )

    v3_checkpoint = load_checkpoint(
        v3_model,
        args.v3_checkpoint,
        device,
    )

    print()
    print("=" * 70)
    print("V3 CLASSIFIER LOADED")
    print("=" * 70)

    if isinstance(v3_checkpoint, dict):
        print(
            "Checkpoint epoch:",
            v3_checkpoint.get("epoch"),
        )

    # -----------------------------------------------------------------
    # Candidate-level validation analysis
    # -----------------------------------------------------------------

    print()
    print("=" * 70)
    print("CANDIDATE DECISION ANALYSIS")
    print("=" * 70)

    candidate_dataset = CandidateDataset(
        candidate_df,
        augment=False,
    )

    loader = torch.utils.data.DataLoader(
        candidate_dataset,
        batch_size=128,
        shuffle=False,
        num_workers=0,
    )

    all_probabilities = []
    all_labels = []

    with torch.no_grad():

        for batch in loader:

            images = batch["image"].to(
                device,
                non_blocking=True,
            )

            features = batch["features"].to(
                device,
                non_blocking=True,
            )

            logits = v3_model(
                images,
                features,
            )

            probabilities = torch.sigmoid(
                logits
            ).flatten()

            all_probabilities.extend(
                probabilities.cpu().numpy().tolist()
            )

            all_labels.extend(
                batch["label"]
                .flatten()
                .cpu()
                .numpy()
                .tolist()
            )

    candidate_df = candidate_df.copy()

    candidate_df[
        "v3_probability"
    ] = np.asarray(
        all_probabilities,
        dtype=np.float32,
    )

    candidate_df[
        "true_label"
    ] = np.asarray(
        all_labels,
        dtype=np.float32,
    )

    candidate_df[
        "v3_accept"
    ] = (
        candidate_df["v3_probability"]
        >= 0.50
    )

    # -----------------------------------------------------------------
    # Candidate classifier threshold sweep
    # -----------------------------------------------------------------

    threshold_rows = []

    for threshold in args.v3_thresholds:

        pred = (
            candidate_df["v3_probability"]
            >= threshold
        )

        true = (
            candidate_df["true_label"]
            > 0.5
        )

        tp = int((pred & true).sum())
        fp = int((pred & ~true).sum())
        tn = int((~pred & ~true).sum())
        fn = int((~pred & true).sum())

        precision = (
            tp / (tp + fp)
            if tp + fp > 0
            else 0.0
        )

        recall = (
            tp / (tp + fn)
            if tp + fn > 0
            else 0.0
        )

        f1 = (
            2 * precision * recall
            / (precision + recall)
            if precision + recall > 0
            else 0.0
        )

        threshold_rows.append({
            "threshold": threshold,
            "tp": tp,
            "fp": fp,
            "tn": tn,
            "fn": fn,
            "precision": precision,
            "recall": recall,
            "f1": f1,
        })

    threshold_df = pd.DataFrame(
        threshold_rows
    )

    threshold_df.to_csv(
        output / "candidate_threshold_sweep.csv",
        index=False,
    )

    print(
        threshold_df.to_string(
            index=False
        )
    )

    # -----------------------------------------------------------------
    # Critical cases
    # -----------------------------------------------------------------

    oil_candidates = candidate_df[
        candidate_df["global_id"].str.startswith(
            "oil_"
        )
    ]

    lookalike_candidates = candidate_df[
        candidate_df["global_id"].str.startswith(
            "lookalike_"
        )
    ]

    no_oil_candidates = candidate_df[
        candidate_df["global_id"].str.startswith(
            "no_oil_"
        )
    ]

    rejected_oil = oil_candidates[
        oil_candidates["v3_probability"] < 0.50
    ].copy()

    accepted_lookalike = lookalike_candidates[
        lookalike_candidates["v3_probability"] >= 0.50
    ].copy()

    accepted_no_oil = no_oil_candidates[
        no_oil_candidates["v3_probability"] >= 0.50
    ].copy()

    rejected_oil = rejected_oil.sort_values(
        "v3_probability",
        ascending=True,
    )

    accepted_lookalike = accepted_lookalike.sort_values(
        "v3_probability",
        ascending=False,
    )

    print()
    print("=" * 70)
    print("REJECTED OIL CANDIDATES")
    print("=" * 70)

    print(
        "Count:",
        len(rejected_oil),
    )

    if len(rejected_oil):
        print(
            rejected_oil[
                [
                    "global_id",
                    "candidate_id",
                    "area",
                    "mean_probability",
                    "p95_probability",
                    "max_probability",
                    "v3_probability",
                    "gt_iou",
                ]
            ]
            .head(30)
            .to_string(index=False)
        )

    print()
    print("=" * 70)
    print("ACCEPTED LOOKALIKE CANDIDATES")
    print("=" * 70)

    print(
        "Count:",
        len(accepted_lookalike),
    )

    if len(accepted_lookalike):
        print(
            accepted_lookalike[
                [
                    "global_id",
                    "candidate_id",
                    "area",
                    "mean_probability",
                    "p95_probability",
                    "max_probability",
                    "v3_probability",
                    "gt_iou",
                ]
            ]
            .head(30)
            .to_string(index=False)
        )

    print()
    print("=" * 70)
    print("ACCEPTED NO-OIL CANDIDATES")
    print("=" * 70)

    print(
        "Count:",
        len(accepted_no_oil),
    )

    if len(accepted_no_oil):
        print(
            accepted_no_oil[
                [
                    "global_id",
                    "candidate_id",
                    "area",
                    "mean_probability",
                    "p95_probability",
                    "max_probability",
                    "v3_probability",
                ]
            ]
            .head(30)
            .to_string(index=False)
        )

    # -----------------------------------------------------------------
    # Save candidate-level analysis
    # -----------------------------------------------------------------

    candidate_df.to_csv(
        output / "validation_candidate_decisions.csv",
        index=False,
    )

    rejected_oil.to_csv(
        output / "rejected_oil_candidates.csv",
        index=False,
    )

    accepted_lookalike.to_csv(
        output / "accepted_lookalike_candidates.csv",
        index=False,
    )

    accepted_no_oil.to_csv(
        output / "accepted_no_oil_candidates.csv",
        index=False,
    )

    # -----------------------------------------------------------------
    # Scene-level analysis from candidate CSV
    # -----------------------------------------------------------------

    scene_rows = []

    for global_id, group in candidate_df.groupby(
        "global_id"
    ):

        dataset = str(
            group["dataset"].iloc[0]
        ) if "dataset" in group.columns else (
            "oil"
            if global_id.startswith("oil_")
            else "lookalike"
            if global_id.startswith("lookalike_")
            else "no_oil"
        )

        row = {
            "global_id": global_id,
            "dataset": dataset,
            "candidate_count": len(group),
            "accepted_050": int(
                (group["v3_probability"] >= 0.50)
                .sum()
            ),
            "accepted_045": int(
                (group["v3_probability"] >= 0.45)
                .sum()
            ),
            "accepted_055": int(
                (group["v3_probability"] >= 0.55)
                .sum()
            ),
            "max_probability": float(
                group["v3_probability"].max()
            ),
            "mean_probability": float(
                group["v3_probability"].mean()
            ),
        }

        if dataset == "oil":
            row["oil_candidate_rejection"] = (
                row["accepted_050"] == 0
            )
        else:
            row["oil_candidate_rejection"] = False

        scene_rows.append(row)

    scene_df = pd.DataFrame(scene_rows)

    scene_df.to_csv(
        output / "validation_scene_decisions.csv",
        index=False,
    )

    # -----------------------------------------------------------------
    # Summary
    # -----------------------------------------------------------------

    oil_scenes = scene_df[
        scene_df["dataset"] == "oil"
    ]

    lookalike_scenes = scene_df[
        scene_df["dataset"] == "lookalike"
    ]

    no_oil_scenes = scene_df[
        scene_df["dataset"] == "no_oil"
    ]

    summary = {
        "experiment": "oil-seg-v3",
        "source": "validation_only",
        "v1_threshold": args.v1_threshold,
        "min_area": args.min_area,
        "v3_thresholds": args.v3_thresholds,

        "oil_scenes": int(
            len(oil_scenes)
        ),

        "oil_scenes_without_accepted_candidate_050": int(
            (
                oil_scenes["accepted_050"] == 0
            ).sum()
        ),

        "lookalike_scenes_with_accepted_candidate_050": int(
            (
                lookalike_scenes["accepted_050"] > 0
            ).sum()
        ),

        "no_oil_scenes_with_accepted_candidate_050": int(
            (
                no_oil_scenes["accepted_050"] > 0
            ).sum()
        ),

        "rejected_oil_candidates": int(
            len(rejected_oil)
        ),

        "accepted_lookalike_candidates": int(
            len(accepted_lookalike)
        ),

        "accepted_no_oil_candidates": int(
            len(accepted_no_oil)
        ),
    }

    with open(
        output / "validation_decision_summary.json",
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
    print("VALIDATION DECISION SUMMARY")
    print("=" * 70)

    print(
        json.dumps(
            summary,
            indent=2,
        )
    )

    print()
    print("=" * 70)
    print("✅ V3 VALIDATION DECISION ANALYSIS COMPLETE")
    print("=" * 70)

    print(
        "Output:",
        output,
    )


if __name__ == "__main__":
    main()