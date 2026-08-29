from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import (
    DataLoader,
    WeightedRandomSampler,
)

from candidate_dataset import CandidateDataset
from candidate_model import CandidateClassifier


SEED = 42


def seed_everything(seed=SEED):

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def evaluate(
    model,
    loader,
    device,
):

    model.eval()

    criterion = nn.BCEWithLogitsLoss()

    total_loss = 0.0
    total = 0

    tp = fp = tn = fn = 0

    with torch.inference_mode():

        for batch in loader:

            images = batch["image"].to(
                device,
                non_blocking=True,
            )

            features = batch["features"].to(
                device,
                non_blocking=True,
            )

            labels = batch["label"].to(
                device,
                non_blocking=True,
            )

            logits = model(
                images,
                features,
            )

            loss = criterion(
                logits,
                labels,
            )

            total_loss += (
                loss.item()
                * labels.size(0)
            )

            total += labels.size(0)

            probs = torch.sigmoid(
                logits
            )

            pred = (
                probs >= 0.5
            ).float()

            tp += int(
                (
                    (pred == 1)
                    & (labels == 1)
                ).sum().item()
            )

            fp += int(
                (
                    (pred == 1)
                    & (labels == 0)
                ).sum().item()
            )

            tn += int(
                (
                    (pred == 0)
                    & (labels == 0)
                ).sum().item()
            )

            fn += int(
                (
                    (pred == 0)
                    & (labels == 1)
                ).sum().item()
            )

    loss = (
        total_loss / max(total, 1)
    )

    precision = (
        tp / max(tp + fp, 1)
    )

    recall = (
        tp / max(tp + fn, 1)
    )

    f1 = (
        2.0 * precision * recall
        / max(
            precision + recall,
            1e-8,
        )
    )

    false_positive_rate = (
        fp / max(fp + tn, 1)
    )

    return {
        "loss": loss,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "false_positive_rate": false_positive_rate,
        "tp": tp,
        "fp": fp,
        "tn": tn,
        "fn": fn,
    }


def load_dataframe(path):

    df = pd.read_csv(path)

    required = {
        "crop_image",
        "crop_mask",
        "label",
        "global_id",
        "candidate_id",
        "area",
        "width",
        "height",
        "centroid_x",
        "centroid_y",
        "mean_probability",
        "p95_probability",
        "max_probability",
    }

    missing = (
        required
        - set(df.columns)
    )

    if missing:
        raise RuntimeError(
            "Missing candidate columns: "
            f"{sorted(missing)}"
        )

    for col in [
        "crop_image",
        "crop_mask",
    ]:

        missing_files = []

        for value in df[col]:

            path = Path(value)

            if not path.exists():
                missing_files.append(
                    str(path)
                )

        if missing_files:

            raise FileNotFoundError(
                missing_files[0]
            )

    return df


def split_by_scene(
    df,
    seed=SEED,
):

    scenes = (
        df["global_id"]
        .drop_duplicates()
        .tolist()
    )

    rng = random.Random(seed)
    rng.shuffle(scenes)

    n = len(scenes)

    n_val = max(
        1,
        int(round(0.20 * n)),
    )

    val_scenes = set(
        scenes[:n_val]
    )

    train_df = df[
        ~df["global_id"].isin(
            val_scenes
        )
    ].copy()

    val_df = df[
        df["global_id"].isin(
            val_scenes
        )
    ].copy()

    return (
        train_df.reset_index(drop=True),
        val_df.reset_index(drop=True),
    )


def save_checkpoint(
    path,
    model,
    optimizer,
    scheduler,
    epoch,
    best_score,
):

    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": (
                scheduler.state_dict()
                if scheduler is not None
                else None
            ),
            "best_score": best_score,
        },
        path,
    )


def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--candidates",
        type=Path,
        default=Path(
            "/content/drive/MyDrive/PS26143/"
            "evaluation/oil_seg_v3/candidates/"
            "train_candidates.csv"
        ),
    )

    parser.add_argument(
        "--output",
        type=Path,
        default=Path(
            "/content/drive/MyDrive/PS26143/"
            "checkpoints/oil_seg_v3_candidate"
        ),
    )

    parser.add_argument(
        "--epochs",
        type=int,
        default=30,
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=64,
    )

    parser.add_argument(
        "--lr",
        type=float,
        default=1e-4,
    )

    args = parser.parse_args()

    seed_everything()

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print("=" * 70)
    print(
        "PS26143 — V3 CANDIDATE CLASSIFIER"
    )
    print("=" * 70)

    print("Device:", device)

    if device.type == "cuda":

        print(
            "GPU:",
            torch.cuda.get_device_name(0),
        )

    # --------------------------------------------------------
    # DATA
    # --------------------------------------------------------

    df = load_dataframe(
        args.candidates
    )

    print(
        "Candidates:",
        len(df),
    )

    print(
        df["label"].value_counts()
    )

    train_df, val_df = split_by_scene(
        df
    )

    print()
    print(
        "Training candidates:",
        len(train_df),
    )

    print(
        "Validation candidates:",
        len(val_df),
    )

    print(
        "Training scenes:",
        train_df["global_id"].nunique(),
    )

    print(
        "Validation scenes:",
        val_df["global_id"].nunique(),
    )

    # --------------------------------------------------------
    # DATASETS
    # --------------------------------------------------------

    train_dataset = CandidateDataset(
        train_df,
        augment=True,
    )

    val_dataset = CandidateDataset(
        val_df,
        augment=False,
    )

    # --------------------------------------------------------
    # BALANCED SAMPLING
    # --------------------------------------------------------

    labels = (
        train_df["label"]
        .map(
            {
                "positive": 1,
                "hard_negative": 0,
            }
        )
        .values
    )

    class_counts = np.bincount(
        labels.astype(int),
        minlength=2,
    )

    weights = np.zeros(
        len(labels),
        dtype=np.float64,
    )

    for cls in [0, 1]:

        if class_counts[cls] > 0:

            weights[
                labels == cls
            ] = (
                1.0
                / class_counts[cls]
            )

    sampler = WeightedRandomSampler(
        torch.as_tensor(
            weights,
            dtype=torch.double,
        ),
        num_samples=len(weights),
        replacement=True,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        sampler=sampler,
        num_workers=0,
        pin_memory=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=True,
    )

    # --------------------------------------------------------
    # MODEL
    # --------------------------------------------------------

    model = CandidateClassifier(
        feature_dim=8,
        pretrained=True,
    ).to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=1e-4,
    )

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="max",
        factor=0.5,
        patience=3,
        min_lr=5e-6,
    )

    criterion = nn.BCEWithLogitsLoss()

    scaler = torch.amp.GradScaler(
        "cuda",
        enabled=(
            device.type == "cuda"
        ),
    )

    args.output.mkdir(
        parents=True,
        exist_ok=True,
    )

    best_path = (
        args.output
        / "best.pt"
    )

    last_path = (
        args.output
        / "last.pt"
    )

    log_path = (
        args.output
        / "training.csv"
    )

    # --------------------------------------------------------
    # RESUME
    # --------------------------------------------------------

    start_epoch = 1
    best_score = -float("inf")

    if last_path.exists():

        print()
        print(
            "Resuming:",
            last_path,
        )

        checkpoint = torch.load(
            last_path,
            map_location=device,
            weights_only=False,
        )

        model.load_state_dict(
            checkpoint[
                "model_state_dict"
            ]
        )

        optimizer.load_state_dict(
            checkpoint[
                "optimizer_state_dict"
            ]
        )

        if (
            checkpoint.get(
                "scheduler_state_dict"
            )
            is not None
        ):

            scheduler.load_state_dict(
                checkpoint[
                    "scheduler_state_dict"
                ]
            )

        start_epoch = (
            checkpoint["epoch"] + 1
        )

        best_score = checkpoint.get(
            "best_score",
            best_score,
        )

    # --------------------------------------------------------
    # TRAIN
    # --------------------------------------------------------

    print()
    print("=" * 70)
    print("TRAINING")
    print("=" * 70)

    history = []

    if log_path.exists():

        history = pd.read_csv(
            log_path
        ).to_dict(
            orient="records"
        )

    for epoch in range(
        start_epoch,
        args.epochs + 1,
    ):

        start_time = time.time()

        model.train()

        running_loss = 0.0
        samples = 0

        for batch in train_loader:

            images = batch[
                "image"
            ].to(
                device,
                non_blocking=True,
            )

            features = batch[
                "features"
            ].to(
                device,
                non_blocking=True,
            )

            labels_batch = batch[
                "label"
            ].to(
                device,
                non_blocking=True,
            )

            optimizer.zero_grad(
                set_to_none=True
            )

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

                loss = criterion(
                    logits,
                    labels_batch,
                )

            scaler.scale(
                loss
            ).backward()

            scaler.unscale_(
                optimizer
            )

            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                1.0,
            )

            scaler.step(
                optimizer
            )

            scaler.update()

            batch_size = (
                labels_batch.size(0)
            )

            running_loss += (
                loss.item()
                * batch_size
            )

            samples += batch_size

        train_loss = (
            running_loss
            / max(samples, 1)
        )

        metrics = evaluate(
            model,
            val_loader,
            device,
        )

        # F1 is the primary classifier criterion.
        scheduler.step(
            metrics["f1"]
        )

        elapsed = (
            time.time()
            - start_time
        )

        row = {
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": metrics["loss"],
            "precision": metrics[
                "precision"
            ],
            "recall": metrics[
                "recall"
            ],
            "f1": metrics["f1"],
            "false_positive_rate": metrics[
                "false_positive_rate"
            ],
            "tp": metrics["tp"],
            "fp": metrics["fp"],
            "tn": metrics["tn"],
            "fn": metrics["fn"],
            "learning_rate": optimizer.param_groups[
                0
            ]["lr"],
            "epoch_seconds": elapsed,
        }

        history.append(row)

        pd.DataFrame(
            history
        ).to_csv(
            log_path,
            index=False,
        )

        save_checkpoint(
            last_path,
            model,
            optimizer,
            scheduler,
            epoch,
            best_score,
        )

        # ----------------------------------------------------
        # Best model
        # ----------------------------------------------------

        if metrics["f1"] > best_score:

            best_score = metrics["f1"]

            save_checkpoint(
                best_path,
                model,
                optimizer,
                scheduler,
                epoch,
                best_score,
            )

            marker = "  <-- BEST"

        else:

            marker = ""

        print()
        print(
            f"Epoch {epoch:02d}/{args.epochs}"
            f" | loss={train_loss:.4f}"
            f" | val_loss={metrics['loss']:.4f}"
            f" | precision={metrics['precision']:.4f}"
            f" | recall={metrics['recall']:.4f}"
            f" | F1={metrics['f1']:.4f}"
            f" | FPR={metrics['false_positive_rate']:.4f}"
            f"{marker}"
        )

    # --------------------------------------------------------
    # SUMMARY
    # --------------------------------------------------------

    summary = {
        "experiment": "oil-seg-v3-candidate-classifier",
        "candidates": int(len(df)),
        "positive_candidates": int(
            (df["label"] == "positive").sum()
        ),
        "hard_negative_candidates": int(
            (
                df["label"]
                == "hard_negative"
            ).sum()
        ),
        "train_candidates": int(
            len(train_df)
        ),
        "validation_candidates": int(
            len(val_df)
        ),
        "train_scenes": int(
            train_df["global_id"].nunique()
        ),
        "validation_scenes": int(
            val_df["global_id"].nunique()
        ),
        "best_validation_f1": float(
            best_score
        ),
        "best_checkpoint": str(
            best_path
        ),
    }

    with open(
        args.output / "summary.json",
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
        "✅ V3 CANDIDATE CLASSIFIER COMPLETE"
    )
    print("=" * 70)

    print(
        "Best F1:",
        best_score,
    )

    print(
        "Best checkpoint:",
        best_path,
    )

    print(
        "Last checkpoint:",
        last_path,
    )

    print(
        "Training log:",
        log_path,
    )


if __name__ == "__main__":
    main()