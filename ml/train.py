from __future__ import annotations

import argparse
import csv
import os
import random
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml
from torch.utils.data import DataLoader

from src.datasets.oil_dataset import OilSegmentationDataset
from src.models.segmentation_model import build_model
from src.training.losses import V4SegmentationLoss


# ============================================================
# PS26143 — V4 SEGMENTATION TRAINING
# ============================================================


REPO = Path("/content/PS26143")
DRIVE_ROOT = Path("/content/drive/MyDrive/PS26143")


# ============================================================
# ARGUMENTS
# ============================================================


def parse_args():

    parser = argparse.ArgumentParser(
        description="PS26143 V4 segmentation training"
    )

    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to V4 YAML configuration.",
    )

    return parser.parse_args()


ARGS = parse_args()


# ============================================================
# CONFIGURATION PATH
# ============================================================


if ARGS.config is not None:

    CONFIG_FILE = Path(
        ARGS.config
    )

else:

    CONFIG_FILE = (
        REPO
        / "ml"
        / "configs"
        / "train_v4.yaml"
    )


# ============================================================
# DRIVE PATHS
# ============================================================


PROCESSED_ROOT = (
    DRIVE_ROOT
    / "data"
    / "processed"
)


TRAIN_MANIFEST = (
    PROCESSED_ROOT
    / "train"
    / "manifest.csv"
)


VAL_MANIFEST = (
    PROCESSED_ROOT
    / "val"
    / "manifest.csv"
)


CHECKPOINT_DIR = (
    DRIVE_ROOT
    / "checkpoints"
)


LOG_DIR = (
    DRIVE_ROOT
    / "logs"
)


CHECKPOINT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


LOG_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


# ============================================================
# CONFIG LOADING
# ============================================================


def load_config():

    if not CONFIG_FILE.exists():

        raise FileNotFoundError(
            "\n"
            "V4 CONFIGURATION ERROR\n"
            f"Configuration file does not exist:\n"
            f"  {CONFIG_FILE}\n"
            "\n"
            "Expected:\n"
            "  ml/configs/train_v4.yaml\n"
        )

    try:

        with CONFIG_FILE.open(
            "r",
            encoding="utf-8",
        ) as f:

            config = yaml.safe_load(f)

    except Exception as exc:

        raise RuntimeError(
            "\n"
            "V4 CONFIGURATION ERROR\n"
            f"Could not parse YAML:\n"
            f"  {CONFIG_FILE}\n"
            f"\nActual error:\n{exc}"
        ) from exc

    if not isinstance(config, dict):

        raise ValueError(
            "V4 configuration must contain "
            "a YAML mapping/object."
        )

    return config


CONFIG = load_config()


# ============================================================
# CONFIG VALIDATION
# ============================================================


def require_config(path, key):

    if key not in path:

        raise KeyError(
            f"V4 configuration missing required key: "
            f"{key}"
        )

    return path[key]


EXPERIMENT_NAME = str(
    require_config(
        CONFIG,
        "name",
    )
)


if EXPERIMENT_NAME != "oil-seg-v4":

    raise ValueError(
        "\n"
        "V4 SAFETY CHECK FAILED\n"
        f"Expected config name: oil-seg-v4\n"
        f"Found: {EXPERIMENT_NAME}\n"
        "\n"
        "This prevents accidentally running "
        "another experiment with V4 checkpoints."
    )


SEED = int(
    require_config(
        CONFIG,
        "seed",
    )
)


IMAGE_SIZE = int(
    CONFIG["data"]["image_size"]
)


BATCH_SIZE = int(
    CONFIG["training"]["batch_size"]
)


EPOCHS = int(
    CONFIG["training"]["epochs"]
)


LEARNING_RATE = float(
    CONFIG["training"]["learning_rate"]
)


WEIGHT_DECAY = float(
    CONFIG["training"]["weight_decay"]
)


MIXED_PRECISION = bool(
    CONFIG["training"].get(
        "mixed_precision",
        True,
    )
)


NUM_WORKERS = int(
    CONFIG["training"].get(
        "num_workers",
        2,
    )
)


THRESHOLD = float(
    CONFIG["evaluation"]["threshold"]
)


# ============================================================
# SCHEDULER
# ============================================================


SCHEDULER_CONFIG = CONFIG[
    "scheduler"
]


LR_PATIENCE = int(
    SCHEDULER_CONFIG["patience"]
)


LR_FACTOR = float(
    SCHEDULER_CONFIG["factor"]
)


MIN_LR = float(
    SCHEDULER_CONFIG.get(
        "min_lr",
        0.0,
    )
)


# ============================================================
# EARLY STOPPING
# ============================================================


EARLY_STOPPING_PATIENCE = int(
    CONFIG[
        "early_stopping"
    ]["patience"]
)


# ============================================================
# V4 LOSS CONFIGURATION
# ============================================================


LOSS_CONFIG = CONFIG[
    "loss"
]


LOSS_TYPE = str(
    LOSS_CONFIG.get(
        "type",
        "v4",
    )
)


if LOSS_TYPE != "v4":

    raise ValueError(
        "\n"
        "V4 SAFETY CHECK FAILED\n"
        f"Expected loss.type = v4\n"
        f"Found: {LOSS_TYPE}"
    )


TVERSKY_WEIGHT = float(
    LOSS_CONFIG.get(
        "tversky_weight",
        0.40,
    )
)


DICE_WEIGHT = float(
    LOSS_CONFIG.get(
        "dice_weight",
        0.25,
    )
)


FOCAL_WEIGHT = float(
    LOSS_CONFIG.get(
        "focal_weight",
        0.20,
    )
)


BOUNDARY_WEIGHT = float(
    LOSS_CONFIG.get(
        "boundary_weight",
        0.10,
    )
)


NEGATIVE_WEIGHT = float(
    LOSS_CONFIG.get(
        "negative_weight",
        0.05,
    )
)


TVERSKY_ALPHA = float(
    LOSS_CONFIG.get(
        "tversky_alpha",
        0.30,
    )
)


TVERSKY_BETA = float(
    LOSS_CONFIG.get(
        "tversky_beta",
        0.70,
    )
)


FOCAL_ALPHA = float(
    LOSS_CONFIG.get(
        "focal_alpha",
        0.60,
    )
)


FOCAL_GAMMA = float(
    LOSS_CONFIG.get(
        "focal_gamma",
        2.0,
    )
)


NEGATIVE_THRESHOLD = float(
    LOSS_CONFIG.get(
        "negative_threshold",
        0.30,
    )
)


# ============================================================
# V4 CHECKPOINTS
# ============================================================


BEST_CHECKPOINT = (
    CHECKPOINT_DIR
    / "oil_seg_v4_best.pt"
)


LAST_CHECKPOINT = (
    CHECKPOINT_DIR
    / "oil_seg_v4_last.pt"
)


LOG_FILE = (
    LOG_DIR
    / "oil_seg_v4_training.csv"
)


# ============================================================
# REPRODUCIBILITY
# ============================================================


def seed_everything(seed: int):

    random.seed(seed)

    np.random.seed(seed)

    torch.manual_seed(seed)

    if torch.cuda.is_available():

        torch.cuda.manual_seed_all(
            seed
        )

    torch.backends.cudnn.deterministic = True

    torch.backends.cudnn.benchmark = False


# ============================================================
# METRICS
# ============================================================


def dice_score(
    logits,
    targets,
    threshold=0.5,
    smooth=1.0,
):

    probabilities = torch.sigmoid(
        logits
    )

    predictions = (
        probabilities >= threshold
    ).float()

    predictions = (
        predictions
        .contiguous()
        .view(
            predictions.size(0),
            -1,
        )
    )

    targets = (
        targets
        .contiguous()
        .view(
            targets.size(0),
            -1,
        )
    )

    intersection = (
        predictions * targets
    ).sum(dim=1)

    dice = (
        2.0 * intersection
        + smooth
    ) / (
        predictions.sum(dim=1)
        + targets.sum(dim=1)
        + smooth
    )

    return dice.mean().item()


def iou_score(
    logits,
    targets,
    threshold=0.5,
    smooth=1.0,
):

    probabilities = torch.sigmoid(
        logits
    )

    predictions = (
        probabilities >= threshold
    ).float()

    predictions = (
        predictions
        .contiguous()
        .view(
            predictions.size(0),
            -1,
        )
    )

    targets = (
        targets
        .contiguous()
        .view(
            targets.size(0),
            -1,
        )
    )

    intersection = (
        predictions * targets
    ).sum(dim=1)

    union = (
        predictions.sum(dim=1)
        + targets.sum(dim=1)
        - intersection
    )

    iou = (
        intersection + smooth
    ) / (
        union + smooth
    )

    return iou.mean().item()


# ============================================================
# DATA
# ============================================================


def load_records(
    path: Path,
):

    if not path.exists():

        raise FileNotFoundError(
            "\n"
            "DATA ERROR\n"
            f"Manifest does not exist:\n"
            f"  {path}"
        )

    try:

        df = pd.read_csv(
            path
        )

    except Exception as exc:

        raise RuntimeError(
            f"Could not read manifest:\n"
            f"{path}\n"
            f"\nActual error:\n{exc}"
        ) from exc

    required = {
        "global_id",
        "dataset",
        "sample_id",
        "split",
        "image",
        "mask",
    }

    missing = (
        required
        - set(df.columns)
    )

    if missing:

        raise ValueError(
            "\n"
            "DATA MANIFEST ERROR\n"
            f"Manifest: {path}\n"
            f"Missing columns: "
            f"{sorted(missing)}"
        )

    return df.to_dict(
        "records"
    )


def verify_records(
    records,
    expected_split,
    expected_count,
):

    if len(records) != expected_count:

        raise ValueError(
            "\n"
            "DATA SPLIT SAFETY ERROR\n"
            f"Split: {expected_split}\n"
            f"Expected records: "
            f"{expected_count}\n"
            f"Found: {len(records)}"
        )

    for record in records:

        if record["split"] != expected_split:

            raise ValueError(
                "\n"
                "DATA SPLIT SAFETY ERROR\n"
                f"Expected split: "
                f"{expected_split}\n"
                f"Record: "
                f"{record['global_id']}\n"
                f"Found split: "
                f"{record['split']}"
            )

        image = Path(
            record["image"]
        )

        mask = Path(
            record["mask"]
        )

        if not image.exists():

            raise FileNotFoundError(
                "\n"
                "DATA FILE ERROR\n"
                f"Missing image:\n"
                f"  {image}\n"
                f"Scene: "
                f"{record['global_id']}"
            )

        if not mask.exists():

            raise FileNotFoundError(
                "\n"
                "DATA FILE ERROR\n"
                f"Missing mask:\n"
                f"  {mask}\n"
                f"Scene: "
                f"{record['global_id']}"
            )


# ============================================================
# TRAINING
# ============================================================


def train_one_epoch(
    model,
    loader,
    optimizer,
    criterion,
    scaler,
    device,
    epoch,
):

    model.train()

    running_loss = 0.0
    running_dice = 0.0
    running_iou = 0.0

    total = len(loader)

    if total == 0:

        raise RuntimeError(
            "Training DataLoader contains zero batches."
        )

    for step, batch in enumerate(
        loader,
        start=1,
    ):

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

        optimizer.zero_grad(
            set_to_none=True
        )

        with torch.autocast(
            device_type=device.type,
            dtype=torch.float16,
            enabled=(
                MIXED_PRECISION
                and device.type == "cuda"
            ),
        ):

            logits = model(
                images
            )

            loss = criterion(
                logits,
                masks,
            )

        if not torch.isfinite(
            loss
        ):

            raise RuntimeError(
                "\n"
                "V4 TRAINING NUMERICAL ERROR\n"
                f"Epoch: {epoch}\n"
                f"Batch: {step}\n"
                f"Loss: {loss.item()}\n"
                "\n"
                "The loss became NaN/Inf. "
                "Training has been stopped."
            )

        if scaler.is_enabled():

            scaler.scale(
                loss
            ).backward()

            scaler.step(
                optimizer
            )

            scaler.update()

        else:

            loss.backward()

            optimizer.step()

        loss_value = loss.item()

        running_loss += (
            loss_value
        )

        running_dice += dice_score(
            logits.detach(),
            masks,
            THRESHOLD,
        )

        running_iou += iou_score(
            logits.detach(),
            masks,
            THRESHOLD,
        )

        if (
            step == 1
            or step % 10 == 0
            or step == total
        ):

            print(
                f"    batch "
                f"{step:3d}/{total} "
                f"| loss "
                f"{loss_value:.6f}",
                flush=True,
            )

    return (
        running_loss / total,
        running_dice / total,
        running_iou / total,
    )


# ============================================================
# VALIDATION
# ============================================================


@torch.no_grad()
def validate(
    model,
    loader,
    criterion,
    device,
):

    model.eval()

    running_loss = 0.0
    running_dice = 0.0
    running_iou = 0.0

    total = len(loader)

    if total == 0:

        raise RuntimeError(
            "Validation DataLoader contains zero batches."
        )

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

        with torch.autocast(
            device_type=device.type,
            dtype=torch.float16,
            enabled=(
                MIXED_PRECISION
                and device.type == "cuda"
            ),
        ):

            logits = model(
                images
            )

            loss = criterion(
                logits,
                masks,
            )

        if not torch.isfinite(
            loss
        ):

            raise RuntimeError(
                "\n"
                "V4 VALIDATION NUMERICAL ERROR\n"
                f"Loss: {loss.item()}"
            )

        running_loss += (
            loss.item()
        )

        running_dice += dice_score(
            logits,
            masks,
            THRESHOLD,
        )

        running_iou += iou_score(
            logits,
            masks,
            THRESHOLD,
        )

    return (
        running_loss / total,
        running_dice / total,
        running_iou / total,
    )


# ============================================================
# CHECKPOINT
# ============================================================


def checkpoint_payload(
    model,
    optimizer,
    scheduler,
    scaler,
    epoch,
    best_val_dice,
    epochs_without_improvement,
):

    return {

        "experiment":
            EXPERIMENT_NAME,

        "version":
            "v4",

        "epoch":
            epoch,

        "model_state_dict":
            model.state_dict(),

        "optimizer_state_dict":
            optimizer.state_dict(),

        "scheduler_state_dict":
            scheduler.state_dict(),

        "scaler_state_dict":
            scaler.state_dict(),

        "best_val_dice":
            best_val_dice,

        "epochs_without_improvement":
            epochs_without_improvement,

        "seed":
            SEED,

        "architecture":
            CONFIG["model"]["architecture"],

        "encoder":
            CONFIG["model"]["encoder"],

        "encoder_weights":
            CONFIG["model"]["encoder_weights"],

        "in_channels":
            CONFIG["model"]["in_channels"],

        "classes":
            CONFIG["model"]["classes"],

        "image_size":
            IMAGE_SIZE,

        "batch_size":
            BATCH_SIZE,

        "learning_rate":
            LEARNING_RATE,

        "weight_decay":
            WEIGHT_DECAY,

        "loss_config":
            CONFIG["loss"],

        "scheduler_config":
            CONFIG["scheduler"],

        "threshold":
            THRESHOLD,

        "python_version":
            os.sys.version,

        "torch_version":
            torch.__version__,

        "cuda_device":
        (
            torch.cuda.get_device_name(0)
            if torch.cuda.is_available()
            else "cpu"
        ),

        "python_random_state":
            random.getstate(),

        "numpy_random_state":
            np.random.get_state(),

        "torch_random_state":
            torch.get_rng_state(),

        "torch_cuda_random_state":
        (
            torch.cuda.get_rng_state_all()
            if torch.cuda.is_available()
            else None
        ),
    }


def atomic_torch_save(
    payload,
    path: Path,
):

    temporary = path.with_suffix(
        path.suffix + ".tmp"
    )

    torch.save(
        payload,
        temporary,
    )

    os.replace(
        temporary,
        path,
    )


def save_checkpoint(
    path,
    model,
    optimizer,
    scheduler,
    scaler,
    epoch,
    best_val_dice,
    epochs_without_improvement,
):

    payload = checkpoint_payload(
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        scaler=scaler,
        epoch=epoch,
        best_val_dice=best_val_dice,
        epochs_without_improvement=
            epochs_without_improvement,
    )

    atomic_torch_save(
        payload,
        path,
    )


# ============================================================
# RESUME
# ============================================================


def restore_checkpoint(
    path,
    model,
    optimizer,
    scheduler,
    scaler,
    device,
):

    print()
    print("=" * 70)
    print("RESUMING V4 TRAINING")
    print("=" * 70)

    try:

        checkpoint = torch.load(
            path,
            map_location=device,
            weights_only=False,
        )

    except Exception as exc:

        raise RuntimeError(
            "\n"
            "V4 CHECKPOINT LOAD ERROR\n"
            f"Checkpoint:\n{path}\n"
            f"\nActual error:\n{exc}"
        ) from exc

    if checkpoint.get(
        "version"
    ) != "v4":

        raise RuntimeError(
            "\n"
            "V4 CHECKPOINT SAFETY ERROR\n"
            f"Checkpoint: {path}\n"
            f"Found version: "
            f"{checkpoint.get('version')}\n"
            "Expected version: v4\n"
            "\n"
            "Refusing to resume from V1/V2/V3 "
            "or an unrelated experiment."
        )

    if checkpoint.get(
        "experiment"
    ) != EXPERIMENT_NAME:

        raise RuntimeError(
            "\n"
            "V4 CHECKPOINT EXPERIMENT MISMATCH\n"
            f"Expected: {EXPERIMENT_NAME}\n"
            f"Found: "
            f"{checkpoint.get('experiment')}"
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

    scheduler.load_state_dict(
        checkpoint[
            "scheduler_state_dict"
        ]
    )

    if "scaler_state_dict" in checkpoint:

        scaler.load_state_dict(
            checkpoint[
                "scaler_state_dict"
            ]
        )

    if "python_random_state" in checkpoint:

        random.setstate(
            checkpoint[
                "python_random_state"
            ]
        )

    if "numpy_random_state" in checkpoint:

        np.random.set_state(
            checkpoint[
                "numpy_random_state"
            ]
        )

    if "torch_random_state" in checkpoint:

        torch.set_rng_state(
            checkpoint[
                "torch_random_state"
            ]
        )

    cuda_state = checkpoint.get(
        "torch_cuda_random_state"
    )

    if (
        torch.cuda.is_available()
        and cuda_state is not None
    ):

        torch.cuda.set_rng_state_all(
            cuda_state
        )

    previous_epoch = int(
        checkpoint["epoch"]
    )

    best_val_dice = float(
        checkpoint[
            "best_val_dice"
        ]
    )

    epochs_without_improvement = int(
        checkpoint.get(
            "epochs_without_improvement",
            0,
        )
    )

    print(
        "Checkpoint epoch     :",
        previous_epoch,
    )

    print(
        "Starting epoch       :",
        previous_epoch + 1,
    )

    print(
        "Best validation Dice :",
        f"{best_val_dice:.6f}",
    )

    print(
        "Checkpoint           :",
        path,
    )

    print(
        "V4 RESUME SUCCESSFUL"
    )

    return (
        previous_epoch + 1,
        best_val_dice,
        epochs_without_improvement,
    )


# ============================================================
# LOGGING
# ============================================================


LOG_COLUMNS = [
    "epoch",
    "train_loss",
    "train_dice",
    "train_iou",
    "val_loss",
    "val_dice",
    "val_iou",
    "learning_rate",
    "epoch_seconds",
]


def prepare_log_file(
    resuming,
):

    if (
        resuming
        and LOG_FILE.exists()
    ):

        return

    with LOG_FILE.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as f:

        writer = csv.writer(f)

        writer.writerow(
            LOG_COLUMNS
        )


def append_log(
    epoch,
    train_loss,
    train_dice,
    train_iou,
    val_loss,
    val_dice,
    val_iou,
    learning_rate,
    epoch_seconds,
):

    with LOG_FILE.open(
        "a",
        newline="",
        encoding="utf-8",
    ) as f:

        writer = csv.writer(f)

        writer.writerow(
            [
                epoch,
                train_loss,
                train_dice,
                train_iou,
                val_loss,
                val_dice,
                val_iou,
                learning_rate,
                epoch_seconds,
            ]
        )


# ============================================================
# MAIN
# ============================================================


def main():

    seed_everything(
        SEED
    )

    print("=" * 70)
    print("PS26143 — V4 SEGMENTATION TRAINING")
    print("=" * 70)

    print()
    print(
        "Experiment :",
        EXPERIMENT_NAME,
    )

    print(
        "Config     :",
        CONFIG_FILE,
    )

    print(
        "PyTorch    :",
        torch.__version__,
    )

    print(
        "CUDA       :",
        torch.cuda.is_available(),
    )

    if not torch.cuda.is_available():

        raise RuntimeError(
            "\n"
            "V4 TRAINING ABORTED\n"
            "CUDA GPU is required."
        )

    device = torch.device(
        "cuda"
    )

    print(
        "GPU        :",
        torch.cuda.get_device_name(0),
    )

    print(
        "VRAM       :",
        round(
            torch.cuda.get_device_properties(
                0
            ).total_memory
            / (1024 ** 3),
            2,
        ),
        "GiB",
    )

    # --------------------------------------------------------
    # CONFIG SUMMARY
    # --------------------------------------------------------

    print()
    print("=" * 70)
    print("V4 CONFIGURATION")
    print("=" * 70)

    print(
        "Architecture     :",
        CONFIG["model"]["architecture"],
    )

    print(
        "Encoder          :",
        CONFIG["model"]["encoder"],
    )

    print(
        "Input channels   :",
        CONFIG["model"]["in_channels"],
    )

    print(
        "Image size       :",
        IMAGE_SIZE,
    )

    print(
        "Batch size       :",
        BATCH_SIZE,
    )

    print(
        "Epochs           :",
        EPOCHS,
    )

    print(
        "Learning rate    :",
        LEARNING_RATE,
    )

    print(
        "Weight decay     :",
        WEIGHT_DECAY,
    )

    print(
        "AMP              :",
        MIXED_PRECISION,
    )

    # --------------------------------------------------------
    # LOSS SUMMARY
    # --------------------------------------------------------

    print()
    print("=" * 70)
    print("V4 LOSS")
    print("=" * 70)

    print(
        "Loss type        : V4"
    )

    print(
        "Tversky weight   :",
        TVERSKY_WEIGHT,
    )

    print(
        "Dice weight      :",
        DICE_WEIGHT,
    )

    print(
        "Focal weight     :",
        FOCAL_WEIGHT,
    )

    print(
        "Boundary weight  :",
        BOUNDARY_WEIGHT,
    )

    print(
        "Negative weight  :",
        NEGATIVE_WEIGHT,
    )

    print(
        "Tversky alpha    :",
        TVERSKY_ALPHA,
    )

    print(
        "Tversky beta     :",
        TVERSKY_BETA,
    )

    print(
        "Focal alpha      :",
        FOCAL_ALPHA,
    )

    print(
        "Focal gamma      :",
        FOCAL_GAMMA,
    )

    print(
        "Negative thresh  :",
        NEGATIVE_THRESHOLD,
    )

    # --------------------------------------------------------
    # DATA SAFETY
    # --------------------------------------------------------

    print()
    print("=" * 70)
    print("DATA")
    print("=" * 70)

    print(
        "Train manifest:",
        TRAIN_MANIFEST,
    )

    print(
        "Val manifest  :",
        VAL_MANIFEST,
    )

    train_records = load_records(
        TRAIN_MANIFEST
    )

    val_records = load_records(
        VAL_MANIFEST
    )

    verify_records(
        train_records,
        "train",
        630,
    )

    verify_records(
        val_records,
        "val",
        135,
    )

    print(
        "Training samples  :",
        len(train_records),
    )

    print(
        "Validation samples:",
        len(val_records),
    )

    print(
        "Test set          : NOT USED"
    )

    # --------------------------------------------------------
    # DATASETS
    # --------------------------------------------------------

    train_dataset = OilSegmentationDataset(
        train_records,
        augment=True,
    )

    val_dataset = OilSegmentationDataset(
        val_records,
        augment=False,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=True,
        persistent_workers=(
            NUM_WORKERS > 0
        ),
        drop_last=False,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=True,
        persistent_workers=(
            NUM_WORKERS > 0
        ),
        drop_last=False,
    )

    # --------------------------------------------------------
    # MODEL
    # --------------------------------------------------------

    print()
    print("=" * 70)
    print("BUILDING V4 MODEL")
    print("=" * 70)

    model = build_model(
        architecture=
            CONFIG["model"]["architecture"],
        encoder=
            CONFIG["model"]["encoder"],
        encoder_weights=
            CONFIG["model"]["encoder_weights"],
        in_channels=
            CONFIG["model"]["in_channels"],
        classes=
            CONFIG["model"]["classes"],
    )

    model = model.to(
        device
    )

    parameter_count = sum(
        p.numel()
        for p in model.parameters()
    )

    print(
        "Architecture :",
        CONFIG["model"]["architecture"],
    )

    print(
        "Encoder      :",
        CONFIG["model"]["encoder"],
    )

    print(
        "Input        :",
        CONFIG["model"]["in_channels"],
        "channels",
    )

    print(
        "Output       :",
        CONFIG["model"]["classes"],
        "channel",
    )

    print(
        "Parameters   :",
        f"{parameter_count:,}",
    )

    # --------------------------------------------------------
    # V4 LOSS
    # --------------------------------------------------------

    criterion = V4SegmentationLoss(
        tversky_weight=
            TVERSKY_WEIGHT,

        dice_weight=
            DICE_WEIGHT,

        focal_weight=
            FOCAL_WEIGHT,

        boundary_weight=
            BOUNDARY_WEIGHT,

        negative_weight=
            NEGATIVE_WEIGHT,

        tversky_alpha=
            TVERSKY_ALPHA,

        tversky_beta=
            TVERSKY_BETA,

        focal_alpha=
            FOCAL_ALPHA,

        focal_gamma=
            FOCAL_GAMMA,

        negative_threshold=
            NEGATIVE_THRESHOLD,
    )

    # --------------------------------------------------------
    # OPTIMIZER
    # --------------------------------------------------------

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
    )

    # --------------------------------------------------------
    # SCHEDULER
    # --------------------------------------------------------

    scheduler = (
        torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode=SCHEDULER_CONFIG["mode"],
            patience=LR_PATIENCE,
            factor=LR_FACTOR,
            min_lr=MIN_LR,
        )
    )

    # --------------------------------------------------------
    # AMP
    # --------------------------------------------------------

    scaler = torch.amp.GradScaler(
        "cuda",
        enabled=MIXED_PRECISION,
    )

    # --------------------------------------------------------
    # RESUME
    # --------------------------------------------------------

    start_epoch = 1

    best_val_dice = -1.0

    epochs_without_improvement = 0

    resuming = (
        LAST_CHECKPOINT.exists()
    )

    if resuming:

        (
            start_epoch,
            best_val_dice,
            epochs_without_improvement,
        ) = restore_checkpoint(
            LAST_CHECKPOINT,
            model,
            optimizer,
            scheduler,
            scaler,
            device,
        )

    prepare_log_file(
        resuming=resuming
    )

    # --------------------------------------------------------
    # TRAINING HEADER
    # --------------------------------------------------------

    print()
    print("=" * 70)
    print("V4 TRAINING")
    print("=" * 70)

    print(
        "Epochs              :",
        EPOCHS,
    )

    print(
        "Batch size          :",
        BATCH_SIZE,
    )

    print(
        "Learning rate       :",
        LEARNING_RATE,
    )

    print(
        "Weight decay        :",
        WEIGHT_DECAY,
    )

    print(
        "Mixed precision     :",
        MIXED_PRECISION,
    )

    print(
        "Checkpoint interval : every epoch"
    )

    print(
        "Resume enabled      : True"
    )

    print(
        "Best checkpoint     :",
        BEST_CHECKPOINT,
    )

    print(
        "Last checkpoint     :",
        LAST_CHECKPOINT,
    )

    print(
        "Training log        :",
        LOG_FILE,
    )

    # --------------------------------------------------------
    # ALREADY COMPLETE
    # --------------------------------------------------------

    if start_epoch > EPOCHS:

        print()
        print(
            "V4 training is already complete."
        )

        print(
            "Best validation Dice:",
            f"{best_val_dice:.6f}",
        )

        return

    # --------------------------------------------------------
    # EPOCH LOOP
    # --------------------------------------------------------

    for epoch in range(
        start_epoch,
        EPOCHS + 1,
    ):

        epoch_start = time.time()

        print()
        print(
            f"Epoch {epoch:02d}/{EPOCHS}"
        )

        print("-" * 70)

        (
            train_loss,
            train_dice,
            train_iou,
        ) = train_one_epoch(
            model=model,
            loader=train_loader,
            optimizer=optimizer,
            criterion=criterion,
            scaler=scaler,
            device=device,
            epoch=epoch,
        )

        (
            val_loss,
            val_dice,
            val_iou,
        ) = validate(
            model=model,
            loader=val_loader,
            criterion=criterion,
            device=device,
        )

        scheduler.step(
            val_dice
        )

        current_lr = (
            optimizer.param_groups[0][
                "lr"
            ]
        )

        elapsed = (
            time.time()
            - epoch_start
        )

        print()
        print(
            f"Epoch {epoch:02d} RESULT"
        )

        print(
            f"  Train loss : "
            f"{train_loss:.6f}"
        )

        print(
            f"  Train Dice : "
            f"{train_dice:.6f}"
        )

        print(
            f"  Train IoU  : "
            f"{train_iou:.6f}"
        )

        print(
            f"  Val loss   : "
            f"{val_loss:.6f}"
        )

        print(
            f"  Val Dice   : "
            f"{val_dice:.6f}"
        )

        print(
            f"  Val IoU    : "
            f"{val_iou:.6f}"
        )

        print(
            f"  LR         : "
            f"{current_lr:.8f}"
        )

        print(
            f"  Time       : "
            f"{elapsed:.1f}s"
        )

        # ----------------------------------------------------
        # LOG
        # ----------------------------------------------------

        append_log(
            epoch=epoch,
            train_loss=train_loss,
            train_dice=train_dice,
            train_iou=train_iou,
            val_loss=val_loss,
            val_dice=val_dice,
            val_iou=val_iou,
            learning_rate=current_lr,
            epoch_seconds=elapsed,
        )

        # ----------------------------------------------------
        # BEST CHECKPOINT
        # ----------------------------------------------------

        if val_dice > best_val_dice:

            best_val_dice = val_dice

            epochs_without_improvement = 0

            save_checkpoint(
                BEST_CHECKPOINT,
                model,
                optimizer,
                scheduler,
                scaler,
                epoch,
                best_val_dice,
                epochs_without_improvement,
            )

            print()
            print(
                "★ NEW V4 BEST CHECKPOINT"
            )

            print(
                "  Val Dice:",
                f"{best_val_dice:.6f}",
            )

        else:

            epochs_without_improvement += 1

        # ----------------------------------------------------
        # LAST CHECKPOINT
        # ----------------------------------------------------

        save_checkpoint(
            LAST_CHECKPOINT,
            model,
            optimizer,
            scheduler,
            scaler,
            epoch,
            best_val_dice,
            epochs_without_improvement,
        )

        print()
        print(
            "✓ V4 LAST CHECKPOINT SAVED"
        )

        # ----------------------------------------------------
        # EARLY STOPPING
        # ----------------------------------------------------

        if (
            epochs_without_improvement
            >= EARLY_STOPPING_PATIENCE
        ):

            print()
            print(
                "V4 EARLY STOPPING"
            )

            print(
                "No validation Dice improvement "
                f"for "
                f"{EARLY_STOPPING_PATIENCE} "
                "epochs."
            )

            break

    # --------------------------------------------------------
    # COMPLETE
    # --------------------------------------------------------

    print()
    print("=" * 70)
    print("V4 TRAINING COMPLETE")
    print("=" * 70)

    print(
        "Best validation Dice:",
        f"{best_val_dice:.6f}",
    )

    print()
    print(
        "BEST:",
        BEST_CHECKPOINT,
    )

    print(
        "LAST:",
        LAST_CHECKPOINT,
    )

    print(
        "LOG :",
        LOG_FILE,
    )


# ============================================================
# ENTRY POINT
# ============================================================


if __name__ == "__main__":

    main()