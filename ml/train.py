
from pathlib import Path
import csv
import os
import random
import time

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from src.datasets.oil_dataset import OilSegmentationDataset
from src.models.segmentation_model import build_model
from src.training.losses import BCEDiceLoss


# ============================================================
# PS26143 — RESUMABLE BASELINE SEGMENTATION TRAINING
# ============================================================

SEED = 42

REPO = Path("/content/PS26143")
DRIVE_ROOT = Path("/content/drive/MyDrive/PS26143")

PROCESSED_ROOT = DRIVE_ROOT / "data/processed"

TRAIN_MANIFEST = PROCESSED_ROOT / "train/manifest.csv"
VAL_MANIFEST = PROCESSED_ROOT / "val/manifest.csv"

CHECKPOINT_DIR = DRIVE_ROOT / "checkpoints"
LOG_DIR = DRIVE_ROOT / "logs"

CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
LOG_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# CONFIG
# ============================================================

IMAGE_SIZE = 512

BATCH_SIZE = 16
EPOCHS = 50

LEARNING_RATE = 1e-4
WEIGHT_DECAY = 1e-4

BCE_WEIGHT = 0.5
DICE_WEIGHT = 0.5

LR_PATIENCE = 3
EARLY_STOPPING_PATIENCE = 5

NUM_WORKERS = 2

THRESHOLD = 0.5

# IMPORTANT:
# Automatically continue from the last checkpoint if one exists.
RESUME = True


# ============================================================
# CHECKPOINT PATHS
# ============================================================

BEST_CHECKPOINT = CHECKPOINT_DIR / "oil_seg_v1_best.pt"
LAST_CHECKPOINT = CHECKPOINT_DIR / "oil_seg_v1_last.pt"

LOG_FILE = LOG_DIR / "oil_seg_v1_training.csv"


# ============================================================
# REPRODUCIBILITY
# ============================================================

def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ============================================================
# METRICS
# ============================================================

def dice_score(logits, targets, threshold=0.5, smooth=1.0):
    probabilities = torch.sigmoid(logits)
    predictions = (probabilities >= threshold).float()

    predictions = predictions.contiguous().view(
        predictions.size(0), -1
    )

    targets = targets.contiguous().view(
        targets.size(0), -1
    )

    intersection = (predictions * targets).sum(dim=1)

    dice = (
        2.0 * intersection + smooth
    ) / (
        predictions.sum(dim=1)
        + targets.sum(dim=1)
        + smooth
    )

    return dice.mean().item()


def iou_score(logits, targets, threshold=0.5, smooth=1.0):
    probabilities = torch.sigmoid(logits)
    predictions = (probabilities >= threshold).float()

    predictions = predictions.contiguous().view(
        predictions.size(0), -1
    )

    targets = targets.contiguous().view(
        targets.size(0), -1
    )

    intersection = (predictions * targets).sum(dim=1)

    union = (
        predictions.sum(dim=1)
        + targets.sum(dim=1)
        - intersection
    )

    iou = (intersection + smooth) / (union + smooth)

    return iou.mean().item()


# ============================================================
# DATA
# ============================================================

def load_records(path):
    df = pd.read_csv(path)

    required = {
        "global_id",
        "dataset",
        "sample_id",
        "split",
        "image",
        "mask",
    }

    missing = required - set(df.columns)

    if missing:
        raise ValueError(
            f"Manifest missing columns: {sorted(missing)}"
        )

    records = df.to_dict("records")

    return records


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

    for step, batch in enumerate(loader, start=1):

        images = batch["image"].to(
            device,
            non_blocking=True,
        )

        masks = batch["mask"].to(
            device,
            non_blocking=True,
        )

        optimizer.zero_grad(set_to_none=True)

        with torch.autocast(
            device_type=device.type,
            dtype=torch.float16,
            enabled=device.type == "cuda",
        ):
            logits = model(images)

            loss = criterion(
                logits,
                masks,
            )

        if scaler.is_enabled():

            scaler.scale(loss).backward()

            scaler.step(optimizer)

            scaler.update()

        else:

            loss.backward()

            optimizer.step()

        running_loss += loss.item()

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
                f"    batch {step:3d}/{total} "
                f"| loss {loss.item():.4f}",
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

    for batch in loader:

        images = batch["image"].to(
            device,
            non_blocking=True,
        )

        masks = batch["mask"].to(
            device,
            non_blocking=True,
        )

        with torch.autocast(
            device_type=device.type,
            dtype=torch.float16,
            enabled=device.type == "cuda",
        ):
            logits = model(images)

            loss = criterion(
                logits,
                masks,
            )

        running_loss += loss.item()

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
        "epoch": epoch,

        "model_state_dict": model.state_dict(),

        "optimizer_state_dict": optimizer.state_dict(),

        "scheduler_state_dict": scheduler.state_dict(),

        "scaler_state_dict": scaler.state_dict(),

        "best_val_dice": best_val_dice,

        "epochs_without_improvement":
            epochs_without_improvement,

        "seed": SEED,

        "architecture": "unet",

        "encoder": "resnet34",

        "encoder_weights": "imagenet",

        "in_channels": 2,

        "classes": 1,

        "image_size": IMAGE_SIZE,

        "batch_size": BATCH_SIZE,

        "learning_rate": LEARNING_RATE,

        "weight_decay": WEIGHT_DECAY,

        "threshold": THRESHOLD,

        "python_version": os.sys.version,

        "torch_version": torch.__version__,

        "cuda_device": (
            torch.cuda.get_device_name(0)
            if torch.cuda.is_available()
            else "cpu"
        ),

        "python_random_state": random.getstate(),

        "numpy_random_state": np.random.get_state(),

        "torch_random_state": torch.get_rng_state(),

        "torch_cuda_random_state": (
            torch.cuda.get_rng_state_all()
            if torch.cuda.is_available()
            else None
        ),
    }


def atomic_torch_save(payload, path):
    """
    Save checkpoint to a temporary file first,
    then atomically replace the destination.

    This reduces the risk of leaving a corrupt checkpoint
    if the runtime dies while writing.
    """

    temporary = path.with_suffix(
        path.suffix + ".tmp"
    )

    torch.save(payload, temporary)

    os.replace(temporary, path)


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
    print("RESUMING FROM CHECKPOINT")
    print("=" * 70)

    checkpoint = torch.load(
        path,
        map_location=device,
        weights_only=False,
    )

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    optimizer.load_state_dict(
        checkpoint["optimizer_state_dict"]
    )

    scheduler.load_state_dict(
        checkpoint["scheduler_state_dict"]
    )

    if "scaler_state_dict" in checkpoint:
        scaler.load_state_dict(
            checkpoint["scaler_state_dict"]
        )

    # Restore RNG states where available.
    if "python_random_state" in checkpoint:
        random.setstate(
            checkpoint["python_random_state"]
        )

    if "numpy_random_state" in checkpoint:
        np.random.set_state(
            checkpoint["numpy_random_state"]
        )

    if "torch_random_state" in checkpoint:
        torch.set_rng_state(
            checkpoint["torch_random_state"]
        )

    if (
        torch.cuda.is_available()
        and checkpoint.get(
            "torch_cuda_random_state"
        ) is not None
    ):
        torch.cuda.set_rng_state_all(
            checkpoint["torch_cuda_random_state"]
        )

    previous_epoch = int(
        checkpoint["epoch"]
    )

    best_val_dice = float(
        checkpoint["best_val_dice"]
    )

    epochs_without_improvement = int(
        checkpoint.get(
            "epochs_without_improvement",
            0,
        )
    )

    start_epoch = previous_epoch + 1

    print(
        "Checkpoint epoch        :",
        previous_epoch,
    )

    print(
        "Starting epoch          :",
        start_epoch,
    )

    print(
        "Best validation Dice    :",
        f"{best_val_dice:.5f}",
    )

    print(
        "No-improvement epochs   :",
        epochs_without_improvement,
    )

    print(
        "Checkpoint              :",
        path,
    )

    print(
        "RESUME SUCCESSFUL"
    )

    return (
        start_epoch,
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


def prepare_log_file(resuming):
    """
    Do not erase the training history when resuming.
    """

    if resuming and LOG_FILE.exists():
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

    seed_everything(SEED)

    print("=" * 70)
    print("PS26143 — BASELINE SEGMENTATION TRAINING")
    print("=" * 70)

    print()
    print("PyTorch :", torch.__version__)
    print("CUDA    :", torch.cuda.is_available())

    if torch.cuda.is_available():

        device = torch.device("cuda")

        print(
            "GPU     :",
            torch.cuda.get_device_name(0),
        )

        print(
            "VRAM    :",
            round(
                torch.cuda.get_device_properties(0)
                .total_memory
                / (1024 ** 3),
                2,
            ),
            "GiB",
        )

    else:

        device = torch.device("cpu")

        print(
            "WARNING: CUDA unavailable."
        )

    print()
    print(
        "Processed root:",
        PROCESSED_ROOT,
    )

    print(
        "Train manifest:",
        TRAIN_MANIFEST,
    )

    print(
        "Val manifest  :",
        VAL_MANIFEST,
    )

    if not TRAIN_MANIFEST.exists():
        raise FileNotFoundError(
            f"Training manifest missing: "
            f"{TRAIN_MANIFEST}"
        )

    if not VAL_MANIFEST.exists():
        raise FileNotFoundError(
            f"Validation manifest missing: "
            f"{VAL_MANIFEST}"
        )

    train_records = load_records(
        TRAIN_MANIFEST
    )

    val_records = load_records(
        VAL_MANIFEST
    )

    print()
    print(
        "Training samples  :",
        len(train_records),
    )

    print(
        "Validation samples:",
        len(val_records),
    )

    if len(train_records) != 630:
        raise ValueError(
            f"Expected 630 training samples, "
            f"found {len(train_records)}"
        )

    if len(val_records) != 135:
        raise ValueError(
            f"Expected 135 validation samples, "
            f"found {len(val_records)}"
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
        pin_memory=device.type == "cuda",
        persistent_workers=NUM_WORKERS > 0,
        drop_last=False,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=device.type == "cuda",
        persistent_workers=NUM_WORKERS > 0,
        drop_last=False,
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
        encoder_weights="imagenet",
        in_channels=2,
        classes=1,
    )

    model = model.to(device)

    parameter_count = sum(
        p.numel()
        for p in model.parameters()
    )

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

    print(
        "Parameters   :",
        f"{parameter_count:,}",
    )

    # --------------------------------------------------------
    # LOSS
    # --------------------------------------------------------

    criterion = BCEDiceLoss(
        bce_weight=BCE_WEIGHT,
        dice_weight=DICE_WEIGHT,
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

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="max",
        patience=LR_PATIENCE,
        factor=0.5,
    )

    # --------------------------------------------------------
    # AMP
    # --------------------------------------------------------

    scaler = torch.amp.GradScaler(
        "cuda",
        enabled=device.type == "cuda",
    )

    # --------------------------------------------------------
    # RESUME STATE
    # --------------------------------------------------------

    start_epoch = 1
    best_val_dice = -1.0
    epochs_without_improvement = 0

    resuming = (
        RESUME
        and LAST_CHECKPOINT.exists()
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
    print("TRAINING")
    print("=" * 70)

    print()
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
        "Mixed precision     :",
        device.type == "cuda",
    )

    print(
        "Checkpoint interval :",
        "every epoch",
    )

    print(
        "Resume enabled      :",
        RESUME,
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
    # IF ALREADY COMPLETE
    # --------------------------------------------------------

    if start_epoch > EPOCHS:

        print()
        print(
            "Requested training epochs "
            "already completed."
        )

        print(
            "Best validation Dice:",
            f"{best_val_dice:.5f}",
        )

        return

    # --------------------------------------------------------
    # EPOCH LOOP
    # --------------------------------------------------------

    for epoch in range(
        start_epoch,
        EPOCHS + 1,
    ):

        start = time.time()

        print()
        print(
            f"Epoch {epoch:02d}/{EPOCHS}"
        )

        print("-" * 70)

        train_loss, train_dice, train_iou = (
            train_one_epoch(
                model,
                train_loader,
                optimizer,
                criterion,
                scaler,
                device,
                epoch,
            )
        )

        val_loss, val_dice, val_iou = (
            validate(
                model,
                val_loader,
                criterion,
                device,
            )
        )

        scheduler.step(
            val_dice
        )

        lr = optimizer.param_groups[0]["lr"]

        elapsed = time.time() - start

        print()
        print(
            f"Epoch {epoch:02d} RESULT"
        )

        print(
            f"  Train loss : {train_loss:.5f}"
        )

        print(
            f"  Train Dice : {train_dice:.5f}"
        )

        print(
            f"  Train IoU  : {train_iou:.5f}"
        )

        print(
            f"  Val loss   : {val_loss:.5f}"
        )

        print(
            f"  Val Dice   : {val_dice:.5f}"
        )

        print(
            f"  Val IoU    : {val_iou:.5f}"
        )

        print(
            f"  LR         : {lr:.7f}"
        )

        print(
            f"  Time       : {elapsed:.1f}s"
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
            learning_rate=lr,
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
                "  ★ NEW BEST CHECKPOINT"
            )

            print(
                f"  Val Dice: "
                f"{best_val_dice:.5f}"
            )

        else:

            epochs_without_improvement += 1

        # ----------------------------------------------------
        # LAST CHECKPOINT
        #
        # ALWAYS saved.
        # This is what makes Colab recovery possible.
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
            "  ✓ LAST CHECKPOINT SAVED"
        )

        print(
            "  ",
            LAST_CHECKPOINT,
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
                "EARLY STOPPING"
            )

            print(
                f"No validation Dice improvement "
                f"for {EARLY_STOPPING_PATIENCE} epochs."
            )

            break

    # --------------------------------------------------------
    # COMPLETE
    # --------------------------------------------------------

    print()
    print("=" * 70)
    print("TRAINING COMPLETE")
    print("=" * 70)

    print()
    print(
        f"Best validation Dice: "
        f"{best_val_dice:.5f}"
    )

    print()
    print(
        "Best checkpoint:"
    )

    print(
        BEST_CHECKPOINT
    )

    print()
    print(
        "Last checkpoint:"
    )

    print(
        LAST_CHECKPOINT
    )

    print()
    print(
        "Training log:"
    )

    print(
        LOG_FILE
    )


if __name__ == "__main__":
    main()

