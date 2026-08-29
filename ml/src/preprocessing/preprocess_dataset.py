import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import rasterio
from PIL import Image


# ============================================================
# PS26143 — DATASET PREPROCESSING
# ============================================================

DATA_ROOT = Path("/content/drive/MyDrive/PS26143")

DEFAULT_WORKSPACE = Path("/content/ps26143_workspace")


# ============================================================
# SELECTED-900 SOURCE DIRECTORIES
# ============================================================

IMAGE_ROOTS = {
    "oil":
        DATA_ROOT / "data/extracted/selected_900/oil/images/Oil",

    "lookalike":
        DATA_ROOT / "data/extracted/selected_900/lookalike/images/Lookalike",

    "no_oil":
        DATA_ROOT / "data/extracted/selected_900/no_oil/images/No_oil",
}


MASK_ROOTS = {
    "oil":
        DATA_ROOT / "data/extracted/selected_900/oil/masks/Mask_oil",

    "lookalike":
        DATA_ROOT / "data/extracted/selected_900/lookalike/masks/Mask_lookalike",

    "no_oil":
        DATA_ROOT / "data/extracted/selected_900/no_oil/masks/Mask_no_oil",
}


# ============================================================
# SAR NORMALIZATION
# ============================================================

def normalize_sar(image):
    """
    Deterministic per-scene percentile normalization.

    Input:
        [2, H, W], float32 SAR values.

    Output:
        [2, H, W], float32 approximately [0, 1].

    VV and VH are normalized independently using
    the 1st and 99th percentiles.
    """

    image = image.astype(np.float32)

    output = np.zeros_like(
        image,
        dtype=np.float32,
    )

    for band in range(image.shape[0]):

        x = image[band]

        finite = np.isfinite(x)

        if not finite.any():
            continue

        valid = x[finite]

        low = np.percentile(
            valid,
            1.0,
        )

        high = np.percentile(
            valid,
            99.0,
        )

        if high <= low:
            output[band] = 0.0
            continue

        x = np.nan_to_num(
            x,
            nan=low,
            posinf=high,
            neginf=low,
        )

        x = np.clip(
            x,
            low,
            high,
        )

        output[band] = (
            (x - low)
            / (high - low)
        )

    return output


# ============================================================
# IMAGE RESIZING
# ============================================================

def resize_image(image, size):
    """
    Resize each SAR band using bilinear interpolation.

    Input:
        [2, H, W]

    Output:
        [2, size, size]
    """

    bands = []

    for band in image:

        pil = Image.fromarray(
            band.astype(np.float32),
            mode="F",
        )

        pil = pil.resize(
            (size, size),
            Image.Resampling.BILINEAR,
        )

        bands.append(
            np.asarray(
                pil,
                dtype=np.float32,
            )
        )

    return np.stack(
        bands,
        axis=0,
    )


# ============================================================
# MASK RESIZING
# ============================================================

def resize_mask(mask, size):
    """
    Resize binary mask using nearest-neighbor interpolation.

    Output:
        [size, size], float32 values {0,1}
    """

    pil = Image.fromarray(
        mask.astype(np.uint8),
        mode="L",
    )

    pil = pil.resize(
        (size, size),
        Image.Resampling.NEAREST,
    )

    mask = np.asarray(
        pil,
        dtype=np.uint8,
    )

    return (
        mask > 0
    ).astype(np.float32)


# ============================================================
# TIFF READERS
# ============================================================

def read_image(path):
    """
    Read SAR image.

    Expected:
        exactly 2 bands.
    """

    with rasterio.open(path) as src:

        image = src.read()

    if image.ndim != 3:

        raise ValueError(
            f"Expected [bands,H,W], "
            f"got {image.shape}: {path}"
        )

    if image.shape[0] != 2:

        raise ValueError(
            f"Expected exactly 2 bands, "
            f"got {image.shape[0]}: {path}"
        )

    return image


def read_mask(path):

    with rasterio.open(path) as src:

        mask = src.read(1)

    return mask


# ============================================================
# FILE LOCATION
# ============================================================

def locate(root, sample_id):

    path = root / f"{int(sample_id):05d}.tif"

    if not path.exists():

        raise FileNotFoundError(
            f"Missing TIFF: {path}"
        )

    return path


# ============================================================
# PROCESS ONE SPLIT
# ============================================================

def process_split(
    manifest_path,
    split,
    workspace,
    image_size,
):

    df = pd.read_csv(
        manifest_path
    )

    if "split" in df.columns:

        df = df[
            df["split"] == split
        ].copy()

    output = (
        workspace
        / "processed"
        / split
    )

    image_out = (
        output
        / "images"
    )

    mask_out = (
        output
        / "masks"
    )

    image_out.mkdir(
        parents=True,
        exist_ok=True,
    )

    mask_out.mkdir(
        parents=True,
        exist_ok=True,
    )

    records = []

    total = len(df)

    print()
    print("=" * 70)
    print(f"PROCESSING {split.upper()}")
    print("=" * 70)
    print(f"Samples: {total}")

    for n, row in enumerate(
        df.itertuples(index=False),
        start=1,
    ):

        dataset = row.dataset

        sample_id = int(
            row.sample_id
        )

        global_id = row.global_id

        # ----------------------------------------------------
        # Locate source TIFFs
        # ----------------------------------------------------

        image_path = locate(
            IMAGE_ROOTS[dataset],
            sample_id,
        )

        mask_path = locate(
            MASK_ROOTS[dataset],
            sample_id,
        )

        # ----------------------------------------------------
        # Read
        # ----------------------------------------------------

        image = read_image(
            image_path
        )

        mask = read_mask(
            mask_path
        )

        # ----------------------------------------------------
        # Normalize SAR
        # ----------------------------------------------------

        image = normalize_sar(
            image
        )

        # ----------------------------------------------------
        # Resize
        # ----------------------------------------------------

        image = resize_image(
            image,
            image_size,
        )

        mask = resize_mask(
            mask,
            image_size,
        )

        # ----------------------------------------------------
        # Output paths
        # ----------------------------------------------------

        image_file = (
            image_out
            / f"{global_id}.npy"
        )

        mask_file = (
            mask_out
            / f"{global_id}.npy"
        )

        # ----------------------------------------------------
        # Save
        # ----------------------------------------------------

        np.save(
            image_file,
            image.astype(
                np.float32
            ),
        )

        np.save(
            mask_file,
            mask.astype(
                np.float32
            ),
        )

        # ----------------------------------------------------
        # Manifest record
        # ----------------------------------------------------

        records.append(
            {
                "global_id":
                    global_id,

                "dataset":
                    dataset,

                "sample_id":
                    sample_id,

                "split":
                    split,

                "image":
                    str(image_file),

                "mask":
                    str(mask_file),
            }
        )

        # ----------------------------------------------------
        # Progress
        # ----------------------------------------------------

        if (
            n == 1
            or n % 25 == 0
            or n == total
        ):

            print(
                f"[{n:3d}/{total}] "
                f"{100.0 * n / total:6.2f}%"
            )

    # --------------------------------------------------------
    # Split manifest
    # --------------------------------------------------------

    manifest_out = (
        output
        / "manifest.csv"
    )

    pd.DataFrame(
        records
    ).to_csv(
        manifest_out,
        index=False,
    )

    print(
        f"Saved: {manifest_out}"
    )

    return records


# ============================================================
# MAIN
# ============================================================

def main():

    parser = argparse.ArgumentParser(
        description=(
            "PS26143 selected-900 "
            "dataset preprocessing"
        )
    )

    parser.add_argument(
        "--workspace",
        type=Path,
        default=DEFAULT_WORKSPACE,
        help=(
            "Output workspace. "
            "Use local Colab storage for "
            "fast preprocessing."
        ),
    )

    parser.add_argument(
        "--image-size",
        type=int,
        default=512,
    )

    args = parser.parse_args()

    manifest_dir = (
        DATA_ROOT
        / "data/manifests"
    )

    # --------------------------------------------------------
    # Verify manifests
    # --------------------------------------------------------

    required_manifests = [
        "train_manifest.csv",
        "val_manifest.csv",
        "test_manifest.csv",
    ]

    for filename in required_manifests:

        path = (
            manifest_dir
            / filename
        )

        if not path.exists():

            raise FileNotFoundError(
                f"Missing manifest: {path}"
            )

    # --------------------------------------------------------
    # Create workspace
    # --------------------------------------------------------

    args.workspace.mkdir(
        parents=True,
        exist_ok=True,
    )

    all_records = []

    # --------------------------------------------------------
    # Process train / val / test
    # --------------------------------------------------------

    for split in [
        "train",
        "val",
        "test",
    ]:

        manifest = (
            manifest_dir
            / f"{split}_manifest.csv"
        )

        records = process_split(
            manifest,
            split,
            args.workspace,
            args.image_size,
        )

        all_records.extend(
            records
        )

    # --------------------------------------------------------
    # Combined manifest
    # --------------------------------------------------------

    processed_dir = (
        args.workspace
        / "processed"
    )

    combined = pd.DataFrame(
        all_records
    )

    combined_path = (
        processed_dir
        / "processed_manifest.csv"
    )

    combined.to_csv(
        combined_path,
        index=False,
    )

    # --------------------------------------------------------
    # Final summary
    # --------------------------------------------------------

    print()
    print("=" * 70)
    print("PREPROCESSING COMPLETE")
    print("=" * 70)

    print(
        f"Total: {len(combined)}"
    )

    print()

    print(
        combined.groupby(
            ["split", "dataset"]
        ).size()
    )

    print()

    print("Processed output:")
    print(processed_dir)

    print()

    print(
        f"Combined manifest: "
        f"{combined_path}"
    )


if __name__ == "__main__":
    main()