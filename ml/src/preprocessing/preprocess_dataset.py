import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import rasterio
from PIL import Image


# ============================================================
# PS26143 — FINAL PREPROCESSING PIPELINE
# ============================================================

DATA_ROOT = Path("/content/drive/MyDrive/PS26143")

DEFAULT_WORKSPACE = Path("/content/ps26143_workspace")


IMAGE_ROOTS = {
    "oil":
        DATA_ROOT /
        "data/extracted/selected_900/oil/images/Oil",

    "lookalike":
        DATA_ROOT /
        "data/extracted/selected_900/lookalike/images/Lookalike",

    "no_oil":
        DATA_ROOT /
        "data/extracted/selected_900/no_oil/images/No_oil",
}


MASK_ROOTS = {
    "oil":
        DATA_ROOT /
        "data/extracted/selected_900/oil/masks/Mask_oil",

    "lookalike":
        DATA_ROOT /
        "data/extracted/selected_900/lookalike/masks/Mask_lookalike",

    "no_oil":
        DATA_ROOT /
        "data/extracted/selected_900/no_oil/masks/Mask_no_oil",
}


# ============================================================
# SAR NORMALIZATION
# ============================================================

def normalize_sar(image):
    """
    Per-scene percentile normalization.

    Input:
        [2, H, W]
        VV + VH SAR values.

    Output:
        [2, H, W]
        float32 approximately in [0, 1].

    VV and VH are normalized independently.
    """

    image = image.astype(np.float32)

    output = np.zeros_like(
        image,
        dtype=np.float32
    )

    for band in range(image.shape[0]):

        x = image[band]

        finite = np.isfinite(x)

        if not finite.any():
            continue

        valid = x[finite]

        low = np.percentile(
            valid,
            1.0
        )

        high = np.percentile(
            valid,
            99.0
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
            high
        )

        output[band] = (
            (x - low) /
            (high - low)
        )

    return output


# ============================================================
# IMAGE RESIZE
# ============================================================

def resize_image(image, size):
    """
    Resize VV and VH independently.

    Bilinear interpolation.
    """

    bands = []

    for band in image:

        pil = Image.fromarray(
            band.astype(np.float32),
            mode="F"
        )

        pil = pil.resize(
            (size, size),
            Image.Resampling.BILINEAR
        )

        bands.append(
            np.asarray(
                pil,
                dtype=np.float32
            )
        )

    return np.stack(
        bands,
        axis=0
    )


# ============================================================
# MASK RESIZE
# ============================================================

def resize_mask(mask, size):
    """
    Resize binary mask.

    Nearest-neighbor interpolation prevents
    fractional mask labels.
    """

    mask = mask.astype(np.uint8)

    pil = Image.fromarray(
        mask,
        mode="L"
    )

    pil = pil.resize(
        (size, size),
        Image.Resampling.NEAREST
    )

    mask = np.asarray(
        pil,
        dtype=np.uint8
    )

    return (
        mask > 0
    ).astype(np.float32)


# ============================================================
# TIFF READING
# ============================================================

def read_image(path):

    with rasterio.open(path) as src:

        image = src.read()

    if image.ndim != 3:

        raise ValueError(
            f"Expected [bands,H,W], "
            f"got {image.shape}: {path}"
        )

    if image.shape[0] != 2:

        raise ValueError(
            f"Expected exactly 2 bands "
            f"(VV,VH), got {image.shape[0]}: {path}"
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

    path = (
        root /
        f"{int(sample_id):05d}.tif"
    )

    if not path.exists():

        raise FileNotFoundError(
            f"Missing TIFF: {path}"
        )

    return path


# ============================================================
# SINGLE SPLIT
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
        workspace /
        "processed" /
        split
    )

    image_out = (
        output /
        "images"
    )

    mask_out = (
        output /
        "masks"
    )

    image_out.mkdir(
        parents=True,
        exist_ok=True
    )

    mask_out.mkdir(
        parents=True,
        exist_ok=True
    )

    records = []

    total = len(df)

    print()
    print("=" * 70)
    print(
        f"PROCESSING {split.upper()}"
    )
    print("=" * 70)

    print(
        f"Samples: {total}"
    )

    for n, row in enumerate(
        df.itertuples(index=False),
        start=1
    ):

        dataset = row.dataset

        sample_id = int(
            row.sample_id
        )

        global_id = row.global_id

        image_path = locate(
            IMAGE_ROOTS[dataset],
            sample_id
        )

        mask_path = locate(
            MASK_ROOTS[dataset],
            sample_id
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
        # CRITICAL ALIGNMENT CHECK
        # ----------------------------------------------------

        if image.shape[-2:] != mask.shape:

            raise ValueError(
                "\nIMAGE/MASK SHAPE MISMATCH\n"
                f"global_id : {global_id}\n"
                f"dataset   : {dataset}\n"
                f"sample_id : {sample_id}\n"
                f"image     : {image.shape}\n"
                f"mask      : {mask.shape}\n"
                f"image     : {image_path}\n"
                f"mask      : {mask_path}"
            )

        # ----------------------------------------------------
        # Validate image
        # ----------------------------------------------------

        if not np.isfinite(image).any():

            raise ValueError(
                f"Image contains no finite values: "
                f"{image_path}"
            )

        # ----------------------------------------------------
        # Validate mask
        # ----------------------------------------------------

        mask_values = np.unique(mask)

        if not np.all(
            np.isin(
                mask_values,
                [0, 1]
            )
        ):

            raise ValueError(
                f"Non-binary mask detected: "
                f"{mask_path}; "
                f"values={mask_values[:20]}"
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
            image_size
        )

        mask = resize_mask(
            mask,
            image_size
        )

        # ----------------------------------------------------
        # Final shape checks
        # ----------------------------------------------------

        if image.shape != (
            2,
            image_size,
            image_size
        ):

            raise ValueError(
                f"Bad processed image shape: "
                f"{image.shape}"
            )

        if mask.shape != (
            image_size,
            image_size
        ):

            raise ValueError(
                f"Bad processed mask shape: "
                f"{mask.shape}"
            )

        # ----------------------------------------------------
        # Save
        # ----------------------------------------------------

        image_file = (
            image_out /
            f"{global_id}.npy"
        )

        mask_file = (
            mask_out /
            f"{global_id}.npy"
        )

        np.save(
            image_file,
            image.astype(np.float32)
        )

        np.save(
            mask_file,
            mask.astype(np.float32)
        )

        records.append(
            {
                "global_id": global_id,
                "dataset": dataset,
                "sample_id": sample_id,
                "split": split,
                "image": str(
                    image_file
                ),
                "mask": str(
                    mask_file
                ),
            }
        )

        if (
            n == 1
            or n % 25 == 0
            or n == total
        ):

            print(
                f"[{n:3d}/{total}] "
                f"{100.0*n/total:6.2f}%"
            )

    manifest_out = (
        output /
        "manifest.csv"
    )

    pd.DataFrame(
        records
    ).to_csv(
        manifest_out,
        index=False
    )

    print(
        f"Saved: {manifest_out}"
    )

    return records


# ============================================================
# MAIN
# ============================================================

def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--workspace",
        type=Path,
        default=DEFAULT_WORKSPACE
    )

    parser.add_argument(
        "--image-size",
        type=int,
        default=512
    )

    args = parser.parse_args()

    manifest_dir = (
        DATA_ROOT /
        "data/manifests"
    )

    args.workspace.mkdir(
        parents=True,
        exist_ok=True
    )

    all_records = []

    expected = {
        "train": 630,
        "val": 135,
        "test": 135,
    }

    # --------------------------------------------------------
    # Process all three splits
    # --------------------------------------------------------

    for split in [
        "train",
        "val",
        "test"
    ]:

        manifest = (
            manifest_dir /
            f"{split}_manifest.csv"
        )

        if not manifest.exists():

            raise FileNotFoundError(
                f"Missing manifest: {manifest}"
            )

        records = process_split(
            manifest,
            split,
            args.workspace,
            args.image_size
        )

        if len(records) != expected[split]:

            raise RuntimeError(
                f"{split}: expected "
                f"{expected[split]} records, "
                f"got {len(records)}"
            )

        all_records.extend(
            records
        )

    # --------------------------------------------------------
    # Combined manifest
    # --------------------------------------------------------

    combined = pd.DataFrame(
        all_records
    )

    if len(combined) != 900:

        raise RuntimeError(
            f"Expected 900 processed "
            f"samples, got {len(combined)}"
        )

    if combined["global_id"].duplicated().any():

        raise RuntimeError(
            "Duplicate global_id detected "
            "in processed dataset."
        )

    combined_path = (
        args.workspace /
        "processed" /
        "processed_manifest.csv"
    )

    combined.to_csv(
        combined_path,
        index=False
    )

    # --------------------------------------------------------
    # Final summary
    # --------------------------------------------------------

    print()
    print("=" * 70)
    print(
        "PREPROCESSING COMPLETE"
    )
    print("=" * 70)

    print(
        f"Total samples: {len(combined)}"
    )

    print()
    print(
        combined.groupby(
            ["split", "dataset"]
        ).size()
    )

    print()
    print(
        "Output:"
    )

    print(
        args.workspace /
        "processed"
    )


if __name__ == "__main__":
    main()