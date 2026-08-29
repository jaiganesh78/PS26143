from pathlib import Path
import csv
import re


DATA_ROOT = Path(r"D:\PS26143_DATA")
OUTPUT = DATA_ROOT / "audit" / "reports" / "dataset_inventory.csv"


DATASETS = {
    "oil": {
        "image_archive": DATA_ROOT / "raw" / "zenodo_part1"
        / "01_Train_Val_Oil_Spill_images.7z",
        "mask_archive": DATA_ROOT / "raw" / "zenodo_part1"
        / "01_Train_Val_Oil_Spill_mask.7z",
        "image_prefix": "Oil",
        "mask_prefix": "Mask_oil",
    },
    "lookalike": {
        "image_archive": DATA_ROOT / "raw" / "zenodo_part2"
        / "01_Train_Val_Lookalike_images.7z",
        "mask_archive": DATA_ROOT / "raw" / "zenodo_part2"
        / "01_Train_Val_Lookalike_mask.7z",
        "image_prefix": "Lookalike",
        "mask_prefix": "Mask_lookalike",
    },
    "no_oil": {
        "image_archive": DATA_ROOT / "raw" / "zenodo_part2"
        / "01_Train_Val_No_Oil_Images.7z",
        "mask_archive": DATA_ROOT / "raw" / "zenodo_part2"
        / "01_Train_Val_No_Oil_mask.7z",
        "image_prefix": "No_oil",
        "mask_prefix": "Mask_no_oil",
    },
}


def get_archive_files(archive: Path):
    """
    Read the file list from a 7z archive using the 7-Zip CLI.
    Returns normalized internal paths for TIFF files.
    """

    import subprocess

    result = subprocess.run(
        ["7z", "l", "-slt", str(archive)],
        capture_output=True,
        text=True,
        check=True,
    )

    paths = []

    for line in result.stdout.splitlines():
        if line.startswith("Path = "):
            path = line[len("Path = "):].strip()

            if path.lower().endswith(".tif"):
                paths.append(path)

    return paths


def extract_id(path: str):
    """
    Extract a six-digit sample ID from a TIFF filename.
    """

    filename = Path(path).name

    match = re.search(r"(\d{5,6})", filename)

    if not match:
        return None

    return match.group(1)


def main():

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)

    rows = []

    for dataset_name, config in DATASETS.items():

        print()
        print("=" * 70)
        print(f"DATASET: {dataset_name}")
        print("=" * 70)

        image_paths = get_archive_files(config["image_archive"])
        mask_paths = get_archive_files(config["mask_archive"])

        image_ids = {
            extract_id(path): path
            for path in image_paths
            if extract_id(path) is not None
        }

        mask_ids = {
            extract_id(path): path
            for path in mask_paths
            if extract_id(path) is not None
        }

        print(f"Images: {len(image_paths)}")
        print(f"Masks:  {len(mask_paths)}")

        image_id_set = set(image_ids)
        mask_id_set = set(mask_ids)

        missing_masks = sorted(image_id_set - mask_id_set)
        orphan_masks = sorted(mask_id_set - image_id_set)

        paired = sorted(image_id_set & mask_id_set)

        print(f"Paired: {len(paired)}")
        print(f"Missing masks: {len(missing_masks)}")
        print(f"Orphan masks: {len(orphan_masks)}")

        for sample_id in paired:

            rows.append(
                {
                    "dataset": dataset_name,
                    "sample_id": sample_id,
                    "image_archive": str(config["image_archive"]),
                    "mask_archive": str(config["mask_archive"]),
                    "image_internal_path": image_ids[sample_id],
                    "mask_internal_path": mask_ids[sample_id],
                    "pair_status": "paired",
                }
            )

        for sample_id in missing_masks:

            rows.append(
                {
                    "dataset": dataset_name,
                    "sample_id": sample_id,
                    "image_archive": str(config["image_archive"]),
                    "mask_archive": str(config["mask_archive"]),
                    "image_internal_path": image_ids[sample_id],
                    "mask_internal_path": "",
                    "pair_status": "missing_mask",
                }
            )

        for sample_id in orphan_masks:

            rows.append(
                {
                    "dataset": dataset_name,
                    "sample_id": sample_id,
                    "image_archive": str(config["image_archive"]),
                    "mask_archive": str(config["mask_archive"]),
                    "image_internal_path": "",
                    "mask_internal_path": mask_ids[sample_id],
                    "pair_status": "orphan_mask",
                }
            )

    fieldnames = [
        "dataset",
        "sample_id",
        "image_archive",
        "mask_archive",
        "image_internal_path",
        "mask_internal_path",
        "pair_status",
    ]

    with OUTPUT.open("w", newline="", encoding="utf-8") as f:

        writer = csv.DictWriter(f, fieldnames=fieldnames)

        writer.writeheader()
        writer.writerows(rows)

    print()
    print("=" * 70)
    print("AUDIT COMPLETE")
    print("=" * 70)
    print(f"Output: {OUTPUT}")


if __name__ == "__main__":
    main()