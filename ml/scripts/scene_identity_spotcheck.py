from pathlib import Path
import csv
import shutil
import tempfile

import py7zr
import rasterio


DATA_ROOT = Path(r"D:\PS26143_DATA")

ARCHIVES = {
    "oil": (
        DATA_ROOT
        / "raw"
        / "zenodo_part1"
        / "01_Train_Val_Oil_Spill_images.7z"
    ),
    "lookalike": (
        DATA_ROOT
        / "raw"
        / "zenodo_part2"
        / "01_Train_Val_Lookalike_images.7z"
    ),
    "no_oil": (
        DATA_ROOT
        / "raw"
        / "zenodo_part2"
        / "01_Train_Val_No_Oil_Images.7z"
    ),
}


# Use IDs that are known to occur across the datasets.
SAMPLE_IDS = {
    "00000",
    "00050",
    "00200",
    "00400",
    "00600",
    "00681",
}


def find_members(archive_path: Path):

    print(f"\nReading archive index:")
    print(f"  {archive_path.name}")

    with py7zr.SevenZipFile(
        archive_path,
        mode="r",
    ) as archive:

        names = archive.getnames()

    wanted = {}

    for name in names:

        if not name.lower().endswith(".tif"):
            continue

        filename = Path(name).stem

        if filename in SAMPLE_IDS:
            wanted[filename] = name

    return wanted


def extract_selected(
    archive_path: Path,
    members,
    output_dir: Path,
):

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    if not members:
        return

    print(
        f"  Extracting {len(members)} selected files..."
    )

    with py7zr.SevenZipFile(
        archive_path,
        mode="r",
    ) as archive:

        archive.extract(
            targets=list(members.values()),
            path=output_dir,
        )


def inspect(path: Path):

    with rasterio.open(path) as src:

        bounds = src.bounds

        return {
            "width": src.width,
            "height": src.height,
            "bands": src.count,
            "crs": str(src.crs),

            "left": bounds.left,
            "bottom": bounds.bottom,
            "right": bounds.right,
            "top": bounds.top,

            "center_lon": (
                bounds.left + bounds.right
            ) / 2,

            "center_lat": (
                bounds.bottom + bounds.top
            ) / 2,
        }


def bbox_area(row):

    return max(
        0.0,
        row["right"] - row["left"],
    ) * max(
        0.0,
        row["top"] - row["bottom"],
    )


def bbox_intersection(a, b):

    left = max(
        a["left"],
        b["left"],
    )

    right = min(
        a["right"],
        b["right"],
    )

    bottom = max(
        a["bottom"],
        b["bottom"],
    )

    top = min(
        a["top"],
        b["top"],
    )

    width = max(
        0.0,
        right - left,
    )

    height = max(
        0.0,
        top - bottom,
    )

    return width * height


def bbox_iou(a, b):

    intersection = bbox_intersection(
        a,
        b,
    )

    union = (
        bbox_area(a)
        + bbox_area(b)
        - intersection
    )

    if union <= 0:
        return 0.0

    return intersection / union


def main():

    report_dir = (
        DATA_ROOT
        / "audit"
        / "reports"
    )

    report_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    output_csv = (
        report_dir
        / "scene_identity_spotcheck.csv"
    )

    temp_dir = Path(
        tempfile.mkdtemp(
            prefix="ps26143_scene_check_"
        )
    )

    rows = []

    try:

        print("=" * 70)
        print("PS26143 SCENE IDENTITY SPOT CHECK")
        print("=" * 70)

        for dataset, archive in ARCHIVES.items():

            print()
            print("=" * 70)
            print(f"DATASET: {dataset}")
            print("=" * 70)

            members = find_members(
                archive
            )

            print(
                f"  Found {len(members)} "
                f"/ {len(SAMPLE_IDS)} sample IDs"
            )

            extract_root = (
                temp_dir
                / dataset
            )

            extract_selected(
                archive,
                members,
                extract_root,
            )

            for sample_id in sorted(
                members.keys()
            ):

                member = members[sample_id]

                path = (
                    extract_root
                    / member
                )

                if not path.exists():

                    print(
                        f"  WARNING: missing "
                        f"{sample_id}"
                    )

                    continue

                stats = inspect(path)

                rows.append(
                    {
                        "sample_id": sample_id,
                        "dataset": dataset,
                        "internal_path": member,
                        **stats,
                    }
                )

                print(
                    f"  {sample_id}: "
                    f"center=("
                    f"{stats['center_lon']:.6f}, "
                    f"{stats['center_lat']:.6f})"
                )

        fieldnames = [
            "sample_id",
            "dataset",
            "internal_path",
            "width",
            "height",
            "bands",
            "crs",
            "left",
            "bottom",
            "right",
            "top",
            "center_lon",
            "center_lat",
        ]

        with output_csv.open(
            "w",
            newline="",
            encoding="utf-8",
        ) as f:

            writer = csv.DictWriter(
                f,
                fieldnames=fieldnames,
            )

            writer.writeheader()
            writer.writerows(rows)

        print()
        print("=" * 70)
        print("GEOGRAPHIC COMPARISON")
        print("=" * 70)

        for sample_id in sorted(
            SAMPLE_IDS
        ):

            sample_rows = [
                r
                for r in rows
                if r["sample_id"] == sample_id
            ]

            if len(sample_rows) < 2:
                continue

            print()
            print(
                f"ID {sample_id}"
            )

            for i in range(
                len(sample_rows)
            ):

                for j in range(
                    i + 1,
                    len(sample_rows),
                ):

                    a = sample_rows[i]
                    b = sample_rows[j]

                    iou = bbox_iou(
                        a,
                        b,
                    )

                    print(
                        f"  "
                        f"{a['dataset']} vs "
                        f"{b['dataset']}: "
                        f"IoU={iou:.6f}"
                    )

        print()
        print("=" * 70)
        print("COMPLETE")
        print("=" * 70)

        print(
            f"Report: {output_csv}"
        )

    finally:

        shutil.rmtree(
            temp_dir,
            ignore_errors=True,
        )


if __name__ == "__main__":
    main()