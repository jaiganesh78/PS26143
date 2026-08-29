from pathlib import Path
import sys

import numpy as np
import rasterio


def inspect_raster(path: Path):
    print("=" * 70)
    print(f"FILE: {path}")
    print("=" * 70)

    with rasterio.open(path) as src:
        print(f"Driver:       {src.driver}")
        print(f"Width:        {src.width}")
        print(f"Height:       {src.height}")
        print(f"Bands:        {src.count}")
        print(f"Dtypes:       {src.dtypes}")
        print(f"CRS:          {src.crs}")
        print(f"NoData:       {src.nodata}")
        print(f"Transform:    {src.transform}")
        print(f"Bounds:       {src.bounds}")

        print("\nBand statistics:")

        for band_idx in range(1, src.count + 1):
            data = src.read(band_idx, masked=True)

            values = data.compressed()

            if values.size == 0:
                print(f"\nBand {band_idx}: EMPTY")
                continue

            print(f"\nBand {band_idx}")
            print(f"  min:       {np.min(values):.6f}")
            print(f"  max:       {np.max(values):.6f}")
            print(f"  mean:      {np.mean(values):.6f}")
            print(f"  std:       {np.std(values):.6f}")
            print(f"  p01:       {np.percentile(values, 1):.6f}")
            print(f"  p05:       {np.percentile(values, 5):.6f}")
            print(f"  p50:       {np.percentile(values, 50):.6f}")
            print(f"  p95:       {np.percentile(values, 95):.6f}")
            print(f"  p99:       {np.percentile(values, 99):.6f}")

            masked_fraction = 1.0 - (values.size / data.size)

            print(f"  masked:    {masked_fraction * 100:.4f}%")


def main():
    if len(sys.argv) != 2:
        print("Usage:")
        print("  python inspect_tiff.py <path-to-tiff>")
        sys.exit(1)

    path = Path(sys.argv[1])

    if not path.exists():
        print(f"ERROR: File does not exist: {path}")
        sys.exit(1)

    inspect_raster(path)


if __name__ == "__main__":
    main()