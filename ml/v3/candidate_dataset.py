from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


class CandidateDataset(Dataset):

    def __init__(
        self,
        dataframe,
        augment=False,
    ):
        self.df = dataframe.reset_index(drop=True)
        self.augment = augment

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):

        row = self.df.iloc[idx]

        image = np.load(
            Path(row["crop_image"])
        ).astype(np.float32)

        candidate_mask = np.load(
            Path(row["crop_mask"])
        ).astype(np.float32)

        if image.shape[0] != 2:
            raise ValueError(
                f"Expected 2 SAR channels, got {image.shape}"
            )

        candidate_mask = (
            candidate_mask > 0.5
        ).astype(np.float32)

        # ----------------------------------------------------
        # 3-channel candidate representation
        # ----------------------------------------------------

        x = np.concatenate(
            [
                image,
                candidate_mask[None, ...],
            ],
            axis=0,
        )

        if self.augment:
            x = self._augment(x)

        # ----------------------------------------------------
        # Geometric / confidence features
        # ----------------------------------------------------

        area = float(row["area"])

        width = max(
            float(row["width"]),
            1.0,
        )

        height = max(
            float(row["height"]),
            1.0,
        )

        aspect_ratio = width / height

        # Candidate area relative to 512x512 source.
        normalized_area = area / (
            512.0 * 512.0
        )

        # Log-area stabilizes the very large dynamic range.
        log_area = np.log1p(area) / np.log1p(
            512.0 * 512.0
        )

        cx = float(row["centroid_x"]) / 512.0
        cy = float(row["centroid_y"]) / 512.0

        mean_probability = float(
            row["mean_probability"]
        )

        p95_probability = float(
            row["p95_probability"]
        )

        max_probability = float(
            row["max_probability"]
        )

        features = np.array(
            [
                normalized_area,
                log_area,
                aspect_ratio / 10.0,
                cx,
                cy,
                mean_probability,
                p95_probability,
                max_probability,
            ],
            dtype=np.float32,
        )

        label = 1.0 if (
            row["label"] == "positive"
        ) else 0.0

        return {
            "image": torch.from_numpy(x),
            "features": torch.from_numpy(
                features
            ),
            "label": torch.tensor(
                [label],
                dtype=torch.float32,
            ),
            "global_id": str(
                row["global_id"]
            ),
            "candidate_id": str(
                row["candidate_id"]
            ),
        }

    @staticmethod
    def _augment(x):

        if np.random.random() < 0.5:
            x = np.flip(
                x,
                axis=2,
            ).copy()

        if np.random.random() < 0.5:
            x = np.flip(
                x,
                axis=1,
            ).copy()

        k = np.random.randint(0, 4)

        if k:
            x = np.rot90(
                x,
                k=k,
                axes=(1, 2),
            ).copy()

        return x