import numpy as np
import torch
from torch.utils.data import Dataset


class OilSegmentationDataset(Dataset):
    """
    Dataset for preprocessed PS26143 samples.

    Expected:
        image: [2, H, W]
        mask:  [H, W]
    """

    def __init__(self, records, augment=False):
        self.records = records
        self.augment = augment

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        record = self.records[idx]

        image = np.load(
            record["image"]
        ).astype(np.float32)

        mask = np.load(
            record["mask"]
        ).astype(np.float32)

        if image.shape != (
            2,
            512,
            512
        ):
            raise ValueError(
                f"Unexpected image shape "
                f"{image.shape} for "
                f"{record['global_id']}"
            )

        if mask.shape != (
            512,
            512
        ):
            raise ValueError(
                f"Unexpected mask shape "
                f"{mask.shape} for "
                f"{record['global_id']}"
            )

        if self.augment:
            image, mask = self._augment(
                image,
                mask
            )

        image = torch.from_numpy(
            image
        )

        mask = torch.from_numpy(
            mask
        ).unsqueeze(0)

        return {
            "image": image,
            "mask": mask,
            "global_id": record["global_id"],
            "dataset": record["dataset"],
        }

    @staticmethod
    def _augment(image, mask):
        """
        Lightweight geometric augmentation.

        Applied ONLY to training data.
        """

        # Horizontal flip
        if np.random.random() < 0.5:
            image = np.flip(
                image,
                axis=2
            ).copy()

            mask = np.flip(
                mask,
                axis=1
            ).copy()

        # Vertical flip
        if np.random.random() < 0.5:
            image = np.flip(
                image,
                axis=1
            ).copy()

            mask = np.flip(
                mask,
                axis=0
            ).copy()

        # Random 90-degree rotation
        k = np.random.randint(
            0,
            4
        )

        if k:
            image = np.rot90(
                image,
                k=k,
                axes=(1, 2)
            ).copy()

            mask = np.rot90(
                mask,
                k=k,
                axes=(0, 1)
            ).copy()

        return image, mask