"""Utility functions for testing."""
from __future__ import annotations

import pandas as pd
from monai import transforms
from monai.data import CacheDataset
from torch.utils.data import DataLoader


def get_test_dataloader(
    batch_size: int,
    test_ids: str,
    num_workers: int = 8,
    upper_limit: int | None = None,
):
    test_transforms = transforms.Compose(
        [
            transforms.LoadImaged(keys=["image", "mask"]),
            transforms.EnsureChannelFirstd(keys=["image", "mask"]),
            transforms.Rotate90d(keys=["image", "mask"], k=-1, spatial_axes=(0, 1)),  # Fix flipped image read
            transforms.Flipd(keys=["image", "mask"], spatial_axis=1),  # Fix flipped image read
            transforms.ScaleIntensityRanged(
                keys=["image", "mask"], a_min=0.0, a_max=255.0, b_min=0.0, b_max=1.0, clip=True
            ),
            transforms.ToTensord(keys=["image", "mask"]),
        ]
    )

    test_dicts = get_datalist(ids_path=test_ids, upper_limit=upper_limit)
    test_ds = CacheDataset(data=test_dicts, transform=test_transforms)
    test_loader = DataLoader(
        test_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        drop_last=False,
        pin_memory=False,
        persistent_workers=True,
    )

    return test_loader


def get_datalist(
    ids_path: str,
    upper_limit: int | None = None,
):
    """Get data dicts for data loaders."""
    df = pd.read_csv(ids_path, sep="\t")

    if upper_limit is not None:
        df = df[:upper_limit]

    data_dicts = []
    for index, row in df.iterrows():
        data_dicts.append(
            {
                "filename": f"{row['filename']}",
                "image": f"{row['image']}",
                "mask": f"{row['mask']}",
                "report": f"{row['caption']}",
            }
        )

    print(f"Found {len(data_dicts)} subjects.")
    return data_dicts
