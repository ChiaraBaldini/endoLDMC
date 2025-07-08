""" Script to compute the performance of the ControlNet."""
import argparse
from pathlib import Path

import numpy as np
from generative.metrics import MultiScaleSSIMMetric
from monai import transforms
from monai.config import print_config
from monai.metrics import MAEMetric, PSNRMetric
from monai.utils import set_determinism
from tqdm import tqdm
from util import get_test_dataloader
import torch


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--seed", type=int, default=2, help="Random seed to use.")
    parser.add_argument("--samples_dir", help="Location of the samples to evaluate.")
    parser.add_argument("--test_ids", help="Location of file with test ids.")
    parser.add_argument("--num_workers", type=int, default=8, help="Number of loader workers")

    args = parser.parse_args()
    return args


def main(args):
    set_determinism(seed=args.seed)
    print_config()

    samples_dir = Path(args.samples_dir)

    sample_transforms = transforms.Compose(
        [
            transforms.LoadImaged(keys=["image"]),
            transforms.EnsureChannelFirstd(keys=["image"]),
            transforms.Rotate90d(keys=["image"], k=-1, spatial_axes=(0, 1)),  # Fix flipped image read
            transforms.Flipd(keys=["image"], spatial_axis=1),  # Fix flipped image read
            transforms.ScaleIntensityRanged(keys=["image"], a_min=0.0, a_max=255.0, b_min=0.0, b_max=1.0, clip=True),
            #transforms.Resize(keys=["image"],size=(224, 224)),
            transforms.ToTensord(keys=["image"]),
        ]
    )

    # Test set
    test_loader = get_test_dataloader(
        batch_size=1,
        test_ids=args.test_ids,
        num_workers=args.num_workers,
        upper_limit=1000,
    )

    psnr_metric = PSNRMetric(max_val=1.0)
    mae_metric = MAEMetric()
    mssim_metric = MultiScaleSSIMMetric(spatial_dims=2, kernel_size=7)

    psnr_list = []
    mae_list = []
    mssim_list = []
    for batch in tqdm(test_loader):
        print(batch)
        img = batch["image"][0]
        print(img.shape)
        resize = transforms.Resize((640, 640))   #was (224, 224)
        img = resize(img)
        img=torch.unsqueeze(img, 0)
        print(img.shape)
        print(Path(batch["filename"][0]).name)
        img_synthetic = sample_transforms(
            {"image": samples_dir / (Path(batch["filename"][0]).name.split('.')[0]+'_synthetic.png')}
        )["image"].unsqueeze(0)

        psnr_value = psnr_metric(img, img_synthetic)
        mae_value = mae_metric(img, img_synthetic)
        mssim_value = mssim_metric(img, img_synthetic)

        psnr_list.append(psnr_value.item())
        mae_list.append(mae_value.item())
        mssim_list.append(mssim_value.item())

    print(f"PSNR: {np.mean(psnr_list)}+-{np.std(psnr_list)}")
    print(f"MAE: {np.mean(mae_list)}+-{np.std(mae_list)}")
    print(f"MSSIM: {np.mean(mssim_list)}+-{np.std(mssim_list)}")


if __name__ == "__main__":
    args = parse_args()
    main(args)
