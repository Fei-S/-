"""
Dataset loading module.

Supports:
    - Haze4K dataset format: hazy = {id}_{beta}_{A}.png, GT = {id}.png
    - RESIDE SOTS dataset format (same filenames)
    - Custom image pair datasets
    - Data augmentation (random flip, resize)
"""

import os
import random
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms as T

# Image file extensions
_IMG_EXTS = (".png", ".jpg", ".jpeg", ".bmp", ".tiff")


def _is_image(fname: str) -> bool:
    """Check if filename is a supported image."""
    return fname.lower().endswith(_IMG_EXTS)


def _match_haze4k(hazy_dir: str, gt_dir: str) -> tuple[list[str], list[str]]:
    """
    Match Haze4K-format files: hazy '{id}_{beta}_{A}.png' → gt '{id}.png'

    Also works for direct 1:1 filenames.
    """
    hazy_all = sorted(f for f in os.listdir(hazy_dir) if _is_image(f))
    gt_all = sorted(f for f in os.listdir(gt_dir) if _is_image(f))

    # Build GT lookup by number (Haze4K uses numeric IDs)
    gt_by_num: dict[str, str] = {}
    for gf in gt_all:
        stem = os.path.splitext(gf)[0]
        gt_by_num[stem] = gf

    matched_hazy: list[str] = []
    matched_gt: list[str] = []

    for hf in hazy_all:
        stem = os.path.splitext(hf)[0]

        # Haze4K: '1000_0.74_1.6' → '1000'
        # Try extracting the first underscore-separated token
        base_id = stem.split("_")[0]

        if base_id in gt_by_num:
            matched_hazy.append(hf)
            matched_gt.append(gt_by_num[base_id])
        elif stem in gt_by_num:
            # Direct match (same filename)
            matched_hazy.append(hf)
            matched_gt.append(gt_by_num[stem])

    if not matched_hazy:
        # Fallback: sort both and pair by index
        n = min(len(hazy_all), len(gt_all))
        print(f"[dataset] Fallback pairwise match: {n} pairs")
        return hazy_all[:n], gt_all[:n]

    print(f"[dataset] Matched {len(matched_hazy)} hazy-GT pairs")
    return matched_hazy, matched_gt


class DehazeDataset(Dataset):
    """Dehazing dataset for hazy/clean image pairs.

    Parameters
    ----------
    hazy_dir : str
        Path to hazy images.
    gt_dir : str
        Path to ground-truth (clean) images.
    image_size : tuple[int, int]
        Output image size (H, W).
    is_train : bool
        Training mode enables data augmentation.
    num_samples : int or None
        Limit total samples (None = all).
    """

    def __init__(
        self,
        hazy_dir: str,
        gt_dir: str,
        image_size: tuple[int, int] = (256, 256),
        is_train: bool = True,
        num_samples: int | None = None,
    ):
        self.hazy_dir = Path(hazy_dir)
        self.gt_dir = Path(gt_dir)
        self.image_size = image_size
        self.is_train = is_train

        # Match hazy-GT pairs (Haze4K naming or direct)
        self.hazy_files, self.gt_files = _match_haze4k(hazy_dir, gt_dir)

        if num_samples is not None and num_samples < len(self.hazy_files):
            self.hazy_files = self.hazy_files[:num_samples]
            self.gt_files = self.gt_files[:num_samples]

        # Data augmentation (training only)
        if is_train:
            self.transform = T.Compose([
                T.ToTensor(),
                T.RandomHorizontalFlip(p=0.5),
                T.Resize(image_size),
            ])
        else:
            self.transform = T.Compose([
                T.ToTensor(),
                T.Resize(image_size),
            ])

    def __len__(self) -> int:
        return len(self.hazy_files)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        hazy_path = self.hazy_dir / self.hazy_files[idx]
        gt_path = self.gt_dir / self.gt_files[idx]

        hazy_img = cv2.imread(str(hazy_path))
        gt_img = cv2.imread(str(gt_path))

        if hazy_img is None:
            raise FileNotFoundError(f"Cannot read: {hazy_path}")
        if gt_img is None:
            raise FileNotFoundError(f"Cannot read: {gt_path}")

        # BGR → RGB
        hazy_img = cv2.cvtColor(hazy_img, cv2.COLOR_BGR2RGB)
        gt_img = cv2.cvtColor(gt_img, cv2.COLOR_BGR2RGB)

        # Synchronised random transforms
        seed = random.randint(0, 2**32 - 1)

        random.seed(seed)
        torch.manual_seed(seed)
        hazy_tensor = self.transform(hazy_img)

        random.seed(seed)
        torch.manual_seed(seed)
        gt_tensor = self.transform(gt_img)

        return hazy_tensor, gt_tensor


def create_dataloaders(
    hazy_dir: str,
    gt_dir: str,
    image_size: tuple[int, int] = (256, 256),
    batch_size: int = 4,
    num_workers: int = 0,
    train_ratio: float = 0.8,
    num_samples: int | None = None,
) -> tuple[DataLoader, DataLoader]:
    """Create train/test dataloaders from a single hazy/gt pair dir.

    Splits images into train/test based on train_ratio.
    Use ``create_dataloaders_split`` if Haze4K already has separate dirs.
    """
    dataset = DehazeDataset(
        hazy_dir=hazy_dir,
        gt_dir=gt_dir,
        image_size=image_size,
        is_train=True,
        num_samples=num_samples,
    )

    n = len(dataset)
    indices = list(range(n))
    random.Random(42).shuffle(indices)
    split = int(n * train_ratio)

    train_ds = torch.utils.data.Subset(dataset, indices[:split])
    test_ds = torch.utils.data.Subset(dataset, indices[split:])

    # Override test transform (no augmentation)
    test_ds = _SubsetWithTransform(
        test_ds,
        T.Compose([T.ToTensor(), T.Resize(image_size)]),
    )

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=True,
    )
    test_loader = DataLoader(
        test_ds, batch_size=1, shuffle=False,
        num_workers=num_workers, pin_memory=True,
    )

    print(f"Train: {len(train_ds)} images, {len(train_loader)} batches")
    print(f"Test:  {len(test_ds)} images, {len(test_loader)} batches")
    return train_loader, test_loader


def create_dataloaders_haze4k(
    train_hazy_dir: str,
    train_gt_dir: str,
    test_hazy_dir: str,
    test_gt_dir: str,
    image_size: tuple[int, int] = (256, 256),
    batch_size: int = 4,
    num_workers: int = 0,
    num_train: int | None = None,
    num_test: int | None = None,
) -> tuple[DataLoader, DataLoader]:
    """Create dataloaders from pre-split Haze4K directories.

    Parameters
    ----------
    train_hazy_dir / train_gt_dir : str
        Haze4K training directories.
    test_hazy_dir / test_gt_dir : str
        Haze4K test directories.
    image_size : tuple[int, int]
        Output image size.
    batch_size : int
        Training batch size.
    num_workers : int
        Data loading workers (0 = main process).
    num_train / num_test : int or None
        Limit number of samples (None = all).

    Returns
    -------
    (train_loader, test_loader)
    """
    train_ds = DehazeDataset(
        hazy_dir=train_hazy_dir,
        gt_dir=train_gt_dir,
        image_size=image_size,
        is_train=True,
        num_samples=num_train,
    )

    test_ds = DehazeDataset(
        hazy_dir=test_hazy_dir,
        gt_dir=test_gt_dir,
        image_size=image_size,
        is_train=False,
        num_samples=num_test,
    )

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=True,
    )
    test_loader = DataLoader(
        test_ds, batch_size=1, shuffle=False,
        num_workers=num_workers, pin_memory=True,
    )

    print(f"Train: {len(train_ds)} images, {len(train_loader)} batches")
    print(f"Test:  {len(test_ds)} images, {len(test_loader)} batches")
    return train_loader, test_loader


class _SubsetWithTransform(torch.utils.data.Subset):
    """Subset that overrides the parent transform."""

    def __init__(self, subset, transform):
        super().__init__(subset.dataset, subset.indices)
        self.custom_transform = transform

    def __getitem__(self, idx):
        data_idx = self.indices[idx]
        hazy_path = self.dataset.hazy_dir / self.dataset.hazy_files[data_idx]
        gt_path = self.dataset.gt_dir / self.dataset.gt_files[data_idx]

        hazy_img = cv2.imread(str(hazy_path))
        gt_img = cv2.imread(str(gt_path))

        hazy_img = cv2.cvtColor(hazy_img, cv2.COLOR_BGR2RGB)
        gt_img = cv2.cvtColor(gt_img, cv2.COLOR_BGR2RGB)

        hazy_tensor = self.custom_transform(hazy_img)
        gt_tensor = self.custom_transform(gt_img)

        return hazy_tensor, gt_tensor


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 3:
        print("Usage: python dataset.py <hazy_dir> <gt_dir>")
        sys.exit(1)

    train_loader, test_loader = create_dataloaders(
        hazy_dir=sys.argv[1], gt_dir=sys.argv[2],
        image_size=(256, 256), batch_size=4,
    )

    hazy, gt = next(iter(train_loader))
    print(f"Batch shapes — hazy: {hazy.shape}, gt: {gt.shape}")
    print(f"Value ranges — hazy: [{hazy.min():.3f}, {hazy.max():.3f}], "
          f"gt: [{gt.min():.3f}, {gt.max():.3f}]")
