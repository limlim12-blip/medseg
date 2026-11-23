import glob
import ssl

import certifi
import numpy as np

context = ssl.create_default_context(cafile=certifi.where())
import os
import shutil
import urllib.request
import zipfile
from pathlib import Path

import torch
import torchvision.transforms.functional as TF
from PIL import Image
from torch.utils.data import Dataset

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

# Kvasir-SEG (polyp segmentation) on Kaggle:
#   curl -L -o ~/Downloads/kvasir-seg.zip \
#     https://www.kaggle.com/api/v1/datasets/download/fkarimovv/kvasir-seg
KVASIR_URL = "https://www.kaggle.com/api/v1/datasets/download/fkarimovv/kvasir-seg"


class MedicalSegmentationDataset(Dataset):
    """
    Map-style dataset for binary medical image segmentation.

    Reads an RGB image and its binary mask, resizes both, optionally applies a
    joint augmentation (flips / 90-degree rotations) and returns
    ``(image [3,H,W], mask [1,H,W])``.
    """

    def __init__(
        self,
        image_paths,
        mask_paths,
        img_size=(256, 256),
        augment=False,
    ):
        super().__init__()
        assert len(image_paths) == len(mask_paths), (
            f"Number of images ({len(image_paths)}) and masks ({len(mask_paths)}) must match"
        )
        self.image_paths = list(image_paths)
        self.mask_paths = list(mask_paths)
        self.img_size = tuple(img_size)
        self.augment = augment

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image = Image.open(self.image_paths[idx]).convert("RGB")
        mask = Image.open(self.mask_paths[idx]).convert("L")

        image = image.resize(self.img_size, Image.Resampling.BILINEAR)
        mask = mask.resize(self.img_size, Image.Resampling.NEAREST)

        image = TF.to_tensor(image)  # [3,H,W] in [0,1]
        mask = torch.from_numpy(
            (np.array(mask, dtype=np.float32) > 127.5).astype(np.float32)
        ).unsqueeze(0)  # [1,H,W] in {0,1}
        rng = np.random.default_rng(seed=42)
        if self.augment:
            if rng.random() > 0.5:
                image, mask = TF.hflip(image), TF.hflip(mask)
            if rng.random() > 0.5:
                image, mask = TF.vflip(image), TF.vflip(mask)
            if rng.random() > 0.2:
                angle = rng.uniform(-180.0, 180.0)
                translate = [
                    rng.uniform(-0.08, 0.08) * self.img_size[1],
                    rng.uniform(-0.08, 0.08) * self.img_size[0],
                ]
                scale = rng.uniform(0.85, 1.20)
                shear = [rng.uniform(-8.0, 8.0)]

                image = TF.affine(
                    image,
                    angle=angle,
                    translate=translate,
                    scale=scale,
                    shear=shear,
                    interpolation=TF.InterpolationMode.BILINEAR,
                    fill=[0],
                )
                mask = TF.affine(
                    mask,
                    angle=angle,
                    translate=translate,
                    scale=scale,
                    shear=shear,
                    interpolation=TF.InterpolationMode.NEAREST,
                    fill=[0],
                )

            if rng.random() > 0.3:
                image = TF.adjust_brightness(image, rng.uniform(0.8, 1.2))
                image = TF.adjust_contrast(image, rng.uniform(0.8, 1.2))
                image = TF.adjust_saturation(image, rng.uniform(0.85, 1.15))
                image = TF.adjust_hue(image, rng.uniform(-0.03, 0.03))
            if rng.random() > 0.8:
                image = TF.gaussian_blur(image, kernel_size=[3, 3])
        image = TF.normalize(image, mean=IMAGENET_MEAN, std=IMAGENET_STD)
        return image, mask


def _locate_dirs(data_dir):
    """
    Find the folders holding images and masks. Looks for sibling ``images`` / ``masks``
    folders anywhere below ``data_dir`` (case-insensitive), which also covers a freshly
    unzipped download that adds one or two wrapper folders. Falls back to ``data_dir`` itself.
    """
    if os.path.isdir(data_dir):
        for dirpath, dirnames, _ in os.walk(data_dir):
            lowered = {d.lower(): d for d in sorted(dirnames)}
            if "images" in lowered and "masks" in lowered:
                return (
                    os.path.join(dirpath, lowered["images"]),
                    os.path.join(dirpath, lowered["masks"]),
                )
    return data_dir, data_dir


def find_image_mask_pairs(data_dir):
    """
    Pair images with masks. Supports ``images/`` + ``masks/`` sub-folders (at any depth below
    ``data_dir``), or everything in one folder using ``_mask`` / ``_segmentation`` suffixes.
    """
    valid_exts = ("*.jpg", "*.jpeg", "*.png", "*.bmp", "*.tif", "*.tiff")

    images_dir, masks_dir = _locate_dirs(data_dir)

    all_images = []
    for ext in valid_exts:
        all_images.extend(glob.glob(os.path.join(images_dir, ext)))
        all_images.extend(glob.glob(os.path.join(images_dir, ext.upper())))
    all_images = sorted(set(all_images))

    image_files = [
        f
        for f in all_images
        if "_mask" not in f.lower() and "_segmentation" not in f.lower()
    ]

    paired_images, paired_masks = [], []
    for img_path in image_files:
        stem = Path(img_path).stem
        candidates = [
            f"{stem}.png",
            f"{stem}.jpg",
            f"{stem}.jpeg",
            f"{stem}_mask.png",
            f"{stem}_mask.jpg",
            f"{stem}_segmentation.png",
            f"{stem}_segmentation.jpg",
        ]
        for name in candidates:
            mask_path = os.path.join(masks_dir, name)
            # when images and masks share a folder, never pair an image with itself
            if os.path.exists(mask_path) and mask_path != img_path:
                paired_images.append(img_path)
                paired_masks.append(mask_path)
                break

    return paired_images, paired_masks


def download_kvasir(data_dir, url=KVASIR_URL):
    """Download the Kvasir-SEG zip from Kaggle into ``data_dir`` and extract it there."""
    data_dir = Path(data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)
    zip_path = data_dir / "kvasir-seg.zip"

    print(f"Downloading Kvasir-SEG from {url} ...")
    request = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    try:
        with (
            urllib.request.urlopen(request, timeout=60, context=context) as response,
            open(zip_path, "wb") as out,
        ):
            shutil.copyfileobj(response, out)
    except OSError as e:
        zip_path.unlink(missing_ok=True)
        raise RuntimeError(
            f"Could not download Kvasir-SEG ({e}). Download it manually with\n"
            f"  curl -L -o kvasir-seg.zip {url}\n"
            f"and unzip it into '{data_dir}'."
        ) from e

    if not zipfile.is_zipfile(zip_path):
        zip_path.unlink(missing_ok=True)
        raise RuntimeError(f"The file downloaded from {url} is not a zip archive.")
    with zipfile.ZipFile(zip_path) as archive:
        archive.extractall(data_dir)
    zip_path.unlink()
    print(f"Extracted Kvasir-SEG to {data_dir}")


def ensure_dataset(data_dir, download=True):
    """Make sure ``data_dir`` holds image/mask pairs, downloading Kvasir-SEG when it does not."""
    if find_image_mask_pairs(data_dir)[0]:
        return
    if not download:
        raise FileNotFoundError(
            f"No image/mask pairs found in '{data_dir}' and download is disabled "
            "(config data_loader.download)."
        )
    print(f"No image/mask pairs found in '{data_dir}'.")
    download_kvasir(data_dir)
    if not find_image_mask_pairs(data_dir)[0]:
        raise FileNotFoundError(
            f"Downloaded Kvasir-SEG into '{data_dir}' but found no image/mask pairs in it."
        )


def split_indices(n, val_ratio, test_ratio, seed=123):
    """Deterministic train / val / test index split."""
    rng = np.random.default_rng(seed)
    indices = list(range(n))
    rng.shuffle(indices)

    n_val = int(n * val_ratio)
    n_test = int(n * test_ratio)
    # keep at least one sample in val / test when the dataset is tiny
    if val_ratio > 0 and n_val == 0 and n >= 3:
        n_val = 1
    if test_ratio > 0 and n_test == 0 and n >= 3:
        n_test = 1
    n_train = n - n_val - n_test

    train_idx = indices[:n_train]
    val_idx = indices[n_train : n_train + n_val]
    test_idx = indices[n_train + n_val :]
    return train_idx, val_idx, test_idx


def build_datasets(data_dir, img_size, val_ratio, test_ratio, seed=123, download=True):
    """
    Build the train / val / test map-style datasets from a data directory.
    If the directory has no image/mask pairs and ``download`` is true, Kvasir-SEG is
    downloaded and extracted into it first.
    """
    ensure_dataset(data_dir, download)
    image_paths, mask_paths = find_image_mask_pairs(data_dir)

    train_idx, val_idx, test_idx = split_indices(
        len(image_paths), val_ratio, test_ratio, seed
    )

    def make(idx, augment):
        return MedicalSegmentationDataset(
            [image_paths[i] for i in idx],
            [mask_paths[i] for i in idx],
            img_size=img_size,
            augment=augment,
        )

    return make(train_idx, True), make(val_idx, False), make(test_idx, False)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Download Kvasir-SEG (Kaggle) into a folder"
    )
    parser.add_argument("--dir", default="data/kvasir-seg")
    args = parser.parse_args()
    ensure_dataset(args.dir, download=True)
    pairs = find_image_mask_pairs(args.dir)[0]
    print(f"{len(pairs)} image/mask pairs ready in {args.dir}")
