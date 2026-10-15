"""Data loading and preprocessing shared by notebooks and the app."""

from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision.transforms import v2

CLASSES = ["glioma", "meningioma", "notumor", "pituitary"]
CLASS_TO_IDX = {c: i for i, c in enumerate(CLASSES)}
IMG_SIZE = 224


def crop_black_borders(img, threshold=10):
    """Crop the black frame around the brain."""
    gray = np.asarray(img.convert("L"))
    mask = gray > threshold
    if not mask.any():          # fully black image: leave it as is
        return img
    rows = np.where(mask.any(axis=1))[0]
    cols = np.where(mask.any(axis=0))[0]
    return img.crop((cols[0], rows[0], cols[-1] + 1, rows[-1] + 1))


class RandomGamma:
    """Non-linear brightness change: x -> x ** gamma (x in [0, 1])."""

    def __init__(self, gamma_range=(0.7, 1.5), p=0.5):
        self.gamma_range = gamma_range
        self.p = p

    def __call__(self, x):
        if torch.rand(1).item() < self.p:
            gamma = torch.empty(1).uniform_(*self.gamma_range).item()
            x = x.clamp(0, 1) ** gamma
        return x


def per_image_normalize(x, eps=1e-6):
    """Zero mean, unit std computed on each image (removes global brightness/contrast)."""
    return (x - x.mean()) / (x.std() + eps)


def get_transforms(train, img_size=IMG_SIZE):
    """Preprocessing pipeline; augmentation only when train=True."""
    steps = [
        v2.Lambda(crop_black_borders),
        v2.ToImage(),
        v2.ToDtype(torch.float32, scale=True),
        v2.Resize((img_size, img_size), antialias=True),
    ]
    if train:
        steps += [
            v2.RandomHorizontalFlip(),
            v2.RandomAffine(degrees=10, translate=(0.05, 0.05), scale=(0.95, 1.05)),
            RandomGamma(),
        ]
    steps.append(v2.Lambda(per_image_normalize))
    return v2.Compose(steps)


class BrainMRIDataset(Dataset):
    """Images listed in a split CSV (columns: path, label), paths relative to root."""

    def __init__(self, df, root, transform):
        self.root = Path(root)
        self.paths = df["path"].tolist()
        self.labels = [CLASS_TO_IDX[label] for label in df["label"]]
        self.transform = transform

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        img = Image.open(self.root / self.paths[idx]).convert("RGB")
        return self.transform(img), self.labels[idx]
