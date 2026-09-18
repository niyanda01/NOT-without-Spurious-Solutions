import math
from pathlib import Path

import torch
from torch.utils.data import ConcatDataset, DataLoader, Dataset
from torchvision.datasets import MNIST
from torchvision.transforms import functional as TF
from torchvision.transforms import Resize
from PIL import Image


class LoaderSampler:
    """Wraps a DataLoader as an infinite `.sample()` source for train.py's
    sample_mu/sample_nu callables."""

    def __init__(self, dataset, batch_size, seed, device=None):
        generator = torch.Generator().manual_seed(seed)
        self.loader = DataLoader(
            dataset, batch_size=batch_size, shuffle=True, drop_last=True,
            num_workers=0, generator=generator,
        )
        self.iterator = iter(self.loader)
        self.device = device

    def sample(self):
        try:
            images, _ = next(self.iterator)
        except StopIteration:
            self.iterator = iter(self.loader)
            images, _ = next(self.iterator)
        return images.to(self.device) if self.device is not None else images


class ColoredParityMNIST(Dataset):
    """Even digits in red, odd digits in blue, normalized to [-1, 1]."""

    def __init__(self, base, parity, color, img_size=32):
        if parity not in (0, 1) or color not in ("red", "blue"):
            raise ValueError((parity, color))
        self.base = base
        self.indices = [
            i for i, target in enumerate(base.targets)
            if int(target) % 2 == parity
        ]
        self.color = color
        self.img_size = img_size

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, index):
        image, label = self.base[self.indices[index]]
        gray = TF.resize(TF.to_tensor(image), [self.img_size, self.img_size])
        zeros = torch.zeros_like(gray)
        if self.color == "red":
            rgb = torch.cat((gray, zeros, zeros), dim=0)
        else:
            rgb = torch.cat((zeros, zeros, gray), dim=0)
        return rgb.mul(2).sub(1), int(label)


def build_mnist_red_even_blue_odd(data_root, img_size=32):
    """Source: all even MNIST digits in red. Target: all odd digits in blue.
    Train+test are combined without a split, matching the paper setup."""
    train = MNIST(data_root, train=True, download=True)
    test = MNIST(data_root, train=False, download=True)
    source = ConcatDataset([
        ColoredParityMNIST(train, 0, "red", img_size),
        ColoredParityMNIST(test, 0, "red", img_size),
    ])
    target = ConcatDataset([
        ColoredParityMNIST(train, 1, "blue", img_size),
        ColoredParityMNIST(test, 1, "blue", img_size),
    ])
    return source, target


class Pix2PixProductPhotoDataset(Dataset):
    """Loads the RGB-photo half of pix2pix edge|photo product images
    (edges2handbags / edges2shoes) and projects it onto a single color
    channel, matching the paper's red-handbag / blue-shoe construction."""

    EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}

    def __init__(self, root, channel_view, img_size=64):
        if channel_view not in {"R", "B"}:
            raise ValueError("channel_view must be 'R' or 'B'")
        self.root = Path(root)
        self.channel_view = channel_view
        self.files = sorted(
            path for path in self.root.rglob("*")
            if path.is_file() and path.suffix.lower() in self.EXTENSIONS
        )
        if not self.files:
            raise RuntimeError(f"No product images found below {self.root}")
        self.resize = Resize((img_size, img_size))

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):
        with Image.open(self.files[index]) as image:
            image = image.convert("RGB")
            width, height = image.size
            if width < 2:
                raise ValueError(f"Invalid paired image: {self.files[index]}")
            # pix2pix edges2shoes/edges2handbags store edge|RGB-photo pairs.
            photo = image.crop((width // 2, 0, width, height))
            rgb = TF.to_tensor(self.resize(photo))
        normalized = rgb.mul(2).sub(1)
        if self.channel_view == "R":
            normalized = torch.cat((
                normalized[:1],
                torch.full_like(normalized[1:], 1.0)), dim=0)
        else:
            normalized = torch.cat((
                torch.full_like(normalized[:2], 1.0),
                normalized[2:3]), dim=0)
        return normalized, 0


def build_handbags_to_shoes(data_root, img_size=64):
    """Source: red handbags (edges2handbags photo half, R channel kept).
    Target: blue shoes (edges2shoes photo half, B channel kept)."""
    data_root = Path(data_root)
    source = Pix2PixProductPhotoDataset(
        data_root / "edges2handbags", channel_view="R", img_size=img_size)
    target = Pix2PixProductPhotoDataset(
        data_root / "edges2shoes", channel_view="B", img_size=img_size)
    return source, target
