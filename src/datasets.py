import math
import os
from pathlib import Path

import numpy as np
import torch
from scipy import ndimage
from torch.utils.data import ConcatDataset, DataLoader, Dataset, Subset
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


def _uniform(batch_size, device):
    return 2 * torch.rand(batch_size, 1, device=device) - 1


def _grid_values(batch_size, device, n_grid=5):
    """Uniform over the odd grid points {-3/4, -1/4, 1/4, 3/4} of linspace(-1, 1, 9)."""
    grid = torch.linspace(-1, 1, 2 * n_grid - 1, device=device)
    idx = 2 * torch.randint(1, n_grid, (batch_size,), device=device) - 1
    return grid[idx].unsqueeze(1)


def _segment_mu(batch_size, device="cpu"):
    x = _uniform(batch_size, device)
    return torch.cat([x, torch.zeros(batch_size, 1, device=device)], dim=1)


def _perpendicular_nu(batch_size, device="cpu"):
    y = _uniform(batch_size, device)
    return torch.cat([torch.zeros(batch_size, 1, device=device), y], dim=1)


def _horizontal_nu(batch_size, device="cpu"):
    x = _uniform(batch_size, device)
    return torch.cat([x, torch.ones(batch_size, 1, device=device)], dim=1)


def _one_to_many_nu(batch_size, device="cpu"):
    x = _uniform(batch_size, device)
    sign = torch.randint(0, 2, (batch_size, 1), device=device) * 2 - 1
    return torch.cat([x, sign.float()], dim=1)


def _grid_mu(batch_size, device="cpu"):
    y = _grid_values(batch_size, device)
    x = _uniform(batch_size, device)
    return torch.cat([x, y], dim=1)


def _grid_nu(batch_size, device="cpu"):
    x = _grid_values(batch_size, device)
    y = _uniform(batch_size, device)
    return torch.cat([x, y], dim=1)


# name -> (sample_mu, sample_nu, W2^2(mu, nu) = min E||x - y||^2, derived analytically)
TOY_DATASETS = {
    # mu on [-1, 1] x {0}, nu on {0} x [-1, 1]: every coupling costs 1/3 + 1/3.
    "perpendicular": (_segment_mu, _perpendicular_nu, 2 / 3),
    # nu is mu shifted up by 1.
    "horizontal": (_segment_mu, _horizontal_nu, 1.0),
    # nu on the segments y = +1 and y = -1; x is kept, the mass splits evenly.
    "one_to_many": (_segment_mu, _one_to_many_nu, 1.0),
    # "Grid" / Multi-Perpendicular: four horizontal segments to four vertical ones.
    "grid": (_grid_mu, _grid_nu, 1 / 24),
}


def toy_dataset(name):
    """2D datasets of Choi et al. (2025) used in notebooks 1-4: returns
    (sample_mu, sample_nu, W2^2) with sample_*(batch_size, device="cpu")."""
    if name not in TOY_DATASETS:
        raise ValueError(f"unknown toy dataset {name!r}; choose from {list(TOY_DATASETS)}")
    return TOY_DATASETS[name]


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


def product_foreground_mask(photo, edges=None, white_thresh=0.9,
                            pure_white_thresh=0.97, min_hole_area=16):
    """Boolean HxW foreground mask for a product photo on a white background.

    photo: HxWx3 float array in [0, 1]. edges: optional HxW bool array of
    pix2pix edge lines, used as a barrier so the background flood fill does
    not leak into white objects through gaps in their outline.

    Background = near-white pixels connected to the image border, plus
    enclosed near-white regions that are pure white (e.g. the hole inside a
    handbag handle) -- shaded white product surfaces stay foreground."""
    min_channel = photo.min(axis=2)
    near_white = min_channel > white_thresh
    fillable = near_white if edges is None else (
        near_white & ~ndimage.binary_dilation(edges, iterations=1))

    labels, n_labels = ndimage.label(fillable)
    border_labels = np.unique(np.concatenate(
        (labels[0], labels[-1], labels[:, 0], labels[:, -1])))
    background = np.isin(labels, border_labels[border_labels > 0])

    if pure_white_thresh is not None and n_labels > 0:
        ids = np.arange(1, n_labels + 1)
        means = ndimage.mean(min_channel, labels, ids)
        sizes = ndimage.sum(np.ones_like(min_channel), labels, ids)
        holes = ids[(means > pure_white_thresh) & (sizes >= min_hole_area)]
        background |= np.isin(labels, holes)

    # Reclaim the white rim left under the dilated edge barrier.
    background |= near_white & ndimage.binary_dilation(background, iterations=2)
    return ~background


class Pix2PixProductPhotoDataset(Dataset):
    """Loads the RGB-photo half of pix2pix edge|photo product images
    (edges2handbags / edges2shoes) and projects it onto a single color
    channel, matching the paper's red-handbag / blue-shoe construction.

    With remove_background=True the white background is masked out, so the
    image is (R, 0, 0) or (0, 0, B) on the object and black elsewhere.
    Otherwise the unused channels are filled with 1 (white background)."""

    EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}

    def __init__(self, root, channel_view, img_size=64, remove_background=True,
                 mask_kwargs=None):
        if channel_view not in {"R", "B"}:
            raise ValueError("channel_view must be 'R' or 'B'")
        self.root = Path(root)
        self.channel_view = channel_view
        self.remove_background = remove_background
        self.mask_kwargs = mask_kwargs or {}
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
            if self.remove_background:
                edge = image.crop((0, 0, width // 2, height)).convert("L")
                return self._masked_channel(photo, edge), 0
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

    def border_white_fraction(self, index, width=4, white_thresh=0.9):
        """Fraction of the photo's outer `width`-pixel ring that is near-white;
        low values flag photos shot on a non-white background."""
        with Image.open(self.files[index]) as image:
            image = image.convert("RGB")
            photo = image.crop((image.width // 2, 0, image.width, image.height))
        white = np.asarray(photo, dtype=np.float32).min(axis=2) / 255.0 > white_thresh
        ring = np.concatenate((white[:width].ravel(), white[-width:].ravel(),
                               white[:, :width].ravel(), white[:, -width:].ravel()))
        return float(ring.mean())

    def _masked_channel(self, photo, edge):
        photo_np = np.asarray(photo, dtype=np.float32) / 255.0
        edges_np = np.asarray(edge, dtype=np.float32) / 255.0 < 0.5
        mask = product_foreground_mask(photo_np, edges_np, **self.mask_kwargs)
        # Mask and channel are downsampled together (bilinear, antialiased)
        # so object boundaries stay soft instead of aliased.
        rgb = self.resize(TF.to_tensor(photo))
        alpha = self.resize(torch.from_numpy(mask).float()[None])
        channel = 0 if self.channel_view == "R" else 2
        out = torch.zeros_like(rgb)
        out[channel] = rgb[channel] * alpha[0]
        return out.mul(2).sub(1)


class CachedChannelDataset(Dataset):
    """In-memory Pix2PixProductPhotoDataset replacement read from a
    (N, H, W) uint8 cache of the one kept channel; the other two channels are
    filled with fill_value (-1 with the background removed, +1 without)."""

    def __init__(self, cache_path, channel_view, fill_value):
        self.images = torch.from_numpy(np.load(cache_path))
        self.channel = 0 if channel_view == "R" else 2
        self.fill_value = fill_value

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        channel = self.images[index].float().div(255).mul(2).sub(1)
        out = torch.full((3, *channel.shape), self.fill_value)
        out[self.channel] = channel
        return out, 0


_CACHE_SOURCE = None


def _map_over_dataset(dataset, fn, num_workers):
    """pool.map(fn, indices) with the dataset shared through fork instead of
    pickled into every task."""
    global _CACHE_SOURCE
    from multiprocessing import get_context

    _CACHE_SOURCE = dataset
    try:
        with get_context("fork").Pool(
                num_workers, initializer=torch.set_num_threads, initargs=(1,)) as pool:
            return pool.map(fn, range(len(dataset)), chunksize=256)
    finally:
        _CACHE_SOURCE = None


def _border_item(index):
    return _CACHE_SOURCE.border_white_fraction(index)


def _cache_item(index):
    image, _ = _CACHE_SOURCE[index]
    channel = image[0 if _CACHE_SOURCE.channel_view == "R" else 2]
    return channel.add(1).mul(127.5).round().clamp(0, 255).to(torch.uint8).numpy()


def _relative_files(dataset):
    return [str(path.relative_to(dataset.root)) for path in dataset.files]


def _save_cache(array, dataset, cache_path):
    """Save a per-image array aligned with dataset.files, plus that file order
    as <cache>.txt next to it. The array goes through a temporary file so
    concurrent jobs never read a partial cache."""
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = cache_path.with_name(f"{cache_path.stem}.tmp{os.getpid()}.npy")
    np.save(tmp_path, array)
    os.replace(tmp_path, cache_path)
    cache_path.with_suffix(".txt").write_text("\n".join(_relative_files(dataset)) + "\n")


def _cache_is_current(dataset, cache_path):
    """True if cache_path exists and its .txt file list matches dataset.files
    exactly. A cache built from a different image folder (re-downloaded or
    converted data) is rebuilt instead of being silently misaligned."""
    list_path = cache_path.with_suffix(".txt")
    return (cache_path.exists() and list_path.exists()
            and list_path.read_text().splitlines() == _relative_files(dataset))


def build_channel_cache(dataset, cache_path, num_workers=8):
    """Preprocess every image of a Pix2PixProductPhotoDataset once and save
    the kept channel as uint8, plus the source file order next to it."""
    channels = _map_over_dataset(dataset, _cache_item, num_workers)
    _save_cache(np.stack(channels), dataset, Path(cache_path))


def border_white_fractions(dataset, cache_dir, num_workers=8):
    """Per-image border_white_fraction, aligned with dataset.files and cached
    as cache_dir/<folder>_border_white.npy (independent of size/mode)."""
    path = Path(cache_dir) / f"{dataset.root.name}_border_white.npy"
    if _cache_is_current(dataset, path):
        return np.load(path)
    print(f"[cache] computing border whiteness for {len(dataset):,} images", flush=True)
    fractions = np.array(_map_over_dataset(dataset, _border_item, num_workers),
                         dtype=np.float32)
    _save_cache(fractions, dataset, path)
    return fractions


def _product_dataset(data_root, name, channel_view, img_size, remove_background,
                     mask_kwargs, use_cache, num_workers, min_border_white):
    dataset = Pix2PixProductPhotoDataset(
        data_root / name, channel_view=channel_view, img_size=img_size,
        remove_background=remove_background, mask_kwargs=mask_kwargs)
    keep = None
    if min_border_white:
        fractions = border_white_fractions(dataset, data_root / "cache", num_workers)
        keep = np.flatnonzero(fractions >= min_border_white).tolist()
        print(f"[data] {name}: kept {len(keep):,} / {len(dataset):,} photos with "
              f"border whiteness >= {min_border_white}", flush=True)
    if use_cache:
        dataset = _cached_product_dataset(dataset, data_root, name, channel_view,
                                          img_size, remove_background, mask_kwargs,
                                          num_workers)
    return dataset if keep is None else Subset(dataset, keep)


def _cached_product_dataset(dataset, data_root, name, channel_view, img_size,
                            remove_background, mask_kwargs, num_workers):
    mode = "masked" if remove_background else "whitebg"
    if mask_kwargs:
        mode += "_" + "_".join(f"{k}{v}" for k, v in sorted(mask_kwargs.items()))
    cache_path = data_root / "cache" / f"{name}_{channel_view}_{img_size}_{mode}.npy"
    if not _cache_is_current(dataset, cache_path):
        print(f"[cache] building {cache_path} from {len(dataset):,} images", flush=True)
        build_channel_cache(dataset, cache_path, num_workers=num_workers)
    return CachedChannelDataset(
        cache_path, channel_view, fill_value=-1.0 if remove_background else 1.0)


def build_handbags_to_shoes(data_root, img_size=64, remove_background=True,
                            mask_kwargs=None, use_cache=True, num_workers=8,
                            min_border_white=0.5):
    """Source: red handbags (edges2handbags photo half, R channel kept).
    Target: blue shoes (edges2shoes photo half, B channel kept).
    remove_background=True turns the white background black (see
    Pix2PixProductPhotoDataset). With use_cache=True the preprocessed images
    are read from data_root/cache/, which is built on first use.
    Photos whose border is less than min_border_white near-white (shot on a
    non-white background, so masking fails) are dropped; 0/None keeps all."""
    data_root = Path(data_root)
    common = dict(img_size=img_size, remove_background=remove_background,
                  mask_kwargs=mask_kwargs, use_cache=use_cache,
                  num_workers=num_workers, min_border_white=min_border_white)
    source = _product_dataset(data_root, "edges2handbags", "R", **common)
    target = _product_dataset(data_root, "edges2shoes", "B", **common)
    return source, target
