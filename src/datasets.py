"""Dataset utilities for KITTI."""

import os
from pathlib import Path
from typing import Callable, Dict, Optional, Tuple

import numpy as np
from PIL import Image

from src.utils import ensure_dir

try:
    import torch
    from torch.utils.data import DataLoader, Dataset
    import torchvision.transforms as T
except ImportError:  # pragma: no cover - TF-only env
    torch = None
    DataLoader = object
    Dataset = object
    T = None

try:
    import tensorflow as tf
except ImportError:  # pragma: no cover
    tf = None

try:
    import mindspore as ms
    import mindspore.dataset as msds
except ImportError:  # pragma: no cover
    ms = None
    msds = None


def read_split_file(split_path: Path) -> Tuple[Path, ...]:
    lines = split_path.read_text().splitlines()
    return tuple(Path(line) for line in lines if line.strip())


def default_pytorch_transform(img_height: int, img_width: int, augment: Optional[Dict] = None) -> Callable:
    augment = augment or {}
    t_list = [T.Resize((img_height, img_width))]
    if augment.get("color_jitter"):
        t_list.append(T.ColorJitter(0.2, 0.2, 0.2, 0.1))
    t_list.append(T.ToTensor())
    return T.Compose(t_list)


class KITTIDatasetTorch(Dataset):
    """PyTorch 数据集，读取单目或立体图像。"""

    def __init__(
        self,
        split_file: str,
        img_height: int,
        img_width: int,
        augment: Optional[Dict] = None,
    ):
        super().__init__()
        self.samples = read_split_file(Path(split_file))
        self.transform = default_pytorch_transform(img_height, img_width, augment)
        self.augment = augment or {}

    def __len__(self) -> int:
        return len(self.samples)

    def _load_image(self, path: Path) -> Image.Image:
        return Image.open(path).convert("RGB")

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:  # type: ignore
        left_path = self.samples[idx]
        right_path = None
        if "image_02" in left_path.as_posix():
            right_candidate = Path(left_path.as_posix().replace("image_02", "image_03"))
            if right_candidate.exists():
                right_path = right_candidate

        left_img = self._load_image(left_path)
        right_img = self._load_image(right_path) if right_path else None

        if self.augment.get("flip") and np.random.rand() > 0.5:
            left_img = left_img.transpose(Image.FLIP_LEFT_RIGHT)
            if right_img:
                right_img = right_img.transpose(Image.FLIP_LEFT_RIGHT)

        left_tensor = self.transform(left_img)
        sample = {"left": left_tensor}
        if right_img:
            sample["right"] = self.transform(right_img)
        return sample


def create_pytorch_dataloader(
    split_file: str,
    img_height: int,
    img_width: int,
    batch_size: int,
    num_workers: int,
    augment: Optional[Dict] = None,
    shuffle: bool = True,
) -> DataLoader:
    dataset = KITTIDatasetTorch(split_file, img_height, img_width, augment)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, drop_last=True)


def _tf_parse(path: bytes, img_height: int, img_width: int):
    img = tf.io.read_file(path)
    img = tf.image.decode_image(img, channels=3)
    img = tf.image.resize(img, [img_height, img_width])
    img = tf.cast(img, tf.float32) / 255.0
    return img


def create_tf_dataset(
    split_file: str, img_height: int, img_width: int, batch_size: int, shuffle: bool = True
) -> "tf.data.Dataset":
    """TensorFlow 数据集，用于 Keras 训练。"""
    if tf is None:
        raise ImportError("TensorFlow is required for TF dataset creation.")
    paths = read_split_file(Path(split_file))
    ds = tf.data.Dataset.from_tensor_slices([p.as_posix() for p in paths])
    if shuffle:
        ds = ds.shuffle(buffer_size=len(paths))
    ds = ds.map(lambda p: _tf_parse(p, img_height, img_width), num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    return ds


class KITTIDatasetMS:
    """MindSpore 数据集生成器。"""

    def __init__(self, split_file: str, img_height: int, img_width: int, augment: Optional[Dict] = None):
        self.samples = read_split_file(Path(split_file))
        self.img_height = img_height
        self.img_width = img_width
        self.augment = augment or {}

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx: int):
        path = self.samples[idx]
        img = Image.open(path).convert("RGB").resize((self.img_width, self.img_height))
        if self.augment.get("flip") and np.random.rand() > 0.5:
            img = img.transpose(Image.FLIP_LEFT_RIGHT)
        arr = np.array(img).astype(np.float32) / 255.0
        arr = np.transpose(arr, (2, 0, 1))  # C,H,W
        return arr


def create_mindspore_dataset(
    split_file: str, img_height: int, img_width: int, batch_size: int, shuffle: bool = True, augment: Optional[Dict] = None
):
    if ms is None or msds is None:
        raise ImportError("MindSpore is required for MindSpore dataset creation.")
    generator = KITTIDatasetMS(split_file, img_height, img_width, augment)
    ds = msds.GeneratorDataset(generator, column_names=["left"], shuffle=shuffle)
    ds = ds.batch(batch_size, drop_remainder=True)
    return ds
