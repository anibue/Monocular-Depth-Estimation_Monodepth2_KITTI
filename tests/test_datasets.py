import os
from pathlib import Path

import numpy as np
from PIL import Image

from src.datasets import KITTIDatasetTorch, create_pytorch_dataloader


def _make_dummy_split(tmp_path: Path, n: int = 2) -> Path:
    img_dir = tmp_path / "images"
    img_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    for i in range(n):
        arr = (np.random.rand(64, 128, 3) * 255).astype("uint8")
        path = img_dir / f"{i}.png"
        Image.fromarray(arr).save(path)
        paths.append(path)
    split = tmp_path / "train_files.txt"
    split.write_text("\n".join(p.as_posix() for p in paths))
    return split


def test_pytorch_dataset_loading(tmp_path):
    split = _make_dummy_split(tmp_path)
    ds = KITTIDatasetTorch(split.as_posix(), img_height=32, img_width=64, augment={"flip": False})
    sample = ds[0]
    assert "left" in sample
    assert sample["left"].shape[1:] == (32, 64)


def test_pytorch_dataloader(tmp_path):
    split = _make_dummy_split(tmp_path, n=4)
    loader = create_pytorch_dataloader(
        split_file=split.as_posix(), img_height=32, img_width=64, batch_size=2, num_workers=0, shuffle=False
    )
    batch = next(iter(loader))
    assert batch["left"].shape == (2, 3, 32, 64)
