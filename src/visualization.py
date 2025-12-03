"""Visualization helpers for depth maps."""

from pathlib import Path
from typing import Optional

import matplotlib

matplotlib.use("Agg")  # 避免无界面环境下崩溃
import matplotlib.pyplot as plt
import numpy as np

from src.utils import ensure_dir


def normalize_depth(depth: np.ndarray) -> np.ndarray:
    if depth.max() - depth.min() < 1e-6:
        return depth
    norm = (depth - depth.min()) / (depth.max() - depth.min() + 1e-8)
    return norm


def save_depth_map(depth: np.ndarray, out_path: str, cmap: str = "magma", vmin: Optional[float] = None, vmax: Optional[float] = None) -> None:
    """将深度图保存为可视化 PNG。"""
    ensure_dir(str(Path(out_path).parent))
    depth_norm = normalize_depth(depth)
    plt.imshow(depth_norm, cmap=cmap, vmin=vmin, vmax=vmax)
    plt.axis("off")
    plt.savefig(out_path, bbox_inches="tight", pad_inches=0)
    plt.close()


def save_grayscale(depth: np.ndarray, out_path: str) -> None:
    save_depth_map(depth, out_path, cmap="gray")
