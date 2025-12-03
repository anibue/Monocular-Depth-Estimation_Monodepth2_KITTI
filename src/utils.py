"""Utility helpers for configuration, logging, seeding, and checkpoint I/O."""

import logging
import os
import random
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import yaml

try:
    import torch
except ImportError:  # pragma: no cover - TF-only environments
    torch = None


def setup_logger(name: str = "monodepth2", level: int = logging.INFO) -> logging.Logger:
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler()
        fmt = logging.Formatter("[%(asctime)s][%(levelname)s] %(message)s")
        handler.setFormatter(fmt)
        logger.addHandler(handler)
    logger.setLevel(level)
    return logger


def load_config(config_path: str) -> Dict[str, Any]:
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    return cfg


def set_seed(seed: int) -> None:
    """设置随机种子，确保可复现。"""
    random.seed(seed)
    np.random.seed(seed)
    if torch is not None:
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


def get_device() -> Optional[str]:
    if torch is None:
        return None
    return "cuda" if torch.cuda.is_available() else "cpu"


def ensure_dir(path: str) -> None:
    Path(path).mkdir(parents=True, exist_ok=True)


def save_checkpoint(state: Dict[str, Any], path: str) -> None:
    if torch is None:
        return
    ensure_dir(str(Path(path).parent))
    torch.save(state, path)


def load_checkpoint(path: str, device: Optional[str] = None) -> Any:
    if torch is None:
        return None
    map_location = None if device is None else device
    return torch.load(path, map_location=map_location)


def count_parameters(model: Any) -> int:
    if torch is None:
        return 0
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def resolve_path(base_dir: str, relative: str) -> str:
    return str(Path(base_dir) / relative)

