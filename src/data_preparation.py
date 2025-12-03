"""Data preparation pipeline for KITTI Monodepth2."""

import argparse
import hashlib
import logging
import os
import random
import zipfile
from pathlib import Path
from typing import List, Tuple

import cv2
import requests
from tqdm import tqdm

from src.utils import ensure_dir, setup_logger

KITTI_URL = "https://s3.eu-central-1.amazonaws.com/avg-kitti/raw_data/2011_09_26_drive_0001_sync.zip"


def download_file(url: str, dest: Path, logger: logging.Logger) -> Path:
    """下载文件并显示进度条。"""
    ensure_dir(str(dest.parent))
    if dest.exists():
        logger.info("File already exists: %s", dest)
        return dest
    logger.info("Downloading %s -> %s", url, dest)
    with requests.get(url, stream=True, timeout=60) as r:
        r.raise_for_status()
        total = int(r.headers.get("content-length", 0))
        with open(dest, "wb") as f, tqdm(total=total, unit="B", unit_scale=True) as pbar:
            for chunk in r.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
                    pbar.update(len(chunk))
    return dest


def verify_checksum(path: Path, checksum: str, logger: logging.Logger) -> bool:
    """简单校验下载完整性。"""
    if not checksum:
        return True
    logger.info("Verifying checksum for %s", path)
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    ok = h.hexdigest() == checksum
    if ok:
        logger.info("Checksum ok")
    else:
        logger.warning("Checksum mismatch: %s != %s", h.hexdigest(), checksum)
    return ok


def unpack_archive(path: Path, dest: Path, logger: logging.Logger) -> None:
    """解压 zip 文件。"""
    logger.info("Unpacking %s -> %s", path, dest)
    ensure_dir(str(dest))
    with zipfile.ZipFile(path, "r") as zf:
        zf.extractall(dest)


def clean_images(root: Path, logger: logging.Logger) -> List[Path]:
    """检测并过滤坏图像，返回保留的路径列表。"""
    valid_paths: List[Path] = []
    for img_path in tqdm(list(root.rglob("*.png")) + list(root.rglob("*.jpg")), desc="Checking images"):
        img = cv2.imread(str(img_path))
        if img is None:
            logger.warning("Corrupted image skipped: %s", img_path)
            continue
        valid_paths.append(img_path)
    logger.info("Valid images: %d", len(valid_paths))
    return valid_paths


def write_split_files(paths: List[Path], split_dir: Path, train_ratio=0.8, val_ratio=0.1) -> None:
    ensure_dir(str(split_dir))
    random.shuffle(paths)
    n = len(paths)
    n_train = int(n * train_ratio)
    n_val = int(n * val_ratio)
    train = paths[:n_train]
    val = paths[n_train : n_train + n_val]
    test = paths[n_train + n_val :]
    (split_dir / "train_files.txt").write_text("\n".join(p.as_posix() for p in train))
    (split_dir / "val_files.txt").write_text("\n".join(p.as_posix() for p in val))
    (split_dir / "test_files.txt").write_text("\n".join(p.as_posix() for p in test))


def prepare_kitti(dataset_root: Path, download: bool, prepare_splits: bool, logger: logging.Logger) -> None:
    raw_dir = dataset_root / "raw"
    processed_dir = dataset_root / "processed"
    split_dir = dataset_root / "splits"

    if download:
        zip_path = dataset_root / "raw_data.zip"
        download_file(KITTI_URL, zip_path, logger)
        unpack_archive(zip_path, raw_dir, logger)

    ensure_dir(str(processed_dir))
    ensure_dir(str(split_dir))

    # 简单复制/处理，真实项目中会做更多几何变换
    valid_images = clean_images(raw_dir, logger)
    for img_path in tqdm(valid_images, desc="Copying to processed"):
        rel = img_path.relative_to(raw_dir)
        dest = processed_dir / rel
        ensure_dir(str(dest.parent))
        if not dest.exists():
            dest.write_bytes(img_path.read_bytes())

    if prepare_splits:
        write_split_files(valid_images, split_dir)
        logger.info("Split files written to %s", split_dir)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="KITTI data preparation for Monodepth2.")
    parser.add_argument("--dataset_root", type=str, default="data/kitti", help="Root directory for KITTI data.")
    parser.add_argument("--download", action="store_true", help="Download KITTI raw subset.")
    parser.add_argument("--prepare_splits", action="store_true", help="Generate train/val/test splits.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logger = setup_logger()
    logger.info("准备 KITTI 数据集，根目录: %s", args.dataset_root)
    prepare_kitti(Path(args.dataset_root), download=args.download, prepare_splits=args.prepare_splits, logger=logger)
    logger.info("数据准备完成")


if __name__ == "__main__":
    main()
