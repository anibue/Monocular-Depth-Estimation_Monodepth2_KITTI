"""Inference script for TensorFlow Monodepth2."""

import argparse
from pathlib import Path
from typing import List

import numpy as np
import tensorflow as tf
from PIL import Image

from models.monodepth2_tf import Monodepth2Model
from src.utils import ensure_dir, load_config, setup_logger
from src.visualization import save_depth_map


def load_image(path: Path, img_h: int, img_w: int) -> np.ndarray:
    img = Image.open(path).convert("RGB").resize((img_w, img_h))
    arr = np.array(img).astype("float32") / 255.0
    return np.expand_dims(arr, axis=0)


def gather_images(input_path: Path) -> List[Path]:
    if input_path.is_file():
        return [input_path]
    images = []
    for ext in ("*.png", "*.jpg", "*.jpeg"):
        images.extend(input_path.rglob(ext))
    return images


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="config/config_tensorflow.yaml")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to TF checkpoint directory or file.")
    parser.add_argument("--input_image", type=str, required=True)
    parser.add_argument("--output", type=str, default="outputs/tensorflow")
    args = parser.parse_args()

    cfg = load_config(args.config)
    logger = setup_logger()

    model = Monodepth2Model(
        encoder=cfg["model"].get("encoder", "resnet18"),
        scales=cfg["model"].get("scales", [0, 1, 2, 3]),
    )
    dummy = np.zeros((1, cfg["data"]["img_height"], cfg["data"]["img_width"], 3), dtype=np.float32)
    model(dummy)  # build weights

    ckpt = tf.train.Checkpoint(model=model)
    ckpt.restore(tf.train.latest_checkpoint(args.checkpoint) or args.checkpoint).expect_partial()
    logger.info("Loaded checkpoint from %s", args.checkpoint)

    images = gather_images(Path(args.input_image))
    ensure_dir(args.output)
    for img_path in images:
        inp = load_image(img_path, cfg["data"]["img_height"], cfg["data"]["img_width"])
        outputs = model(inp, training=False)
        disp = outputs[("disp", 0)][0, :, :, 0]
        depth = 1.0 / (disp + 1e-6)
        out_path = Path(args.output) / f"{img_path.stem}_depth.png"
        save_depth_map(depth, out_path.as_posix())
        logger.info("Saved depth %s", out_path)


if __name__ == "__main__":
    main()
