"""Run inference with PyTorch Monodepth2."""

import argparse
from pathlib import Path
from typing import List

import numpy as np
import torch
import torchvision.transforms as T
from PIL import Image

from models.monodepth2_pytorch import Monodepth2Model, generate_depth_from_disp
from src.utils import ensure_dir, get_device, load_checkpoint, load_config, setup_logger
from src.visualization import save_depth_map


def load_image(path: Path, img_height: int, img_width: int) -> torch.Tensor:
    transform = T.Compose([T.Resize((img_height, img_width)), T.ToTensor()])
    img = Image.open(path).convert("RGB")
    return transform(img).unsqueeze(0)


def infer_single(model: Monodepth2Model, device: str, img_path: Path, img_h: int, img_w: int, min_d: float, max_d: float):
    """单张图像推理。"""
    tensor = load_image(img_path, img_h, img_w).to(device)
    with torch.no_grad():
        outputs = model(tensor)
        disp = outputs[("disp", 0)]
        depth = generate_depth_from_disp(disp, min_d, max_d)
    depth_np = depth.squeeze().cpu().numpy()
    return depth_np


def gather_images(input_path: Path) -> List[Path]:
    if input_path.is_file():
        return [input_path]
    images = []
    for ext in ("*.png", "*.jpg", "*.jpeg"):
        images.extend(input_path.rglob(ext))
    return images


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="config/config_pytorch.yaml")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--input_image", type=str, required=True, help="Image file or directory.")
    parser.add_argument("--output", type=str, default="outputs/pytorch")
    args = parser.parse_args()

    cfg = load_config(args.config)
    logger = setup_logger()
    device = get_device() or "cpu"

    model = Monodepth2Model(
        encoder=cfg["model"].get("encoder", "resnet18"),
        pretrained=False,
        scales=cfg["model"].get("scales", [0, 1, 2, 3]),
    ).to(device)

    ckpt = load_checkpoint(args.checkpoint, device)
    if ckpt and "model" in ckpt:
        model.load_state_dict(ckpt["model"])
    model.eval()

    images = gather_images(Path(args.input_image))
    logger.info("Processing %d images", len(images))
    ensure_dir(args.output)

    for img_path in images:
        depth = infer_single(
            model,
            device,
            img_path,
            cfg["data"]["img_height"],
            cfg["data"]["img_width"],
            cfg["model"].get("min_depth", 0.1),
            cfg["model"].get("max_depth", 100.0),
        )
        out_path = Path(args.output) / f"{img_path.stem}_depth.png"
        save_depth_map(depth, out_path.as_posix())
        logger.info("Saved depth to %s", out_path)


if __name__ == "__main__":
    main()
