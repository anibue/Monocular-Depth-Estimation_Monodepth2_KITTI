"""Inference with MindSpore Monodepth2."""

import argparse
from pathlib import Path
from typing import List

import mindspore as ms
import numpy as np
from PIL import Image

from models.monodepth2_mindspore import Monodepth2Model, generate_depth_from_disp
from src.utils import ensure_dir, load_config, setup_logger
from src.visualization import save_depth_map


def load_image(path: Path, img_h: int, img_w: int) -> ms.Tensor:
    img = Image.open(path).convert("RGB").resize((img_w, img_h))
    arr = np.array(img).astype(np.float32) / 255.0
    arr = np.transpose(arr, (2, 0, 1))
    return ms.Tensor(arr[None, ...])


def gather_images(input_path: Path) -> List[Path]:
    if input_path.is_file():
        return [input_path]
    images = []
    for ext in ("*.png", "*.jpg", "*.jpeg"):
        images.extend(input_path.rglob(ext))
    return images


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="config/config_mindspore.yaml")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--input_image", type=str, required=True)
    parser.add_argument("--output", type=str, default="outputs/mindspore")
    args = parser.parse_args()

    cfg = load_config(args.config)
    logger = setup_logger()
    ms.set_context(mode=ms.GRAPH_MODE, device_target="CPU")

    model = Monodepth2Model(scales=cfg["model"].get("scales", [0, 1, 2, 3]))
    param_dict = ms.load_checkpoint(args.checkpoint)
    ms.load_param_into_net(model, param_dict)
    model.set_train(False)

    images = gather_images(Path(args.input_image))
    ensure_dir(args.output)
    for img_path in images:
        inp = load_image(img_path, cfg["data"]["img_height"], cfg["data"]["img_width"])
        outputs = model(inp)
        disp = outputs[("disp", 0)][0, 0].asnumpy()
        depth = generate_depth_from_disp(
            ms.Tensor(outputs[("disp", 0)]),
            cfg["model"].get("min_depth", 0.1),
            cfg["model"].get("max_depth", 100.0),
        )[0, 0].asnumpy()
        out_path = Path(args.output) / f"{img_path.stem}_depth.png"
        save_depth_map(depth, out_path.as_posix())
        logger.info("Saved MindSpore depth %s", out_path)


if __name__ == "__main__":
    main()
