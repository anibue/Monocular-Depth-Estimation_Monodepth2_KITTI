"""MindSpore training entrypoint for Monodepth2 (simplified)."""

import argparse
import os

import mindspore as ms
from mindspore import Tensor, context, nn, ops

from models.monodepth2_mindspore import (
    Monodepth2Model,
    SSIM,
    generate_depth_from_disp,
    photometric_reconstruction_loss,
    smoothness_loss,
)
from src.datasets import create_mindspore_dataset
from src.utils import ensure_dir, load_config, set_seed, setup_logger


def train(cfg, logger):
    set_seed(cfg.get("seed", 42))
    context.set_context(mode=context.GRAPH_MODE, device_target="GPU" if ms.get_context("device_target") == "GPU" else "CPU")

    data_cfg = cfg["data"]
    train_ds = create_mindspore_dataset(
        split_file=os.path.join(data_cfg["split_dir"], data_cfg["train_split"]),
        img_height=data_cfg["img_height"],
        img_width=data_cfg["img_width"],
        batch_size=data_cfg["batch_size"],
        shuffle=True,
        augment=data_cfg.get("augment", {}),
    )

    model = Monodepth2Model(scales=cfg["model"].get("scales", [0, 1, 2, 3]))
    optimizer = nn.Adam(model.trainable_params(), learning_rate=cfg["optimization"]["learning_rate"])
    ssim_module = SSIM()

    def forward_fn(left):
        outputs = model(left)
        disp = outputs[("disp", 0)]
        depth = generate_depth_from_disp(disp, cfg["model"].get("min_depth", 0.1), cfg["model"].get("max_depth", 100.0))
        recon = left  # 简化：未做视差 warp，主要关注管线
        photo = photometric_reconstruction_loss(ssim_module, left, recon)
        smooth = smoothness_loss(disp, left)
        loss = photo + 0.1 * smooth
        return loss, (depth,)

    grad_fn = ops.value_and_grad(forward_fn, None, optimizer.parameters, has_aux=True)

    for epoch in range(cfg["optimization"]["epochs"]):
        total_loss = 0.0
        for step, batch in enumerate(train_ds.create_dict_iterator()):
            left = Tensor(batch["left"])
            (loss, _), grads = grad_fn(left)
            loss = ops.depend(loss, optimizer(grads))
            total_loss += float(loss.asnumpy())
            if (step + 1) % cfg["logging"].get("log_interval", 50) == 0:
                logger.info("Epoch %d Step %d Loss %.4f", epoch + 1, step + 1, float(loss.asnumpy()))
        logger.info("Epoch %d Train Loss %.4f", epoch + 1, total_loss / max(1, train_ds.get_dataset_size()))

        ensure_dir(cfg["logging"]["checkpoint_dir"])
        ckpt_path = os.path.join(cfg["logging"]["checkpoint_dir"], f"epoch_{epoch+1}.ckpt")
        ms.save_checkpoint(model, ckpt_path)
        logger.info("Saved MindSpore checkpoint %s", ckpt_path)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="config/config_mindspore.yaml")
    return parser.parse_args()


def main():
    args = parse_args()
    cfg = load_config(args.config)
    logger = setup_logger()
    train(cfg, logger)


if __name__ == "__main__":
    main()
