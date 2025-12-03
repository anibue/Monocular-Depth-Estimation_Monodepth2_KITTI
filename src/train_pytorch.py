"""PyTorch training entrypoint for Monodepth2 on KITTI."""

import argparse
import os
from pathlib import Path
from typing import Dict

import torch
import torch.nn.functional as F
from torch import nn, optim
from torch.utils.data import DataLoader

from models.monodepth2_pytorch import (
    Monodepth2Model,
    SSIM,
    generate_depth_from_disp,
    photometric_reconstruction_loss,
    smoothness_loss,
)
from src.datasets import create_pytorch_dataloader
from src.utils import count_parameters, ensure_dir, get_device, load_config, save_checkpoint, set_seed, setup_logger


def build_dataloaders(cfg: Dict) -> Dict[str, DataLoader]:
    data_cfg = cfg["data"]
    train_loader = create_pytorch_dataloader(
        split_file=data_cfg["split_dir"] + "/" + data_cfg["train_split"],
        img_height=data_cfg["img_height"],
        img_width=data_cfg["img_width"],
        batch_size=data_cfg["batch_size"],
        num_workers=data_cfg.get("num_workers", 4),
        augment=data_cfg.get("augment", {}),
        shuffle=True,
    )
    val_loader = create_pytorch_dataloader(
        split_file=data_cfg["split_dir"] + "/" + data_cfg["val_split"],
        img_height=data_cfg["img_height"],
        img_width=data_cfg["img_width"],
        batch_size=data_cfg["batch_size"],
        num_workers=data_cfg.get("num_workers", 4),
        augment={"flip": False, "color_jitter": False},
        shuffle=False,
    )
    return {"train": train_loader, "val": val_loader}


def warp_right_to_left(right: torch.Tensor, disp: torch.Tensor) -> torch.Tensor:
    """使用视差将右图像 warp 到左图像坐标系 (简化版)。"""
    b, c, h, w = right.shape
    # 归一化网格
    grid_x, grid_y = torch.meshgrid(
        torch.linspace(-1, 1, w, device=right.device),
        torch.linspace(-1, 1, h, device=right.device),
        indexing="xy",
    )
    grid = torch.stack((grid_x, grid_y), dim=2)
    grid = grid.unsqueeze(0).repeat(b, 1, 1, 1)
    # disp 范围 [0,1]，乘以 2 映射到 [-1,1] 平移
    grid[:, :, :, 0] = grid[:, :, :, 0] - disp.squeeze(1) * 2
    reconstructed = F.grid_sample(right, grid, padding_mode="border", align_corners=True)
    return reconstructed


def train(cfg: Dict, logger) -> None:
    device = get_device() or "cpu"
    set_seed(cfg.get("seed", 42))
    logger.info("Using device: %s", device)

    loaders = build_dataloaders(cfg)
    model = Monodepth2Model(
        encoder=cfg["model"].get("encoder", "resnet18"),
        pretrained=cfg["model"].get("pretrained", True),
        scales=cfg["model"].get("scales", [0, 1, 2, 3]),
    ).to(device)
    logger.info("Model params: %.2f M", count_parameters(model) / 1e6)

    optimizer = optim.Adam(
        model.parameters(),
        lr=cfg["optimization"]["learning_rate"],
        weight_decay=cfg["optimization"].get("weight_decay", 0.0),
    )
    scheduler = optim.lr_scheduler.StepLR(
        optimizer,
        step_size=cfg["optimization"].get("scheduler_step_size", 10),
        gamma=cfg["optimization"].get("scheduler_gamma", 0.5),
    )
    scaler = torch.cuda.amp.GradScaler(enabled=cfg["optimization"].get("use_amp", False))
    ssim_module = SSIM()

    ensure_dir(cfg["logging"]["checkpoint_dir"])

    for epoch in range(cfg["optimization"]["epochs"]):
        model.train()
        total_loss = 0.0
        for step, batch in enumerate(loaders["train"]):
            left = batch["left"].to(device)
            right = batch.get("right")
            if right is not None:
                right = right.to(device)
            optimizer.zero_grad()
            with torch.cuda.amp.autocast(enabled=cfg["optimization"].get("use_amp", False)):
                outputs = model(left, right)
                disp = outputs[("disp", 0)]
                depth = generate_depth_from_disp(
                    disp, cfg["model"].get("min_depth", 0.1), cfg["model"].get("max_depth", 100.0)
                )

                if right is not None:
                    recon = warp_right_to_left(right, disp)
                    photo_loss = photometric_reconstruction_loss(ssim_module, left, recon)
                else:
                    photo_loss = torch.tensor(0.0, device=device)

                smooth_loss = smoothness_loss(disp, left)
                loss = photo_loss + 0.1 * smooth_loss

            scaler.scale(loss).backward()
            if cfg["optimization"].get("grad_clip_norm"):
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(model.parameters(), cfg["optimization"]["grad_clip_norm"])
            scaler.step(optimizer)
            scaler.update()
            total_loss += loss.item()

            if (step + 1) % cfg["logging"].get("log_interval", 50) == 0:
                logger.info("Epoch %d Step %d Loss %.4f", epoch + 1, step + 1, loss.item())

        scheduler.step()
        logger.info("Epoch %d train loss %.4f", epoch + 1, total_loss / max(1, len(loaders["train"])))

        if (epoch + 1) % cfg["logging"].get("save_every", 1) == 0:
            ckpt_path = os.path.join(cfg["logging"]["checkpoint_dir"], f"epoch_{epoch+1}.pth")
            save_checkpoint({"model": model.state_dict(), "optimizer": optimizer.state_dict(), "epoch": epoch}, ckpt_path)
            logger.info("Saved checkpoint %s", ckpt_path)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="config/config_pytorch.yaml")
    return parser.parse_args()


def main():
    args = parse_args()
    cfg = load_config(args.config)
    logger = setup_logger()
    train(cfg, logger)


if __name__ == "__main__":
    main()
