"""TensorFlow training entrypoint for Monodepth2 (simplified)."""

import argparse
import os
from typing import Dict

import tensorflow as tf

from models.monodepth2_tf import Monodepth2Model, smoothness_loss, ssim
from src.datasets import create_tf_dataset
from src.utils import ensure_dir, load_config, set_seed, setup_logger


def photometric_loss_tf(target: tf.Tensor, reconstructed: tf.Tensor, alpha: float = 0.85) -> tf.Tensor:
    """简化光度一致性损失。"""
    l1 = tf.reduce_mean(tf.abs(target - reconstructed))
    return alpha * (1 - ssim(target, reconstructed)) + (1 - alpha) * l1


def train(cfg: Dict, logger) -> None:
    set_seed(cfg.get("seed", 42))
    strategy = tf.distribute.MirroredStrategy() if len(tf.config.list_physical_devices("GPU")) > 1 else None

    data_cfg = cfg["data"]
    train_ds = create_tf_dataset(
        split_file=os.path.join(data_cfg["split_dir"], data_cfg["train_split"]),
        img_height=data_cfg["img_height"],
        img_width=data_cfg["img_width"],
        batch_size=data_cfg["batch_size"],
    )
    val_ds = create_tf_dataset(
        split_file=os.path.join(data_cfg["split_dir"], data_cfg["val_split"]),
        img_height=data_cfg["img_height"],
        img_width=data_cfg["img_width"],
        batch_size=data_cfg["batch_size"],
        shuffle=False,
    )

    with (strategy.scope() if strategy else tf.device("/GPU:0" if tf.config.list_physical_devices("GPU") else "/CPU:0")):
        model = Monodepth2Model(
            encoder=cfg["model"].get("encoder", "resnet18"),
            scales=cfg["model"].get("scales", [0, 1, 2, 3]),
        )
        optimizer = tf.keras.optimizers.Adam(cfg["optimization"]["learning_rate"])

    @tf.function
    def train_step(batch):
        with tf.GradientTape() as tape:
            outputs = model(batch, training=True)
            disp = outputs[("disp", 0)]
            # 简化：直接使用输入作为重建目标，真实实现应 warp 邻近帧
            recon = batch
            photo = photometric_loss_tf(batch, recon)
            smooth = smoothness_loss(disp, batch)
            loss = photo + 0.1 * smooth
        grads = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(grads, model.trainable_variables))
        return loss

    @tf.function
    def val_step(batch):
        outputs = model(batch, training=False)
        disp = outputs[("disp", 0)]
        recon = batch
        photo = photometric_loss_tf(batch, recon)
        smooth = smoothness_loss(disp, batch)
        return photo + 0.1 * smooth

    ensure_dir(cfg["logging"]["checkpoint_dir"])
    ckpt = tf.train.Checkpoint(step=tf.Variable(1), optimizer=optimizer, model=model)
    ckpt_manager = tf.train.CheckpointManager(ckpt, cfg["logging"]["checkpoint_dir"], max_to_keep=3)

    for epoch in range(cfg["optimization"]["epochs"]):
        train_losses = []
        for batch in train_ds:
            loss = train_step(batch)
            train_losses.append(loss.numpy())
        val_losses = [val_step(b).numpy() for b in val_ds]
        logger.info("Epoch %d Train %.4f Val %.4f", epoch + 1, float(tf.reduce_mean(train_losses)), float(tf.reduce_mean(val_losses)))
        ckpt_manager.save()


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="config/config_tensorflow.yaml")
    return parser.parse_args()


def main():
    args = parse_args()
    cfg = load_config(args.config)
    logger = setup_logger()
    train(cfg, logger)


if __name__ == "__main__":
    main()
