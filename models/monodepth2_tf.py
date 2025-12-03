"""TensorFlow / Keras version of a simplified Monodepth2 implementation."""

from typing import Dict, List, Optional, Tuple

import tensorflow as tf
from tensorflow.keras import layers, models as kmodels


def conv_block(x, filters, kernel_size=3):
    x = layers.Conv2D(filters, kernel_size, padding="same")(x)
    x = layers.ELU()(x)
    return x


def upsample(x):
    return tf.image.resize(x, [tf.shape(x)[1] * 2, tf.shape(x)[2] * 2], method="nearest")


def build_encoder(encoder: str = "resnet18"):
    """Use TF Keras Applications as encoder backbone."""
    if encoder == "resnet50":
        base = tf.keras.applications.ResNet50(include_top=False, weights="imagenet")
    else:
        # TensorFlow 没有官方 resnet18，这里用 MobileNetV2 作为轻量近似
        base = tf.keras.applications.MobileNetV2(include_top=False, weights="imagenet")
    outputs = [base.get_layer(name).output for name in [l.name for l in base.layers if "block_6_expand" in l.name][-1:]]
    encoder_model = kmodels.Model(inputs=base.input, outputs=base.outputs)
    return encoder_model


class Monodepth2Model(tf.keras.Model):
    """TensorFlow Monodepth2 模型，部分模块为近似实现。"""

    def __init__(self, encoder: str = "resnet18", scales: Optional[List[int]] = None, **kwargs):
        super().__init__(**kwargs)
        self.scales = scales or [0, 1, 2, 3]
        self.encoder_type = encoder
        self.encoder = build_encoder(encoder)
        # Simple decoder
        self.dec_convs = {
            s: layers.Conv2D(1, 3, padding="same", activation="sigmoid") for s in self.scales
        }
        self.pose_conv = tf.keras.Sequential(
            [
                layers.Conv2D(256, 3, padding="same", activation="elu"),
                layers.Conv2D(6, 1, padding="same"),
                layers.GlobalAveragePooling2D(),
            ]
        )

    def call(self, x: tf.Tensor, x_right: Optional[tf.Tensor] = None, training=False) -> Dict[Tuple[str, int], tf.Tensor]:
        # 编码特征
        feats = self.encoder(x, training=training)
        # 解码为单尺度视差 (简化)
        outputs: Dict[Tuple[str, int], tf.Tensor] = {}
        up = feats
        for s in self.scales:
            up = conv_block(up, 64)
            up = upsample(up)
            disp = self.dec_convs[s](up)
            outputs[("disp", s)] = disp
        if x_right is not None:
            stacked = tf.concat([x, x_right], axis=-1)
            pose = self.pose_conv(stacked)
            pose = tf.reshape(pose, [-1, 1, 6])
            outputs["pose"] = 0.01 * pose
        return outputs

    # TODO: 完整匹配 PyTorch 版本的多尺度特征与光度损失，这里提供可运行骨架。


def ssim(x: tf.Tensor, y: tf.Tensor) -> tf.Tensor:
    return tf.reduce_mean(tf.image.ssim(x, y, max_val=1.0))


def smoothness_loss(disp: tf.Tensor, image: tf.Tensor) -> tf.Tensor:
    dy_disp = tf.abs(disp[:, 1:, :, :] - disp[:, :-1, :, :])
    dx_disp = tf.abs(disp[:, :, 1:, :] - disp[:, :, :-1, :])
    dy_img = tf.abs(image[:, 1:, :, :] - image[:, :-1, :, :])
    dx_img = tf.abs(image[:, :, 1:, :] - image[:, :, :-1, :])
    weight_x = tf.exp(-tf.reduce_mean(dx_img, axis=-1, keepdims=True))
    weight_y = tf.exp(-tf.reduce_mean(dy_img, axis=-1, keepdims=True))
    return tf.reduce_mean(dx_disp * weight_x) + tf.reduce_mean(dy_disp * weight_y)

