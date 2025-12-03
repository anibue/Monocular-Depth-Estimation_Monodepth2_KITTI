"""MindSpore implementation of a simplified Monodepth2."""

from typing import Dict, List, Optional, Tuple

import mindspore as ms
from mindspore import Tensor, nn, ops


def upsample(x: Tensor) -> Tensor:
    return ops.ResizeNearestNeighbor((x.shape[2] * 2, x.shape[3] * 2))(x)


class ConvBlock(nn.Cell):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 3):
        super().__init__()
        pad = kernel_size // 2
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, pad_mode="pad", padding=pad)
        self.elu = nn.ELU()

    def construct(self, x: Tensor) -> Tensor:
        return self.elu(self.conv(x))


class DepthDecoder(nn.Cell):
    """Decoder producing multi-scale disparity maps."""

    def __init__(self, num_ch_enc: List[int], scales: List[int]):
        super().__init__()
        self.scales = scales
        self.num_ch_dec = [16, 32, 64, 128, 256]

        self.upconvs = nn.CellList()
        self.iconvs = nn.CellList()
        for i in range(4, -1, -1):
            num_ch_in = num_ch_enc[i] if i == 4 else self.num_ch_dec[i + 1]
            num_ch_out = self.num_ch_dec[i]
            self.upconvs.insert(0, ConvBlock(num_ch_in, num_ch_out))
            num_ch_in = num_ch_out + (num_ch_enc[i - 1] if i > 0 else 0)
            self.iconvs.insert(0, ConvBlock(num_ch_in, num_ch_out))

        self.dispconvs = nn.CellList([nn.Conv2d(self.num_ch_dec[s], 1, 3, pad_mode="pad", padding=1) for s in scales])

    def construct(self, feats: List[Tensor]) -> Dict[Tuple[str, int], Tensor]:
        outputs: Dict[Tuple[str, int], Tensor] = {}
        x = feats[-1]
        disp_idx = 0
        for i in range(4, -1, -1):
            x = self.upconvs[4 - i](x)
            x = upsample(x)
            if i > 0:
                x = ops.concat((x, feats[i - 1]), axis=1)
            x = self.iconvs[4 - i](x)
            if i in self.scales:
                disp = ops.Sigmoid()(self.dispconvs[disp_idx](x))
                outputs[("disp", i)] = disp
                disp_idx += 1
        return outputs


class EncoderMS(nn.Cell):
    """轻量级编码器，替代 ResNet18。"""

    def __init__(self):
        super().__init__()
        self.layer1 = nn.SequentialCell(
            nn.Conv2d(3, 64, 7, stride=2, pad_mode="pad", padding=3),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2, pad_mode="same"),
        )
        self.layer2 = nn.SequentialCell(nn.Conv2d(64, 64, 3, pad_mode="pad", padding=1), nn.ReLU())
        self.layer3 = nn.SequentialCell(nn.Conv2d(64, 128, 3, stride=2, pad_mode="pad", padding=1), nn.ReLU())
        self.layer4 = nn.SequentialCell(nn.Conv2d(128, 256, 3, stride=2, pad_mode="pad", padding=1), nn.ReLU())
        self.layer5 = nn.SequentialCell(nn.Conv2d(256, 512, 3, stride=2, pad_mode="pad", padding=1), nn.ReLU())

    def construct(self, x: Tensor) -> List[Tensor]:
        feats = []
        x = self.layer1(x)
        feats.append(x)
        x = self.layer2(x)
        feats.append(x)
        x = self.layer3(x)
        feats.append(x)
        x = self.layer4(x)
        feats.append(x)
        x = self.layer5(x)
        feats.append(x)
        return feats


class PoseNetMS(nn.Cell):
    def __init__(self, num_frames_to_predict: int = 1):
        super().__init__()
        self.conv = nn.SequentialCell(
            nn.Conv2d(512, 256, 3, pad_mode="pad", padding=1),
            nn.ELU(),
            nn.Conv2d(256, 6 * num_frames_to_predict, 1),
            nn.ELU(),
        )
        self.num_frames_to_predict = num_frames_to_predict
        self.pool = ops.ReduceMean(keep_dims=False)

    def construct(self, feats: Tensor) -> Tensor:
        pose = self.conv(feats)
        pose = self.pool(pose, (2, 3))
        pose = ops.Reshape()(pose, (-1, self.num_frames_to_predict, 6))
        return 0.01 * pose


class Monodepth2Model(nn.Cell):
    """MindSpore Monodepth2 模型，包含编码器、解码器和位姿网络。"""

    def __init__(self, scales: Optional[List[int]] = None):
        super().__init__()
        scales = scales or [0, 1, 2, 3]
        self.encoder = EncoderMS()
        self.decoder = DepthDecoder([64, 64, 128, 256, 512], scales=scales)
        self.pose_net = PoseNetMS(num_frames_to_predict=1)
        self.scales = scales

    def construct(self, x: Tensor, x_right: Optional[Tensor] = None) -> Dict[Tuple[str, int], Tensor]:
        feats = self.encoder(x)
        outputs = self.decoder(feats)
        if x_right is not None:
            # 拼接左右视图，计算位姿
            pose_in = ops.concat((x, x_right), axis=1)
            outputs["pose"] = self.pose_net(self.encoder.layer5(self.encoder.layer4(self.encoder.layer3(self.encoder.layer2(self.encoder.layer1(pose_in))))))
        return outputs


class SSIM(nn.Cell):
    def __init__(self):
        super().__init__()
        self.pool = nn.AvgPool2d(3, stride=1, pad_mode="same")

    def construct(self, x: Tensor, y: Tensor) -> Tensor:
        mu_x = self.pool(x)
        mu_y = self.pool(y)
        sigma_x = self.pool(x * x) - mu_x * mu_x
        sigma_y = self.pool(y * y) - mu_y * mu_y
        sigma_xy = self.pool(x * y) - mu_x * mu_y
        c1 = 0.01 ** 2
        c2 = 0.03 ** 2
        ssim_n = (2 * mu_x * mu_y + c1) * (2 * sigma_xy + c2)
        ssim_d = (mu_x * mu_x + mu_y * mu_y + c1) * (sigma_x + sigma_y + c2)
        ssim_map = ops.clip_by_value((1 - ssim_n / ssim_d) / 2, 0.0, 1.0)
        return ssim_map


def generate_depth_from_disp(disp: Tensor, min_depth: float = 0.1, max_depth: float = 100.0) -> Tensor:
    scaled_disp = min_depth + (max_depth - min_depth) * disp
    depth = 1.0 / ops.clip_by_value(scaled_disp, min_depth, max_depth)
    return depth


def smoothness_loss(disp: Tensor, image: Tensor) -> Tensor:
    grad_disp_x = ops.abs(disp[:, :, :, :-1] - disp[:, :, :, 1:])
    grad_disp_y = ops.abs(disp[:, :, :-1, :] - disp[:, :, 1:, :])
    grad_img_x = ops.reduce_mean(ops.abs(image[:, :, :, :-1] - image[:, :, :, 1:]), 1, keep_dims=True)
    grad_img_y = ops.reduce_mean(ops.abs(image[:, :, :-1, :] - image[:, :, 1:, :]), 1, keep_dims=True)
    grad_disp_x *= ops.exp(-grad_img_x)
    grad_disp_y *= ops.exp(-grad_img_y)
    return grad_disp_x.mean() + grad_disp_y.mean()


def photometric_reconstruction_loss(ssim: SSIM, target: Tensor, reconstructed: Tensor, alpha: float = 0.85) -> Tensor:
    l1 = ops.abs(target - reconstructed).mean(axis=1, keep_dims=True)
    ssim_val = ssim(target, reconstructed)
    return alpha * ssim_val.mean() + (1 - alpha) * l1.mean()
