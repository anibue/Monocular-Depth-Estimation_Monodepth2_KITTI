"""Simplified Monodepth2 implementation in PyTorch with basic depth decoder and pose network."""

from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models


def upsample(x: torch.Tensor) -> torch.Tensor:
    return F.interpolate(x, scale_factor=2, mode="nearest")


class ConvBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 3):
        super().__init__()
        padding = kernel_size // 2
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, padding=padding)
        self.elu = nn.ELU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.elu(self.conv(x))


class DepthDecoder(nn.Module):
    """Decoder that produces multi-scale disparity maps."""

    def __init__(self, num_ch_enc: List[int], scales: List[int]):
        super().__init__()
        self.scales = scales
        self.num_ch_dec = [16, 32, 64, 128, 256]

        # upconvs
        self.convs = nn.ModuleDict()
        for i in range(4, -1, -1):
            num_ch_in = num_ch_enc[i] if i == 4 else self.num_ch_dec[i + 1]
            num_ch_out = self.num_ch_dec[i]
            self.convs[f"upconv_{i}"] = ConvBlock(num_ch_in, num_ch_out)
            num_ch_in = num_ch_out + (num_ch_enc[i - 1] if i > 0 else 0)
            self.convs[f"iconv_{i}"] = ConvBlock(num_ch_in, num_ch_out)

        for s in self.scales:
            self.convs[f"dispconv_{s}"] = nn.Conv2d(self.num_ch_dec[s], 1, 3, 1, 1)

    def forward(self, input_features: List[torch.Tensor]) -> Dict[int, torch.Tensor]:
        outputs: Dict[int, torch.Tensor] = {}

        x = input_features[-1]
        for i in range(4, -1, -1):
            x = self.convs[f"upconv_{i}"](x)
            x = upsample(x)
            if i > 0:
                x = torch.cat([x, input_features[i - 1]], dim=1)
            x = self.convs[f"iconv_{i}"](x)
            if i in self.scales:
                disp = torch.sigmoid(self.convs[f"dispconv_{i}"](x))
                outputs[("disp", i)] = disp
        return outputs


class PoseNet(nn.Module):
    """Pose network that predicts relative pose between two frames."""

    def __init__(self, num_input_images: int = 2, num_frames_to_predict: int = 1):
        super().__init__()
        self.num_frames_to_predict = num_frames_to_predict
        self.encoder = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        self.encoder.fc = nn.Identity()
        self.pose_pred = nn.Sequential(
            nn.Conv2d(512, 256, 3, 1, 1),
            nn.ELU(inplace=True),
            nn.Conv2d(256, 6 * num_frames_to_predict, 1),
        )

    def forward(self, input_images: torch.Tensor) -> torch.Tensor:
        """Predict 6DoF pose vectors.

        Args:
            input_images: Tensor of shape (B, num_images, C, H, W)
        """
        b, n, c, h, w = input_images.shape
        x = input_images.view(b * n, c, h, w)
        feats = self.encoder.conv1(x)
        feats = self.encoder.bn1(feats)
        feats = self.encoder.relu(feats)
        feats = self.encoder.maxpool(feats)
        feats = self.encoder.layer1(feats)
        feats = self.encoder.layer2(feats)
        feats = self.encoder.layer3(feats)
        feats = self.encoder.layer4(feats)
        # reshape back to batch
        feats = feats.view(b, n, feats.shape[1], feats.shape[2], feats.shape[3])
        # use only the last frame features for pose prediction
        feats = feats[:, -1]
        pose = self.pose_pred(feats)
        pose = pose.mean([2, 3]).view(b, self.num_frames_to_predict, 6)
        return 0.01 * pose


class Monodepth2Model(nn.Module):
    """PyTorch Monodepth2 模型，包含编码器、解码器和位姿网络。"""

    def __init__(self, encoder: str = "resnet18", pretrained: bool = True, scales: Optional[List[int]] = None):
        super().__init__()
        scales = scales or [0, 1, 2, 3]
        if encoder == "resnet18":
            self.encoder = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None)
        else:
            self.encoder = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1 if pretrained else None)

        num_ch_enc = [64, 64, 128, 256, 512] if encoder == "resnet18" else [64, 256, 512, 1024, 2048]
        self.depth_decoder = DepthDecoder(num_ch_enc=num_ch_enc, scales=scales)
        self.pose_net = PoseNet(num_input_images=2, num_frames_to_predict=1)
        self.scales = scales

    def forward(self, x: torch.Tensor, x_right: Optional[torch.Tensor] = None) -> Dict[Tuple[str, int], torch.Tensor]:
        """前向传播：返回多尺度视差和可选位姿。

        Args:
            x: 左图像 tensor，形状 (B, 3, H, W)
            x_right: 右图像 tensor，可选
        """
        features = self._encode(x)
        outputs = self.depth_decoder(features)
        if x_right is not None:
            pose_input = torch.stack([x, x_right], dim=1)
            outputs["pose"] = self.pose_net(pose_input)
        return outputs

    def _encode(self, x: torch.Tensor) -> List[torch.Tensor]:
        feats = []
        x = self.encoder.conv1(x)
        x = self.encoder.bn1(x)
        x = self.encoder.relu(x)
        feats.append(x)
        x = self.encoder.maxpool(x)
        x = self.encoder.layer1(x)
        feats.append(x)
        x = self.encoder.layer2(x)
        feats.append(x)
        x = self.encoder.layer3(x)
        feats.append(x)
        x = self.encoder.layer4(x)
        feats.append(x)
        return feats


class SSIM(nn.Module):
    """基于论文的简化版 SSIM，用于光度一致性损失。"""

    def __init__(self):
        super().__init__()
        self.mu_x_pool = nn.AvgPool2d(3, 1, padding=1)
        self.mu_y_pool = nn.AvgPool2d(3, 1, padding=1)
        self.sig_x_pool = nn.AvgPool2d(3, 1, padding=1)
        self.sig_y_pool = nn.AvgPool2d(3, 1, padding=1)
        self.sig_xy_pool = nn.AvgPool2d(3, 1, padding=1)
        self.refl = nn.ReflectionPad2d(1)

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        x = self.refl(x)
        y = self.refl(y)
        mu_x = self.mu_x_pool(x)
        mu_y = self.mu_y_pool(y)

        sig_x = self.sig_x_pool(x * x) - mu_x * mu_x
        sig_y = self.sig_y_pool(y * y) - mu_y * mu_y
        sig_xy = self.sig_xy_pool(x * y) - mu_x * mu_y

        c1 = 0.01 ** 2
        c2 = 0.03 ** 2

        ssim_n = (2 * mu_x * mu_y + c1) * (2 * sig_xy + c2)
        ssim_d = (mu_x ** 2 + mu_y ** 2 + c1) * (sig_x + sig_y + c2)
        ssim = ssim_n / ssim_d
        ssim = torch.clamp((1 - ssim) / 2, 0, 1)
        return ssim


def generate_depth_from_disp(disp: torch.Tensor, min_depth: float = 0.1, max_depth: float = 100.0) -> torch.Tensor:
    """把视差转换为深度。"""
    scaled_disp = min_depth + (max_depth - min_depth) * disp
    depth = 1.0 / torch.clamp(scaled_disp, min=min_depth)
    return depth


def smoothness_loss(disp: torch.Tensor, image: torch.Tensor) -> torch.Tensor:
    """深度平滑正则，鼓励相邻像素平滑。"""
    grad_disp_x = torch.abs(disp[:, :, :, :-1] - disp[:, :, :, 1:])
    grad_disp_y = torch.abs(disp[:, :, :-1, :] - disp[:, :, 1:, :])

    grad_img_x = torch.mean(torch.abs(image[:, :, :, :-1] - image[:, :, :, 1:]), dim=1, keepdim=True)
    grad_img_y = torch.mean(torch.abs(image[:, :, :-1, :] - image[:, :, 1:, :]), dim=1, keepdim=True)

    grad_disp_x *= torch.exp(-grad_img_x)
    grad_disp_y *= torch.exp(-grad_img_y)
    return grad_disp_x.mean() + grad_disp_y.mean()


def photometric_reconstruction_loss(
    ssim: SSIM, target: torch.Tensor, reconstructed: torch.Tensor, alpha: float = 0.85
) -> torch.Tensor:
    """光度重建损失：结合 SSIM 和 L1。
    alpha 控制 SSIM 与 L1 的权重。
    """
    l1 = torch.abs(target - reconstructed).mean(1, True)
    ssim_val = ssim(target, reconstructed)
    return alpha * ssim_val.mean() + (1 - alpha) * l1.mean()

