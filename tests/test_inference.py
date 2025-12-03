import tempfile
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from src.inference_pytorch import infer_single

torch = pytest.importorskip("torch")
from models.monodepth2_pytorch import Monodepth2Model  # noqa: E402


def test_infer_single_outputs(tmp_path):
    # 生成假图像
    img = (np.random.rand(32, 64, 3) * 255).astype("uint8")
    img_path = tmp_path / "dummy.png"
    Image.fromarray(img).save(img_path)

    model = Monodepth2Model(scales=[0])
    # 保存并加载模型参数，模拟 checkpoint
    ckpt_path = tmp_path / "model.pth"
    torch.save({"model": model.state_dict()}, ckpt_path)
    model.load_state_dict(torch.load(ckpt_path)["model"])
    model.eval()

    depth = infer_single(model, "cpu", img_path, 32, 64, 0.1, 100.0)
    assert depth.shape == (32, 64)
