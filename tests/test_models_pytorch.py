import pytest

torch = pytest.importorskip("torch")

from models.monodepth2_pytorch import Monodepth2Model


def test_forward_shapes_cpu():
    model = Monodepth2Model()
    x = torch.randn(1, 3, 32, 64)
    outputs = model(x)
    assert ("disp", 0) in outputs
    disp0 = outputs[("disp", 0)]
    assert disp0.shape[2:] == (32, 64)
    y = torch.randn(1, 3, 32, 64)
    outputs = model(x, y)
    assert "pose" in outputs
    assert outputs["pose"].shape == (1, 1, 6)
