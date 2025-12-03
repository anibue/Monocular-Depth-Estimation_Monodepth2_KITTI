import pytest

ms = pytest.importorskip("mindspore")

from models.monodepth2_mindspore import Monodepth2Model


def test_mindspore_forward_shapes():
    model = Monodepth2Model(scales=[0])
    x = ms.Tensor(ms.numpy.randn(1, 3, 32, 64), ms.float32)
    outputs = model(x)
    assert ("disp", 0) in outputs
    disp = outputs[("disp", 0)]
    assert disp.shape[2] == 32 and disp.shape[3] == 64
