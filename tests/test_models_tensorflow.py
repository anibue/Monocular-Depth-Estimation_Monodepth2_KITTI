import pytest

tf = pytest.importorskip("tensorflow")

from models.monodepth2_tf import Monodepth2Model


def test_tf_forward_shapes():
    model = Monodepth2Model(scales=[0])
    x = tf.random.normal([1, 32, 64, 3])
    outputs = model(x, training=False)
    assert ("disp", 0) in outputs
    disp = outputs[("disp", 0)]
    assert disp.shape[1] == 32 and disp.shape[2] == 64
