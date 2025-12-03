from .monodepth2_pytorch import Monodepth2Model as Monodepth2ModelTorch

try:
    from .monodepth2_tf import Monodepth2Model as Monodepth2ModelTF  # type: ignore
except ImportError:  # pragma: no cover - allow environments without TF
    Monodepth2ModelTF = None  # type: ignore

__all__ = ["Monodepth2ModelTorch", "Monodepth2ModelTF"]
