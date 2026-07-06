__version__ = "1.0.0"
__author__ = "lxr"
__email__ = "xinran.liu@zju.edu.cn"

from .inversion import MT2DInverter, MT2DInverterWeightedCost, make_weighted_cost_fn
from .forward import MT2DFD_Torch
from .models import MT2DTrueModels

# Optional dense variants (may be absent in minimal installs)
try:
    from .MTinv_2d_dense import MT2DInverterdense  # noqa: F401
except ImportError:
    MT2DInverterdense = None  # type: ignore

try:
    from .MTinv_2d_weighted_cost_dense import MT2DInverterWeightedCost_dense  # noqa: F401
except ImportError:
    MT2DInverterWeightedCost_dense = None  # type: ignore

try:
    from .MT2Ddense import MT2DFD_Torchdense  # noqa: F401
except ImportError:
    MT2DFD_Torchdense = None  # type: ignore

__all__ = [
    "MT2DInverter",
    "MT2DInverterWeightedCost",
    "make_weighted_cost_fn",
    "MT2DFD_Torch",
    "MT2DTrueModels",
]
