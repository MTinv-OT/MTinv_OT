# Version info
__version__ = "1.0.0"
__author__ = "lrr czz"
__email__ = "xinran.liu@zju.edu.cn"


# 1D model and inversion
from .model import MT1D
from .MTinv import MT1DInverter
from .visualize import MT1D_3DVisualizer