# 版本信息
__version__ = "1.0.0"
__author__ = "lrr czz"
__email__ = "xinran.liu@zju.edu.cn"


# 1D模型和反演
from .model import MT1D
from .MTinv import MT1DInverter
from .visualize import MT1D_3DVisualizer