"""
1D geo-electric model: layer thicknesses and conductivities.
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import matplotlib.patches as patches
from typing import Tuple, Optional, Dict
import math

class MT1D:
    """
    Layered 1D conductivity model.

    Attributes
    ----------
    dz : Tensor
        Layer thicknesses (m).
    sig : Tensor
        Layer conductivities (S/m).
    n_layers : int
        Number of finite layers (excludes the half-space).
    """

    def __init__(self, dz, sig):
        """
        Parameters
        ----------
        dz : Tensor or list
            Layer thicknesses.
        sig : Tensor or list
            Layer conductivities (one more entry than ``dz``, including the
            terminating half-space).
        """
        self.dz = torch.tensor(dz, dtype=torch.float32)
        self.sig = torch.tensor(sig, dtype=torch.float32)

        # Validate lengths
        if len(self.dz) + 1 != len(self.sig):
            raise ValueError(
                "len(sig) must be len(dz)+1 (finite layers plus half-space)"
            )

    @property
    def n_layers(self):
        """Number of finite layers (excluding the half-space)."""
        return len(self.dz)

    def __repr__(self):
        return f"GeoElectricModel(dz={self.dz.tolist()}, sig={self.sig.tolist()})"

    def to_dict(self):
        """Serialize thicknesses and conductivities."""
        return {
            'dz': self.dz.tolist(),
            'sig': self.sig.tolist()
        }

    @classmethod
    def from_dict(cls, data):
        """Build a model from :meth:`to_dict` output."""
        return cls(data['dz'], data['sig'])

    def visualize(self, title="Geo-electric model"):
        """
        Plot conductivity vs depth.

        Parameters
        ----------
        title : str
            Figure title.
        """
        # Layer interfaces
        depths = [0]
        for i, thickness in enumerate(self.dz):
            depths.append(depths[i] + thickness.item())

        # Stair-step conductivity for plotting
        sig_plot = []
        for i in range(self.n_layers + 1):
            sig_plot.append(self.sig[i].item())
            if i < self.n_layers:
                sig_plot.append(self.sig[i].item())

        depth_plot = [0]
        for depth in depths[1:]:
            depth_plot.append(depth)
            depth_plot.append(depth)

        plt.figure(figsize=(10, 6))
        plt.plot(sig_plot, depth_plot, 'r-', linewidth=2)
        plt.yscale('linear')
        plt.gca().invert_yaxis()
        plt.xlabel('Conductivity (S/m)')
        plt.ylabel('Depth (m)')
        plt.title(title)
        plt.grid(True, which='both', linestyle='--', alpha=0.5)

        # Layer boundaries
        for depth in depths[1:-1]:
            plt.axhline(y=depth, color='k', linestyle='-', alpha=0.3)

        plt.tight_layout()
        return plt
