"""
Geo-electric model definition module.
Defines layer thickness and conductivity parameters.
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
    Geo-electric model class.
    
    Attributes:
        dz (torch.Tensor): Layer thickness (m)
        sig (torch.Tensor): Layer conductivity (S/m)
        n_layers (int): Number of layers
    """
    
    def __init__(self, dz, sig):
        """
        Initialize geo-electric model.
        
        Args:
            dz (torch.Tensor or list): Layer thickness
            sig (torch.Tensor or list): Layer conductivity
        """
        self.dz = torch.tensor(dz, dtype=torch.float32)
        self.sig = torch.tensor(sig, dtype=torch.float32)
        
        # Validate parameters
        if len(self.dz) + 1 != len(self.sig):
            raise ValueError("Number of conductivity values should exceed thickness by 1 (including top and bottom half-space)")
    
    @property
    def n_layers(self):
        """Return model layer count (excluding half-space)"""
        return len(self.dz)
    
    def __repr__(self):
        return f"GeoElectricModel(dz={self.dz.tolist()}, sig={self.sig.tolist()})"
    
    def to_dict(self):
        """Convert model parameters to dict"""
        return {
            'dz': self.dz.tolist(),
            'sig': self.sig.tolist()
        }
    
    @classmethod
    def from_dict(cls, data):
        """Create model from dict"""
        return cls(data['dz'], data['sig'])
    
    def visualize(self, title="Geo-electric Model"):
        """
        Visualize the model.
        
        Args:
            title (str): Plot title
        """
        # Compute layer depths
        depths = [0]
        for i, thickness in enumerate(self.dz):
            depths.append(depths[i] + thickness.item())
        
        # Expand conductivity for plotting
        sig_plot = []
        for i in range(self.n_layers + 1):
            sig_plot.append(self.sig[i].item())
            if i < self.n_layers:
                sig_plot.append(self.sig[i].item())
        
        depth_plot = [0]
        for depth in depths[1:]:
            depth_plot.append(depth)
            depth_plot.append(depth)
        
        # Create figure
        plt.figure(figsize=(10, 6))
        plt.plot(sig_plot, depth_plot, 'r-', linewidth=2)
        plt.yscale('linear')
        plt.gca().invert_yaxis()
        plt.xlabel('Conductivity (S/m)')
        plt.ylabel('Depth (m)')
        plt.title(title)
        plt.grid(True, which='both', linestyle='--', alpha=0.5)
        
        # Add layer boundary lines
        for depth in depths[1:-1]:
            plt.axhline(y=depth, color='k', linestyle='-', alpha=0.3)
        
        plt.tight_layout()
        return plt
