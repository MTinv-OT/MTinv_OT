"""
Constraint and regularization module.
Contains various constraint calculations for 1D/2D inversion; 2D theory (deGroot-Hedlin & Constable, 1990).
"""

import torch
from typing import Optional, Tuple


class ConstraintCalculator:
    """
    Constraint term calculator.
    Supports various constraint types for 1D and 2D models.
    """
    
    def __init__(self, device: str = "cpu"):
        """
        Initialize constraint calculator.
        
        Args:
            device: Compute device
        """
        self.device = device
    
    def build_roughness_matrix(self, n_layers: int) -> torch.Tensor:
        """
        Build 1D roughness matrix (first-order difference).
        
        Args:
            n_layers: Number of layers
            
        Returns:
            R: Roughness matrix [n_layers-1, n_layers]
        """
        R = torch.zeros((n_layers-1, n_layers), device=self.device)
        for i in range(n_layers-1):
            R[i, i] = -1
            R[i, i+1] = 1
        return R
    
    def build_curvature_matrix(self, n_layers: int) -> torch.Tensor:
        """
        Build 1D curvature matrix (second-order difference).
        
        Args:
            n_layers: Number of layers
            
        Returns:
            C: Curvature matrix [n_layers-2, n_layers]
        """
        C = torch.zeros((n_layers-2, n_layers), device=self.device)
        for i in range(n_layers-2):
            C[i, i] = 1
            C[i, i+1] = -2
            C[i, i+2] = 1
        return C
    
    def calculate_1d_model_norm(self, model: torch.Tensor, 
                                constraint_type: str = "roughness",
                                dz: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Compute 1D model constraint term.
        
        Args:
            model: Model parameters [n_layers]
            constraint_type: Constraint type ("roughness" or "curvature")
            dz: Layer thickness [n_layers] (optional, for depth weighting)
            
        Returns:
            norm: Constraint scalar value
        """
        n_layers = len(model)
        
        if constraint_type == "roughness":
            R = self.build_roughness_matrix(n_layers)
            model_diff = R @ model
        elif constraint_type == "curvature":
            C = self.build_curvature_matrix(n_layers)
            model_diff = C @ model
        else:
            # Default: deviation from mean
            model_diff = model - torch.mean(model)
        
        return torch.sum(model_diff ** 2)
    
    def calculate_2d_smoothness(self, model: torch.Tensor,
                               lateral_weight: float = 1.0,
                               vertical_weight: float = 1.0) -> torch.Tensor:
        """
        Compute 2D smoothness constraint (core regularization).
        
        Args:
            model: 2D model parameters [n_positions, n_layers+1]
            lateral_weight: Lateral smoothness weight
            vertical_weight: Vertical smoothness weight
            
        Returns:
            smooth_norm: Smoothness scalar value
        """
        n_pos, n_lay = model.shape
        
        # Lateral roughness (along profile direction)
        if n_pos > 1:
            lateral_diff = model[1:, :] - model[:-1, :]
            lateral_norm = torch.sum(lateral_diff ** 2)
        else:
            lateral_norm = torch.tensor(0.0, device=self.device)
        
        # Vertical roughness (along depth direction)
        if n_lay > 1:
            vertical_diff = model[:, 1:] - model[:, :-1]
            vertical_norm = torch.sum(vertical_diff ** 2)
        else:
            vertical_norm = torch.tensor(0.0, device=self.device)
        
        # Weighted combination
        total_norm = (
            lateral_weight * lateral_norm +
            vertical_weight * vertical_norm
        )
        
        return total_norm
    


