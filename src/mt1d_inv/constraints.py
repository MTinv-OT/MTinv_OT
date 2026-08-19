"""
Constraint and regularization helpers for 1D (and a simple 2D smoothness term).
2D roughness follows deGroot-Hedlin & Constable (1990).
"""

import torch
from typing import Optional, Tuple


class ConstraintCalculator:
    """Constraint calculator for 1D and 2D model penalties."""

    def __init__(self, device: str = "cpu"):
        """
        Parameters
        ----------
        device : str
            Compute device.
        """
        self.device = device

    def build_roughness_matrix(self, n_layers: int) -> torch.Tensor:
        """
        First-difference roughness matrix for a 1D layered model.

        Returns
        -------
        R : Tensor
            Shape ``[n_layers-1, n_layers]``.
        """
        R = torch.zeros((n_layers-1, n_layers), device=self.device)
        for i in range(n_layers-1):
            R[i, i] = -1
            R[i, i+1] = 1
        return R

    def build_curvature_matrix(self, n_layers: int) -> torch.Tensor:
        """
        Second-difference curvature matrix for a 1D layered model.

        Returns
        -------
        C : Tensor
            Shape ``[n_layers-2, n_layers]``.
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
        1D model constraint value.

        Parameters
        ----------
        model : Tensor
            Layer parameters, shape ``[n_layers]``.
        constraint_type : str
            ``"roughness"`` or ``"curvature"``.
        dz : Tensor, optional
            Layer thicknesses (unused; kept for API compatibility).
        """
        n_layers = len(model)

        if constraint_type == "roughness":
            R = self.build_roughness_matrix(n_layers)
            model_diff = R @ model
        elif constraint_type == "curvature":
            C = self.build_curvature_matrix(n_layers)
            model_diff = C @ model
        else:
            # Fallback: deviation from the mean
            model_diff = model - torch.mean(model)

        return torch.sum(model_diff ** 2)

    def calculate_2d_smoothness(self, model: torch.Tensor,
                               lateral_weight: float = 1.0,
                               vertical_weight: float = 1.0) -> torch.Tensor:
        """
        2D smoothness (core regularization).

        Parameters
        ----------
        model : Tensor
            Shape ``[n_positions, n_layers+1]``.
        lateral_weight, vertical_weight : float
            Weights for horizontal and vertical first differences.
        """
        n_pos, n_lay = model.shape

        # Lateral roughness (along stations)
        if n_pos > 1:
            lateral_diff = model[1:, :] - model[:-1, :]
            lateral_norm = torch.sum(lateral_diff ** 2)
        else:
            lateral_norm = torch.tensor(0.0, device=self.device)

        # Vertical roughness (along depth)
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
