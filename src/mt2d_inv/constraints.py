"""Constraint and regularization utilities for 1-D and 2-D inversion."""
from __future__ import annotations

from typing import Optional, Tuple

import torch


class ConstraintCalculator:
    """Calculate mesh-aware model constraints.

    The 2-D roughness term is discretized as a physical area integral of the
    squared (or smoothed absolute) model gradient.  This makes the value stable
    when the same physical model is represented on a refined/coarsened mesh.
    The reference-model term is an area-weighted mean, so it is also stable with
    respect to cell count and padding of the computational mesh.

    Naming note: ``nx``/``dx`` here refer to the horizontal profile direction
    (matching the ``alpha_x`` roughness weight in ``MT2DInverter.run_inversion``),
    which the calling ``MT2DInverter`` grid setup instead calls ``ny``/``dy``
    (``yn`` are the horizontal grid-edge coordinates). ``nz``/``dz`` (vertical /
    depth direction) are named consistently across both classes.
    """

    def __init__(self, nx: int, nz: int, dx, dz, device: str = "cpu"):
        self.nx = int(nx)
        self.nz = int(nz)
        self.dx = (
            torch.as_tensor(dx, device=device)
            if not isinstance(dx, torch.Tensor)
            else dx.to(device)
        )
        self.dz = (
            torch.as_tensor(dz, device=device)
            if not isinstance(dz, torch.Tensor)
            else dz.to(device)
        )
        self.device = device

    def _spacing_vectors(
        self,
        *,
        dtype: torch.dtype,
        device: torch.device,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return positive 1-D cell-width vectors of lengths ``nx`` and ``nz``."""
        dx = self.dx.to(device=device, dtype=dtype).reshape(-1)
        dz = self.dz.to(device=device, dtype=dtype).reshape(-1)

        if dx.numel() == 1:
            dx = dx.expand(self.nx)
        if dz.numel() == 1:
            dz = dz.expand(self.nz)

        if dx.numel() != self.nx:
            raise ValueError(
                f"dx size mismatch: expected {self.nx}, got {dx.numel()}"
            )
        if dz.numel() != self.nz:
            raise ValueError(
                f"dz size mismatch: expected {self.nz}, got {dz.numel()}"
            )
        if (dx <= 0).any() or (dz <= 0).any():
            raise ValueError("All dx/dz cell widths must be strictly positive")
        return dx, dz

    @staticmethod
    def _validate_model_shape(model: torch.Tensor, nz: int, nx: int) -> None:
        if tuple(model.shape) != (nz, nx):
            raise ValueError(
                f"Model shape mismatch: expected {(nz, nx)}, got {tuple(model.shape)}"
            )

    def compute_depth_weights_from_zn(
        self,
        zn: torch.Tensor,
        nza: int,
        beta: float = 0.3,
        clamp_max: float = 500.0,
        normalize: bool = True,
    ) -> torch.Tensor:
        """Compute earth-only depth weights ``w(z) = (z/z0)^beta``."""
        zn_tensor = (
            zn.to(self.device, dtype=torch.float64)
            if isinstance(zn, torch.Tensor)
            else torch.as_tensor(zn, device=self.device, dtype=torch.float64)
        )
        if zn_tensor.ndim != 1 or zn_tensor.numel() < 2:
            raise ValueError("zn must be a 1D tensor/array with length >= 2")

        nza_i = int(max(0, nza))
        z_centers_full = (zn_tensor[:-1] + zn_tensor[1:]) * 0.5
        z_centers_earth = z_centers_full[nza_i:]

        if int(z_centers_earth.numel()) != self.nz:
            raise ValueError(
                f"Depth weights size mismatch: expected nz_earth={self.nz}, "
                f"got {int(z_centers_earth.numel())}. Check zn and nza consistency."
            )

        z_pos = torch.clamp(z_centers_earth, min=1.0)
        z0 = z_pos.min().clamp(min=1.0)
        w_z = (z_pos / z0).pow(float(beta)).unsqueeze(1).expand(-1, self.nx)
        w_z = torch.clamp(w_z, max=float(clamp_max))
        if normalize:
            w_z = w_z / w_z.max().clamp(min=1e-12)
        return w_z.to(device=self.device, dtype=torch.float64)

    def calculate_weighted_roughness(
        self,
        model_log_sigma: torch.Tensor,
        weights: Optional[torch.Tensor] = None,
        norm_type: str = "L2",
        alpha_x: float = 1.0,
        alpha_z: float = 1.0,
        tv_epsilon: float = 1e-8,
    ) -> torch.Tensor:
        """Return a mesh-stable 2-D roughness constraint.

        For L2 the discretization approximates

            integral [ (dm/dx)^2 + (dm/dz)^2 ] dA.

        In two dimensions this physical integral is invariant, to discretization
        error, when a fixed physical model/domain is remeshed.  For L1 a smooth
        TV density ``sqrt(gradient**2 + tv_epsilon**2)`` is used.

        Spatial weights are interpolated to interfaces and multiply the
        gradient before the norm, preserving the previous weighting convention.
        """
        self._validate_model_shape(model_log_sigma, self.nz, self.nx)
        dx, dz = self._spacing_vectors(
            dtype=model_log_sigma.dtype,
            device=model_log_sigma.device,
        )

        if weights is not None:
            weights = weights.to(
                device=model_log_sigma.device,
                dtype=model_log_sigma.dtype,
            )
            self._validate_model_shape(weights, self.nz, self.nx)

        zero = model_log_sigma.new_zeros(())
        loss_x = zero
        loss_z = zero

        if self.nx > 1:
            diff_x = model_log_sigma[:, 1:] - model_log_sigma[:, :-1]
            spacing_x = 0.5 * (dx[:-1] + dx[1:])
            grad_x = diff_x / spacing_x.view(1, -1)
            if weights is not None:
                grad_x = grad_x * 0.5 * (weights[:, 1:] + weights[:, :-1])
            area_x = dz.view(-1, 1) * spacing_x.view(1, -1)
            if norm_type.upper() == "L2":
                density_x = grad_x.square()
            elif norm_type.upper() == "L1":
                eps = model_log_sigma.new_tensor(float(tv_epsilon))
                density_x = torch.sqrt(grad_x.square() + eps.square())
            else:
                raise ValueError("Unsupported norm_type. Choose 'L1' or 'L2'.")
            # Interface dual cells omit half a boundary cell on each side.
            # Rescale their covered area to the full model area so a constant
            # physical gradient has the same integral on coarse and fine grids.
            domain_area = torch.sum(dx) * torch.sum(dz)
            covered_area_x = torch.sum(area_x).clamp_min(
                torch.finfo(model_log_sigma.dtype).eps
            )
            loss_x = (
                torch.sum(density_x * area_x)
                * domain_area
                / covered_area_x
            )

        if self.nz > 1:
            diff_z = model_log_sigma[1:, :] - model_log_sigma[:-1, :]
            spacing_z = 0.5 * (dz[:-1] + dz[1:])
            grad_z = diff_z / spacing_z.view(-1, 1)
            if weights is not None:
                grad_z = grad_z * 0.5 * (weights[1:, :] + weights[:-1, :])
            area_z = spacing_z.view(-1, 1) * dx.view(1, -1)
            if norm_type.upper() == "L2":
                density_z = grad_z.square()
            elif norm_type.upper() == "L1":
                eps = model_log_sigma.new_tensor(float(tv_epsilon))
                density_z = torch.sqrt(grad_z.square() + eps.square())
            else:
                raise ValueError("Unsupported norm_type. Choose 'L1' or 'L2'.")
            domain_area = torch.sum(dx) * torch.sum(dz)
            covered_area_z = torch.sum(area_z).clamp_min(
                torch.finfo(model_log_sigma.dtype).eps
            )
            loss_z = (
                torch.sum(density_z * area_z)
                * domain_area
                / covered_area_z
            )

        return float(alpha_x) * loss_x + float(alpha_z) * loss_z

    def calculate_reference_model_constraint(
        self,
        model_log_sigma: torch.Tensor,
        reference_model_log_sigma: torch.Tensor,
        weights: Optional[torch.Tensor] = None,
        norm_type: str = "L2",
        tv_epsilon: float = 1e-8,
    ) -> torch.Tensor:
        """Return an area-weighted mean distance to a reference model.

        The denominator is the total active-cell area rather than the number of
        cells, so non-uniform and refined meshes represent the same continuous
        constraint on the same scale.
        """
        self._validate_model_shape(model_log_sigma, self.nz, self.nx)
        self._validate_model_shape(reference_model_log_sigma, self.nz, self.nx)

        dx, dz = self._spacing_vectors(
            dtype=model_log_sigma.dtype,
            device=model_log_sigma.device,
        )
        diff = model_log_sigma - reference_model_log_sigma.to(
            device=model_log_sigma.device,
            dtype=model_log_sigma.dtype,
        )

        if weights is not None:
            weights = weights.to(
                device=model_log_sigma.device,
                dtype=model_log_sigma.dtype,
            )
            self._validate_model_shape(weights, self.nz, self.nx)
            diff = diff * weights

        if norm_type.upper() == "L2":
            density = diff.square()
        elif norm_type.upper() == "L1":
            eps = model_log_sigma.new_tensor(float(tv_epsilon))
            density = torch.sqrt(diff.square() + eps.square())
        else:
            raise ValueError("Unsupported norm_type. Choose 'L1' or 'L2'.")

        cell_area = dz.view(-1, 1) * dx.view(1, -1)
        total_area = torch.sum(cell_area).clamp_min(
            torch.finfo(model_log_sigma.dtype).eps
        )
        return torch.sum(density * cell_area) / total_area

    def calculate_combined_constraint(
        self,
        model_log_sigma: torch.Tensor,
        reference_model_log_sigma: Optional[torch.Tensor] = None,
        roughness_weights: Optional[torch.Tensor] = None,
        reference_weights: Optional[torch.Tensor] = None,
        roughness_norm: str = "L2",
        reference_norm: str = "L2",
        reference_weight: float = 0.0,
        alpha_x: float = 1.0,
        alpha_z: float = 1.0,
        tv_epsilon: float = 1e-8,
    ) -> torch.Tensor:
        """Return roughness plus an optional reference-model constraint."""
        roughness_loss = self.calculate_weighted_roughness(
            model_log_sigma,
            roughness_weights,
            roughness_norm,
            alpha_x=alpha_x,
            alpha_z=alpha_z,
            tv_epsilon=tv_epsilon,
        )

        if reference_model_log_sigma is not None and reference_weight > 0.0:
            reference_loss = self.calculate_reference_model_constraint(
                model_log_sigma,
                reference_model_log_sigma,
                reference_weights,
                reference_norm,
                tv_epsilon=tv_epsilon,
            )
            return roughness_loss + float(reference_weight) * reference_loss
        return roughness_loss
