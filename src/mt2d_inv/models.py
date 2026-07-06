"""Standard test model generators."""
from __future__ import annotations

import torch
import numpy as np


class MT2DTrueModels:
    """2D MT true-model generators.

    This class provides several standard benchmark/test models that can be used
    by the main workflow when building the true conductivity model.

    COMMEMI (Comparison of Modelling Methods for EM Induction) benchmarks
    --------------------------------------------------------------------
    * ``create_commemi_2d1`` — isolated conductor in a resistive host (classic 2D-1).
    * ``create_commemi_2d4`` — variable-thickness conductive overburden + basement
      contact (static-shift style 2D-4-style section; tune parameters to match a
      given paper/plot).
    * ``create_commemi_2d0`` — legacy grid used by ``test_CM2D-0`` (different return
      signature: ``zn, yn, freq, ry, sig``).

    Primary literature: M.S. Zhdanov et al., *Journal of Applied Geophysics* 40
    (1997), DOI: https://doi.org/10.1016/S0926-9851(97)00013-X
    """

    @staticmethod
    def _default_grid(nza=10):
        """Generate the default grid.

        Returns
        -------
        (yn, zn, nza)
            yn, zn are 1D arrays of node coordinates (y and z). `nza` is the
            number of air cells used to build the grid.
        """
        n_shallow = 10
        z_shallow = np.logspace(np.log10(100.0), np.log10(5000.0), n_shallow)

        n_deep = 20
        z_deep = np.logspace(np.log10(6000.0), np.log10(100000.0), n_deep)

        z_sub = np.concatenate([z_shallow, z_deep])
        # Air–earth interface: last air layer thickness = first subsurface layer dz0
        dz0 = float(z_sub[0])
        z_air = -np.logspace(np.log10(dz0), np.log10(50000.0), nza)
        z_air = np.flip(z_air)
        z_air = np.append(z_air, 0.0)

        zn = np.concatenate([z_air[:-1], np.array([0.0]), z_sub])

        y_center = np.linspace(-10000.0, 10000.0, 21)
        y_left = -np.logspace(np.log10(11000.0), np.log10(50000.0), 10)
        y_right = np.logspace(np.log10(11000.0), np.log10(50000.0), 10)
        y_left = np.flip(y_left)
        yn = np.concatenate([y_left, y_center, y_right])

        return yn, zn, nza

    @staticmethod
    def _infer_nza(zn_np):
        """Infer the number of air cells from zn (by counting cell centers with z < 0)."""
        zn_np = np.asarray(zn_np)
        z_centers = 0.5 * (zn_np[:-1] + zn_np[1:])
        return int(np.sum(z_centers < 0))

    @staticmethod
    def create_single_anomaly(zn=None, yn=None, nza=10, device="cpu", depth="shallow"):
        """
        Single-anomaly model.

        If zn/yn are not provided, this function uses the default grid (controlled
        by `nza`).

        Conductivity:
        - Background: 0.01 S/m
        - Air: 1e-9 S/m
        - Anomaly: 1.0 S/m where |Y| < 3000 m

        Parameters
        ----------
        depth : str
            "shallow": anomaly at 5–10 km depth (5000 < Z < 10000 m)
            "deep": anomaly at 25–30 km depth (25000 < Z < 30000 m, shifted down 20 km)

        Returns (yn, zn, nza, sig_true)
        """

        if zn is None and yn is None:
            yn, zn, nza = MT2DTrueModels._default_grid(nza=nza)
        elif zn is None or yn is None:
            raise ValueError("create_single_anomaly: zn and yn must be provided together, or both be None to use the default grid")
        else:
            zn_np_tmp = zn.cpu().numpy() if isinstance(zn, torch.Tensor) else np.asarray(zn)
            nza = MT2DTrueModels._infer_nza(zn_np_tmp)

        zn_np = zn.cpu().numpy() if isinstance(zn, torch.Tensor) else np.asarray(zn)
        yn_np = yn.cpu().numpy() if isinstance(yn, torch.Tensor) else np.asarray(yn)
        yc = (yn_np[:-1] + yn_np[1:]) / 2.0
        zc = (zn_np[:-1] + zn_np[1:]) / 2.0
        Y, Z = np.meshgrid(yc, zc)

        sig_true = np.ones_like(Y) * 0.01
        sig_true[Z < 0] = 1e-9

        if depth == "shallow":
            z_lo, z_hi = 5000.0, 10000.0
        elif depth == "deep":
            z_lo, z_hi = 25000.0, 30000.0
        else:
            raise ValueError(f"depth must be 'shallow' or 'deep', got {depth!r}")

        mask_anomaly = (np.abs(Y) < 3000) & (Z > z_lo) & (Z < z_hi)
        sig_true[mask_anomaly] = 1.0

        sig_true = torch.tensor(sig_true, dtype=torch.float64, device=device)
        return yn, zn, nza, sig_true

    @staticmethod
    def create_dual_block(zn=None, yn=None, nza=10, device="cpu", bg_rho=100.0, anomaly_rho=1.0):
        """
        Dual-block model (Two-Brick / Dual-Prism).

        Useful for testing lateral resolution and vertical discrimination.

        Returns (yn, zn, nza, sigma)
        """

        if zn is None and yn is None:
            yn, zn, nza = MT2DTrueModels._default_grid(nza=nza)
        elif zn is None or yn is None:
            raise ValueError("create_dual_block: zn and yn must be provided together, or both be None to use the default grid")
        else:
            zn_np_tmp = zn.cpu().numpy() if isinstance(zn, torch.Tensor) else np.asarray(zn)
            nza = MT2DTrueModels._infer_nza(zn_np_tmp)

        zn_np = zn.cpu().numpy() if isinstance(zn, torch.Tensor) else np.asarray(zn)
        yn_np = yn.cpu().numpy() if isinstance(yn, torch.Tensor) else np.asarray(yn)
        nz, ny = len(zn_np)-1, len(yn_np)-1

        sigma = np.ones((nz, ny)) * (1.0 / bg_rho)

        z_centers = 0.5 * (zn_np[:-1] + zn_np[1:])
        y_centers = 0.5 * (yn_np[:-1] + yn_np[1:])

        block1_y = [-5000, -2000]
        block1_z = [1000, 4000]
        block2_y = [2000, 5000]
        block2_z = [1000, 4000]

        def get_indices(centers, limits):
            return np.where((centers >= limits[0]) & (centers <= limits[1]))[0]

        z_idx1 = get_indices(z_centers, block1_z)
        y_idx1 = get_indices(y_centers, block1_y)
        z_idx2 = get_indices(z_centers, block2_z)
        y_idx2 = get_indices(y_centers, block2_y)

        if len(z_idx1) > 0 and len(y_idx1) > 0:
            sigma[np.ix_(z_idx1, y_idx1)] = 1.0 / anomaly_rho
        if len(z_idx2) > 0 and len(y_idx2) > 0:
            sigma[np.ix_(z_idx2, y_idx2)] = 1.0 / anomaly_rho

        sigma[z_centers < 0, :] = 1e-9

        sigma = torch.tensor(sigma, dtype=torch.float64, device=device)
        return yn, zn, nza, sigma

    @staticmethod
    def create_checkerboard(zn=None, yn=None, nza=10, device="cpu", bg_rho=100.0, anomaly_rho_low=10.0, anomaly_rho_high=1000.0, block_w=4, block_h=3):
        """
        Cross-shaped resistivity pattern (Ohm·m) for testing 3D vs 6D OT boundary detection.

        Pattern within depth 0~20 km, lateral -10~10 km:

        - Vertical bar:  y in [-1, 1] km, z in [0, 20] km  (center column)
        - Horizontal bar: y in [-5, 5] km (10 km length), z in [7.5, 12.5] km (5 km width)

        Cross uses anomaly_rho_low; background uses bg_rho.

        Notes
        -----
        `anomaly_rho_high`, `block_w`, `block_h` are kept for signature compatibility.

        Returns (yn, zn, nza, sigma)
        """

        if zn is None and yn is None:
            yn, zn, nza = MT2DTrueModels._default_grid(nza=nza)
        elif zn is None or yn is None:
            raise ValueError("create_checkerboard: zn and yn must be provided together, or both be None to use the default grid")
        else:
            zn_np_tmp = zn.cpu().numpy() if isinstance(zn, torch.Tensor) else np.asarray(zn)
            nza = MT2DTrueModels._infer_nza(zn_np_tmp)

        zn_np = zn.cpu().numpy() if isinstance(zn, torch.Tensor) else np.asarray(zn)
        yn_np = yn.cpu().numpy() if isinstance(yn, torch.Tensor) else np.asarray(yn)

        nz, ny = len(zn_np) - 1, len(yn_np) - 1

        sigma = np.ones((nz, ny), dtype=float) * (1.0 / bg_rho)

        z_centers_km = 0.5 * (zn_np[:-1] + zn_np[1:]) / 1000.0
        y_centers_km = 0.5 * (yn_np[:-1] + yn_np[1:]) / 1000.0
        Ykm, Zkm = np.meshgrid(y_centers_km, z_centers_km)

        # Air
        sigma[Zkm < 0.0] = 1e-9

        # Cross: vertical bar (y in [-1, 1] km, z in [0, 20] km) + horizontal bar (y in [-5, 5] km, z in [7.5, 12.5] km, 10km×5km)
        rho_cross = float(anomaly_rho_low)
        vert_mask = (Ykm >= -2.0) & (Ykm <= 2.0) & (Zkm >= 0.0) & (Zkm <= 20.0)
        horz_mask = (Ykm >= -5.0) & (Ykm <= 5.0) & (Zkm >= 8.0) & (Zkm <= 12.0)
        cross_mask = vert_mask | horz_mask
        sigma[cross_mask] = 1.0 / rho_cross

        sigma = torch.tensor(sigma, dtype=torch.float64, device=device)
        return yn, zn, nza, sigma

    @staticmethod
    def create_single_block(
        zn=None,
        yn=None,
        nza=10,
        device="cpu",
        y_range_km=(5.0, 10.0),
        z_range_km=(5.0, 10.0),
        block_rho=1000.0,
        bg_sigma=0.01,
    ):
        """Single resistive block model.

        This model is useful for testing recovery from a laterally shifted initial model.

        Parameters
        ----------
        y_range_km, z_range_km
            Block extent in km.
        block_rho
            Block resistivity (Ohm·m). Conductivity is set to 1 / block_rho.
        bg_sigma
            Background conductivity (S/m).

        Returns (yn, zn, nza, sigma)
        """

        if zn is None and yn is None:
            yn, zn, nza = MT2DTrueModels._default_grid(nza=nza)
        elif zn is None or yn is None:
            raise ValueError("create_single_block: zn and yn must be provided together, or both be None to use the default grid")
        else:
            zn_np_tmp = zn.cpu().numpy() if isinstance(zn, torch.Tensor) else np.asarray(zn)
            nza = MT2DTrueModels._infer_nza(zn_np_tmp)

        zn_np = zn.cpu().numpy() if isinstance(zn, torch.Tensor) else np.asarray(zn)
        yn_np = yn.cpu().numpy() if isinstance(yn, torch.Tensor) else np.asarray(yn)

        z_centers = 0.5 * (zn_np[:-1] + zn_np[1:])
        y_centers = 0.5 * (yn_np[:-1] + yn_np[1:])
        Y, Z = np.meshgrid(y_centers, z_centers)

        sigma = np.ones_like(Y) * bg_sigma
        sigma[z_centers < 0, :] = 1e-9

        y_min, y_max = y_range_km[0] * 1e3, y_range_km[1] * 1e3
        z_min, z_max = z_range_km[0] * 1e3, z_range_km[1] * 1e3
        mask = (Y >= y_min) & (Y < y_max) & (Z >= z_min) & (Z < z_max)
        sigma[mask] = 1.0 / block_rho

        sigma = torch.tensor(sigma, dtype=torch.float64, device=device)
        return yn, zn, nza, sigma

    @staticmethod
    def create_three_block(
        zn=None,
        yn=None,
        nza=10,
        device="cpu",
        bg_sigma=0.01,
        high_rho=1000.0,
        low_rho=1.0,
        z_range_km: tuple[float, float] = (5.0, 10.0),
    ):
        """Three-block model (default anomaly depth 5–10 km).

        Left high-resistivity, center low-resistivity, right high-resistivity (symmetric).

        - Left:  -10 to  -1 km (high_rho)
        - Center: -1 to   1 km (low_rho)
        - Right:  1 to  10 km (high_rho)

        Depth extent of all three blocks (subsurface, positive down):

        - ``z_range_km``: ``(z_min_km, z_max_km)`` — anomaly top/bottom in km.
          Default ``(5.0, 10.0)`` matches the historical hard-coded 5–10 km.

        Background conductivity: bg_sigma (S/m).

        Returns (yn, zn, nza, sigma)
        """

        if zn is None and yn is None:
            yn, zn, nza = MT2DTrueModels._default_grid(nza=nza)
        elif zn is None or yn is None:
            raise ValueError("create_three_block: zn and yn must be provided together, or both be None to use the default grid")
        else:
            zn_np_tmp = zn.cpu().numpy() if isinstance(zn, torch.Tensor) else np.asarray(zn)
            nza = MT2DTrueModels._infer_nza(zn_np_tmp)

        zn_np = zn.cpu().numpy() if isinstance(zn, torch.Tensor) else np.asarray(zn)
        yn_np = yn.cpu().numpy() if isinstance(yn, torch.Tensor) else np.asarray(yn)

        z_centers = 0.5 * (zn_np[:-1] + zn_np[1:])
        y_centers = 0.5 * (yn_np[:-1] + yn_np[1:])
        Y, Z = np.meshgrid(y_centers, z_centers)

        sigma = np.ones_like(Y) * bg_sigma
        sigma[z_centers < 0, :] = 1e-9

        z0, z1 = float(z_range_km[0]), float(z_range_km[1])
        if z0 >= z1:
            raise ValueError(f"z_range_km must be (z_min_km, z_max_km) with z_min < z_max; got {z_range_km!r}")
        z_min, z_max = z0 * 1e3, z1 * 1e3
        mask_l = (Y >= -10e3) & (Y < -1e3) & (Z >= z_min) & (Z < z_max)
        sigma[mask_l] = 1.0 / high_rho
        mask_c = (Y >= -1e3) & (Y < 1e3) & (Z >= z_min) & (Z < z_max)
        sigma[mask_c] = 1.0 / low_rho
        mask_r = (Y >= 1e3) & (Y < 10e3) & (Z >= z_min) & (Z < z_max)
        sigma[mask_r] = 1.0 / high_rho

        sigma = torch.tensor(sigma, dtype=torch.float64, device=device)
        return yn, zn, nza, sigma

    @staticmethod
    def create_geological_models(
        zn=None,
        yn=None,
        nza=10,
        model_type="salt_dome",
        device="cpu",
        add_shielding=False,
    ):
        """Classic geological models for OT vs MSE tests.

        Parameters
        ----------
        model_type
            "salt_dome" (low surrounds high; salt dome) or "magma_chamber" (high surrounds low).
        add_shielding
            Only effective for "magma_chamber": adds a shallow (0–2 km) conductive
            shielding layer with 10 Ohm·m.

        Returns (yn, zn, nza, sigma)
        """

        if zn is None and yn is None:
            yn, zn, nza = MT2DTrueModels._default_grid(nza=nza)
        elif zn is None or yn is None:
            raise ValueError("create_geological_models: zn and yn must be provided together, or both be None to use the default grid")
        else:
            zn_np_tmp = zn.cpu().numpy() if isinstance(zn, torch.Tensor) else np.asarray(zn)
            nza = MT2DTrueModels._infer_nza(zn_np_tmp)

        zn_np = zn.cpu().numpy() if isinstance(zn, torch.Tensor) else np.asarray(zn)
        yn_np = yn.cpu().numpy() if isinstance(yn, torch.Tensor) else np.asarray(yn)
        nz, ny = len(zn_np) - 1, len(yn_np) - 1

        z_centers_km = 0.5 * (zn_np[:-1] + zn_np[1:]) / 1000.0
        y_centers_km = 0.5 * (yn_np[:-1] + yn_np[1:]) / 1000.0
        Y, Z = np.meshgrid(y_centers_km, z_centers_km)

        sigma = np.zeros((nz, ny))

        if model_type == "salt_dome":
            bg_rho = 5.0
            sigma[:] = 1.0 / bg_rho
            stem_mask = (np.abs(Y) < 2.0) & (Z > 2.0) & (Z < 8.0)
            cap_mask = (np.abs(Y) < 4.0) & (Z > 2.0) & (Z < 4.0)
            sigma[stem_mask | cap_mask] = 1.0 / 1000.0
        elif model_type == "magma_chamber":
            bg_rho = 1000.0
            sigma[:] = 1.0 / bg_rho
            radius = 5.0
            dist_sq = (Y - 5.0) ** 2 + (Z - 10.0) ** 2
            sigma[dist_sq < radius**2] = 1.0 / 1.0
            if add_shielding:
                sigma[(Z >= 0.0) & (Z < 2.0)] = 1.0 / 10.0
        else:
            raise ValueError(f"model_type must be 'salt_dome' or 'magma_chamber', got '{model_type}'")

        sigma[Z < 0.0] = 1e-9
        sigma = torch.tensor(sigma, dtype=torch.float64, device=device)
        return yn, zn, nza, sigma

    @staticmethod
    def create_commemi_2d0(nza, device="cpu"):
        """
        COMMEMI 2D-0 benchmark model.

        Returns (zn, yn, freq, ry, sig), matching the original generate_model in
        test_CM2D-0.
        """
        y = 10e3
        z = 50e3
        nz = 50
        nz_b = 10
        ny = 50
        multiple_t = 3.0
        multiple_b = 3.0
        multiple_l = 10.0
        multiple_r = 10.0

        z_air = (-2**(np.linspace(1, np.log2(multiple_t*z), nza+1)))[::-1]
        zn0 = np.concatenate(([0], 2**(np.linspace(1, np.log2(z), nz))), 0)
        z_b = 2**(np.linspace(np.log2(zn0[-1]), np.log2(multiple_b*zn0[-1]), nz_b+1))
        zn = np.concatenate((z_air[:-1], zn0, z_b[1:]))

        y0 = 2**(np.linspace(0, np.log2(y), int(ny/2))) - y-1
        y1 = -y0[::-1]
        yn0 = np.concatenate((y0, [0], y1), 0)
        y_l = -2**(np.linspace(np.log2(multiple_l*yn0[-1]), np.log2(yn0[-1]), 2*ny+1))
        y_r = 2**(np.linspace(np.log2(yn0[-1]), np.log2(multiple_r*yn0[-1]), 2*ny+1))
        yn = np.concatenate((y_l[:-1], yn0, y_r[1:]))

        freq = np.array([1.0/300])
        ry = np.linspace(-30e3, 30e3, 100+1)

        sig = np.ones((len(zn)-1, len(yn)-1)) * 1e-2
        sig[:nza, :] = 1e-9
        sig[nza:nza+nz, 0:2*ny] = 1.0/10
        sig[nza:nza+nz, 2*ny:3*ny] = 1.0
        sig[nza:nza+nz, 3*ny:] = 1.0/2

        return zn, yn, freq, ry, sig

    @staticmethod
    def create_commemi_2d1(
        zn=None,
        yn=None,
        nza=10,
        device="cpu",
        rho_host: float = 100.0,
        rho_block: float = 10.0,
        y_half_width_m: float = 2000.0,
        z_top_m: float = 25000.0,
        z_bottom_m: float = 45000.0,
    ):
        """
        COMMEMI **2D-1**-type model: 10 Ω·m rectangular inclusion in a 100 Ω·m host.

        Default geometry matches the form most often reproduced in MT modelling
        papers derived from the COMMEMI intercomparison (horizontal extent
        ±2 km; depth 25–45 km, *z* positive downward from the surface).

        Parameters
        ----------
        rho_host, rho_block
            Resistivities (Ω·m) of background and anomaly.
        y_half_width_m
            Inclusion occupies ``|y| <= y_half_width_m``.
        z_top_m, z_bottom_m
            Inclusion vertical extent in metres (depth below surface).

        Returns
        -------
        (yn, zn, nza, sig_true)
            ``sig_true`` conductivity (S/m) on cell centers, shape ``(nz, ny)``.

        References
        ----------
        COMMEMI project overview and model suite: Zhdanov et al., 1997,
        https://doi.org/10.1016/S0926-9851(97)00013-X
        """
        if zn is None and yn is None:
            yn, zn, nza = MT2DTrueModels._default_grid(nza=nza)
        elif zn is None or yn is None:
            raise ValueError("create_commemi_2d1: provide both zn and yn, or neither.")
        else:
            zn_np_tmp = zn.cpu().numpy() if isinstance(zn, torch.Tensor) else np.asarray(zn)
            nza = MT2DTrueModels._infer_nza(zn_np_tmp)

        zn_np = zn.cpu().numpy() if isinstance(zn, torch.Tensor) else np.asarray(zn)
        yn_np = yn.cpu().numpy() if isinstance(yn, torch.Tensor) else np.asarray(yn)
        z_centers = 0.5 * (zn_np[:-1] + zn_np[1:])
        y_centers = 0.5 * (yn_np[:-1] + yn_np[1:])
        Y, Z = np.meshgrid(y_centers, z_centers)

        sig = np.ones_like(Y, dtype=float) / float(rho_host)
        sig[Z < 0] = 1e-9

        y0 = float(y_half_width_m)
        zt, zb = float(z_top_m), float(z_bottom_m)
        if zt >= zb:
            raise ValueError(f"z_top_m must be < z_bottom_m; got {zt}, {zb}")
        mask = (np.abs(Y) <= y0) & (Z >= zt) & (Z <= zb)
        sig[mask] = 1.0 / float(rho_block)

        sig_true = torch.tensor(sig, dtype=torch.float64, device=device)
        return yn, zn, nza, sig_true

    @staticmethod
    def create_commemi_2d4(
        zn=None,
        yn=None,
        nza=10,
        device="cpu",
        rho_overburden: float = 20.0,
        rho_basement_left: float = 3000.0,
        rho_basement_right: float = 300.0,
        fault_y_m: float = 0.0,
        wedge_y_min_m: float = -20000.0,
        wedge_y_max_m: float = 20000.0,
        overburden_thickness_left_m: float = 200.0,
        overburden_thickness_right_m: float = 1200.0,
    ):
        """
        COMMEMI **2D-4**-style model: lateral change in surficial conductive cover
        (static-shift generator) over a resistive basement with a vertical
        resistivity jump (*fault / contact*).

        Published diagrams for model 2D-4 differ in exact thicknesses and
        resistivities; this implementation uses a **linear wedge** for interface
        depth between ``wedge_y_min_m`` and ``wedge_y_max_m``, and a vertical
        basement contact at ``fault_y_m``. Adjust keyword arguments to match a
        specific reproduction (e.g. from the COMMEMI special issue or a later
        compilation).

        For each subsurface cell center (y, z) with z >= 0:
        - If ``z < h(y)``: conductivity ``1/rho_overburden``
        - Else: ``1/rho_basement_left`` if ``y < fault_y_m`` else ``1/rho_basement_right``

        Returns
        -------
        (yn, zn, nza, sig_true)

        References
        ----------
        COMMEMI project: Zhdanov et al., 1997,
        https://doi.org/10.1016/S0926-9851(97)00013-X
        """
        if zn is None and yn is None:
            yn, zn, nza = MT2DTrueModels._default_grid(nza=nza)
        elif zn is None or yn is None:
            raise ValueError("create_commemi_2d4: provide both zn and yn, or neither.")
        else:
            zn_np_tmp = zn.cpu().numpy() if isinstance(zn, torch.Tensor) else np.asarray(zn)
            nza = MT2DTrueModels._infer_nza(zn_np_tmp)

        zn_np = zn.cpu().numpy() if isinstance(zn, torch.Tensor) else np.asarray(zn)
        yn_np = yn.cpu().numpy() if isinstance(yn, torch.Tensor) else np.asarray(yn)
        z_centers = 0.5 * (zn_np[:-1] + zn_np[1:])
        y_centers = 0.5 * (yn_np[:-1] + yn_np[1:])
        Y, Z = np.meshgrid(y_centers, z_centers)

        ya, yb = float(wedge_y_min_m), float(wedge_y_max_m)
        h0, h1 = float(overburden_thickness_left_m), float(overburden_thickness_right_m)
        if ya >= yb:
            raise ValueError("wedge_y_min_m must be < wedge_y_max_m")
        fy = float(fault_y_m)

        def thickness_m(y: np.ndarray) -> np.ndarray:
            y_clip = np.clip(y, ya, yb)
            t = h0 + (h1 - h0) * (y_clip - ya) / (yb - ya)
            return t

        sig = np.zeros_like(Y, dtype=float)
        sig[Z < 0] = 1e-9

        T = thickness_m(Y)
        subs = Z >= 0
        in_ov = subs & (Z < T)
        in_bs = subs & ~in_ov
        sig[in_ov] = 1.0 / float(rho_overburden)
        left = in_bs & (Y < fy)
        right = in_bs & (Y >= fy)
        sig[left] = 1.0 / float(rho_basement_left)
        sig[right] = 1.0 / float(rho_basement_right)

        sig_true = torch.tensor(sig, dtype=torch.float64, device=device)
        return yn, zn, nza, sig_true

    @staticmethod
    def create_commemi(
        model_id: str,
        zn=None,
        yn=None,
        nza=10,
        device="cpu",
        **kwargs,
    ):
        """
        Dispatch COMMEMI-style builders by name.

        Parameters
        ----------
        model_id
            ``"2d-1"``, ``"2D-1"``, ``"2d-4"``, or ``"2D-4"``.
        **kwargs
            Forwarded to ``create_commemi_2d1`` or ``create_commemi_2d4``.

        Returns
        -------
        (yn, zn, nza, sig_true)
        """
        key = str(model_id).strip().lower().replace("_", "-")
        if key == "2d-1":
            return MT2DTrueModels.create_commemi_2d1(
                zn=zn, yn=yn, nza=nza, device=device, **kwargs
            )
        if key == "2d-4":
            return MT2DTrueModels.create_commemi_2d4(
                zn=zn, yn=yn, nza=nza, device=device, **kwargs
            )
        raise ValueError(
            f"Unknown COMMEMI model_id={model_id!r}; use '2d-1' or '2d-4'."
        )

    @staticmethod
    def create_true_checkerboard(
        yn,
        zn,
        nza=10,
        device="cpu",
        bg_rho=100.0,
        rho_1=10.0,
        rho_2=1000.0,
        y_bounds=(-15000.0, 15000.0),
        dy_block=5000.0,
        z_bounds=(2000.0, 22000.0),
        dz_block=4000.0,
    ):
        """Alternating high/low resistivity checkerboard model.

        Parameters
        ----------
        bg_rho
            Background resistivity (Ohm·m).
        rho_1, rho_2
            Alternating block resistivities (Ohm·m).
        y_bounds, z_bounds
            Lateral and depth extent (m) of the checkerboard region.
        dy_block, dz_block
            Block width and height (m).

        Returns
        -------
        sig_true : torch.Tensor
            Conductivity (S/m) on cell centers, shape ``(nz, ny)``.
        """
        # 兼容 Tensor 和 Numpy
        zn_np = zn.cpu().numpy() if isinstance(zn, torch.Tensor) else np.asarray(zn)
        yn_np = yn.cpu().numpy() if isinstance(yn, torch.Tensor) else np.asarray(yn)

        z_centers = 0.5 * (zn_np[:-1] + zn_np[1:])
        y_centers = 0.5 * (yn_np[:-1] + yn_np[1:])
        Y, Z = np.meshgrid(y_centers, z_centers)

        # 1. 设置背景和空气层
        sigma = np.ones_like(Y, dtype=float) / bg_rho
        sigma[z_centers < 0, :] = 1e-9 # 空气层

        # 2. 计算网格点所在的“块”索引
        y_idx = np.floor((Y - y_bounds[0]) / dy_block)
        z_idx = np.floor((Z - z_bounds[0]) / dz_block)

        # 3. 限制棋盘格生成的物理范围
        valid_mask = (Y >= y_bounds[0]) & (Y <= y_bounds[1]) & (Z >= z_bounds[0]) & (Z <= z_bounds[1])

        # 4. 交替逻辑：利用行列索引之和的奇偶性
        checker_mask = (y_idx + z_idx) % 2 == 0

        # 5. 赋值交替电阻率
        sigma[valid_mask & checker_mask] = 1.0 / rho_1
        sigma[valid_mask & ~checker_mask] = 1.0 / rho_2

        return torch.tensor(sigma, dtype=torch.float64, device=device)