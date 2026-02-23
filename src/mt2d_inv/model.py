"""Standard test model generators."""
import torch
import numpy as np


class MT2DTrueModels:
    """2D MT true-model generators.

    This class provides several standard benchmark/test models that can be used
    by the main workflow when building the true conductivity model.
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
        z_air = -np.logspace(np.log10(10.0), np.log10(50000.0), nza)
        z_air = np.flip(z_air)
        z_air = np.append(z_air, 0.0)

        n_shallow = 10
        z_shallow = np.logspace(np.log10(100.0), np.log10(5000.0), n_shallow)

        n_deep = 20
        z_deep = np.logspace(np.log10(6000.0), np.log10(100000.0), n_deep)

        z_sub = np.concatenate([z_shallow, z_deep])
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
    def create_single_anomaly(zn=None, yn=None, nza=10, device="cpu"):
        """
        Single-anomaly model.

        If zn/yn are not provided, this function uses the default grid (controlled
        by `nza`).

        Conductivity:
        - Background: 0.01 S/m
        - Air: 1e-9 S/m
        - Anomaly: 1.0 S/m where |Y| < 3000 m and 5000 < Z < 10000 m

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

        mask_anomaly = (np.abs(Y) < 3000) & (Z > 5000) & (Z < 10000)
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
        3x3 block resistivity pattern (Ohm·m) within a specified window.

        Pattern (rows are depth, columns are lateral):

        - 0–10 km:   [500, 20, 500]
        - 10–20 km:  [10,  10, 10]
        - 20–30 km:  [500, 20, 500]

        Lateral window: -15 to 15 km, split into 3 equal-width blocks.
        Vertical window: 0 to 30 km, split into 3 layers (10 km each).

        Notes
        -----
        `anomaly_rho_low`, `anomaly_rho_high`, `block_w`, `block_h` are kept only for
        signature compatibility and are not used in this implementation.

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

        # 3x3 resistivity assignment inside [-15, 15] km and [0, 30] km
        rho_table = np.array(
            [
                [500.0, 20.0, 500.0],
                [10.0, 10.0, 10.0],
                [500.0, 20.0, 500.0],
            ],
            dtype=float,
        )

        y_edges = np.array([-15.0, -5.0, 5.0, 15.0], dtype=float)
        z_edges = np.array([0.0, 10.0, 20.0, 30.0], dtype=float)

        for iz in range(3):
            z0, z1 = z_edges[iz], z_edges[iz + 1]
            z_mask = (Zkm >= z0) & (Zkm < z1) if iz < 2 else (Zkm >= z0) & (Zkm <= z1)
            for iy in range(3):
                y0, y1 = y_edges[iy], y_edges[iy + 1]
                y_mask = (Ykm >= y0) & (Ykm < y1) if iy < 2 else (Ykm >= y0) & (Ykm <= y1)
                mask = z_mask & y_mask
                sigma[mask] = 1.0 / rho_table[iz, iy]

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
    def create_three_block_5_10km(
        zn=None,
        yn=None,
        nza=10,
        device="cpu",
        bg_sigma=0.01,
        high_rho=1000.0,
        low_rho=1.0,
    ):
        """Three-block model at 5–10 km depth.

        Left high-resistivity, center low-resistivity, right high-resistivity (symmetric).

        - Left:  -10 to  -1 km (high_rho)
        - Center: -1 to   1 km (low_rho)
        - Right:  1 to  10 km (high_rho)

        Background conductivity: bg_sigma (S/m).

        Returns (yn, zn, nza, sigma)
        """

        if zn is None and yn is None:
            yn, zn, nza = MT2DTrueModels._default_grid(nza=nza)
        elif zn is None or yn is None:
            raise ValueError("create_three_block_5_10km: zn and yn must be provided together, or both be None to use the default grid")
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

        z_min, z_max = 5e3, 10e3
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
            dist_sq = (Y - 5.0) ** 2 + (Z - 20.0) ** 2
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
