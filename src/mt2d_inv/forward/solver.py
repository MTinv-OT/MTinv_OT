# 2D total-field forward modelling with full autodiff
# (sparse operators + frequency-parallel ThreadPool).
import os
# Pin BLAS/OpenMP to one thread per solve so the C++ math libraries
# do not oversubscribe CPU cores against the Python frequency pool.
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
import concurrent.futures

try:
    import pypardiso
    HAS_PYPARDISO = True
except ImportError:
    HAS_PYPARDISO = False

# Ensure default dtype double for stability
torch.set_default_dtype(torch.float64)

# ==========================================
# Sparse autodiff solver (CPU adjoint / native complex)
# ==========================================

class SparseSolveComplex(torch.autograd.Function):
    """Sparse complex linear solver for Ax = b using Wirtinger adjoints."""
    @staticmethod
    def forward(ctx, indices, values, size_N, b):
        idx_np = indices.detach().cpu().numpy()
        val_np = values.detach().contiguous().cpu().numpy()
        b_np = b.detach().contiguous().cpu().numpy()

        ctx.b_shape = tuple(b.shape)

        if b_np.ndim == 2 and b_np.shape[1] == 1:
            b_np_solve = b_np[:, 0]
        else:
            b_np_solve = b_np

        A_sp = sp.coo_matrix((val_np, (idx_np[0], idx_np[1])), shape=(size_N, size_N)).tocsr()
        
        if HAS_PYPARDISO:
            x_np = pypardiso.spsolve(A_sp, b_np_solve)
        else:
            x_np = spla.spsolve(A_sp, b_np_solve)
        
        x = torch.from_numpy(x_np).to(b.device).to(b.dtype)

        ctx.save_for_backward(indices, values, x)
        ctx.A_sp_H = A_sp.conj().T 

        return x

    @staticmethod
    def backward(ctx, grad_x):
        indices, values, x = ctx.saved_tensors
        A_sp_H = ctx.A_sp_H
        
        grad_x_np = grad_x.detach().contiguous().cpu().numpy()

        if HAS_PYPARDISO:
            lambda_np = pypardiso.spsolve(A_sp_H, grad_x_np)
        else:
            lambda_np = spla.spsolve(A_sp_H, grad_x_np)
            
        lam = torch.from_numpy(lambda_np).to(grad_x.device).to(grad_x.dtype)

        b_shape = getattr(ctx, "b_shape", None)
        if b_shape is not None:
            grad_b = lam.reshape(b_shape)
        else:
            grad_b = lam

        row, col = indices[0], indices[1]
        grad_values = -lam[row] * torch.conj(x[col])

        return None, grad_values, None, grad_b


def complex_sparse_solve(indices, values_complex, b_complex, N):
    values_c = values_complex.to(torch.complex128)
    b_c = b_complex.to(torch.complex128)
    x_complex = SparseSolveComplex.apply(indices, values_c, N, b_c)
    return x_complex.reshape(b_complex.shape)


# ==========================================
# Main forward class (precomputed topology + native 1D BC solves)
# ==========================================
class MT2DFD_Torch(nn.Module):
    def __init__(self, nza, zn, yn, freq, ry, sig, device='cpu'):
        super().__init__()
        self.device = torch.device(device if device in ['cpu','cuda'] else 'cpu')
        self.miu = 4.0e-7 * np.pi
        self.nza = nza
        self.zn = torch.as_tensor(zn, dtype=torch.float64, device=self.device)
        self.yn = torch.as_tensor(yn, dtype=torch.float64, device=self.device)
        self.dz = self.zn[1:] - self.zn[:-1]
        self.dy = self.yn[1:] - self.yn[:-1]
        self.nz = len(zn)
        self.ny = len(yn)
        self.freq = torch.as_tensor(freq, dtype=torch.float64, device=self.device)
        self.nf = len(freq)
        self.ry = torch.as_tensor(ry, dtype=torch.float64, device=self.device)
        self.nry = len(ry)
        self.sig = torch.as_tensor(sig, dtype=torch.float64, device=self.device)

        if self.sig.shape != (self.nz-1, self.ny-1):
            raise ValueError(f"Sigma size mismatch. Expected ({self.nz-1}, {self.ny-1}), got {self.sig.shape}")

        # --------------------------------------------------------------------------
        # Precompute TE and TM grids independently
        # --------------------------------------------------------------------------
        self._precompute_geometry()

    def _precompute_geometry(self):
        ny, nz = self.ny - 1, self.nz - 1
        
        # =====================================================
        # 1. TE precompute (keep the air layers)
        # =====================================================
        self.dy0 = self.dy.view(1, -1).repeat(nz, 1)
        self.dz0 = self.dz.view(-1, 1).repeat(1, ny)
        
        self.dyc = (self.dy0[:-1, :-1] + self.dy0[:-1, 1:]) / 2.0
        self.dzc = (self.dz0[:-1, :-1] + self.dz0[1:, :-1]) / 2.0
        
        self.w1 = self.dy0[:-1, :-1] * self.dz0[:-1, :-1]
        self.w2 = self.dy0[:-1, 1:]  * self.dz0[:-1, :-1]
        self.w3 = self.dy0[:-1, :-1] * self.dz0[1:, :-1]
        self.w4 = self.dy0[:-1, 1:]  * self.dz0[1:, :-1]
        self.area = (self.w1 + self.w2 + self.w3 + self.w4) / 4.0

        self.te_val = self.dzc / self.dy0[:-1, :-1] + self.dzc / self.dy0[:-1, 1:] + \
                      self.dyc / self.dz0[:-1, :-1] + self.dyc / self.dz0[1:, :-1]

        self.coef = torch.zeros((nz+1, ny+1), dtype=torch.complex128, device=self.device)
        self.dzck = (self.dz0[:-1, 0] + self.dz0[1:, 0]) / 2.0
        self.dycj = (self.dy0[0, :-1] + self.dy0[0, 1:]) / 2.0
        self.steps = torch.linspace(0, 1, ny+1, device=self.device, dtype=torch.complex128)
        self.mid_zeros = torch.zeros((nz-1, ny-1), dtype=torch.complex128, device=self.device)

        self.num_inner_z, self.num_inner_y = nz - 1, ny - 1
        self.N = self.num_inner_z * self.num_inner_y
        self.rng = torch.arange(self.N, device=self.device)
        
        def get_idx(iz, iy, n_inner_z): return iy * n_inner_z + iz

        self.te_term_z = self.dyc[1:, :] / self.dz0[1:-1, :-1]
        iz_g_te, iy_g_te = torch.meshgrid(torch.arange(1, self.num_inner_z, device=self.device),
                                          torch.arange(self.num_inner_y, device=self.device), indexing='ij')
        self.te_row_idx, self.te_col_idx = get_idx(iz_g_te, iy_g_te, self.num_inner_z).flatten(), get_idx(iz_g_te-1, iy_g_te, self.num_inner_z).flatten()
        self.te_val_z = self.te_term_z.flatten().to(dtype=torch.complex128)

        self.te_term_y = self.dzc[:, 1:] / self.dy0[:-1, 1:-1]
        iz_range_y_te, iy_range_y_te = torch.arange(self.num_inner_z, device=self.device), torch.arange(1, self.num_inner_y, device=self.device)
        iz_gy_te, iy_gy_te = torch.meshgrid(iz_range_y_te, iy_range_y_te, indexing='ij')
        self.te_row_idy, self.te_col_idy = get_idx(iz_gy_te, iy_gy_te, self.num_inner_z).flatten(), get_idx(iz_gy_te, iy_gy_te-1, self.num_inner_z).flatten()
        self.te_val_y = self.te_term_y.flatten().to(dtype=torch.complex128)

        self.te_indices = torch.cat([
            torch.stack([self.rng, self.rng]),
            torch.stack([self.te_row_idx, self.te_col_idx]), torch.stack([self.te_col_idx, self.te_row_idx]),
            torch.stack([self.te_row_idy, self.te_col_idy]), torch.stack([self.te_col_idy, self.te_row_idy])
        ], dim=1)

        # =====================================================
        # 2. TM precompute (air layers removed)
        # =====================================================
        nz_tm = nz - self.nza
        dz_tm = self.dz[self.nza:]
        
        self.tm_dy0 = self.dy.view(1, -1).repeat(nz_tm, 1)
        self.tm_dz0 = dz_tm.view(-1, 1).repeat(1, ny)
        
        self.tm_dyc = (self.tm_dy0[:-1, :-1] + self.tm_dy0[:-1, 1:]) / 2.0
        self.tm_dzc = (self.tm_dz0[:-1, :-1] + self.tm_dz0[1:, :-1]) / 2.0

        self.tm_w1 = 2 * self.tm_dz0[:-1, :-1]
        self.tm_w2 = 2 * self.tm_dz0[1:, :-1]
        self.tm_w3 = 2 * self.tm_dy0[:-1, :-1]
        self.tm_w4 = 2 * self.tm_dy0[:-1, 1:]

        self.tm_coef = torch.zeros((nz_tm+1, ny+1), dtype=torch.complex128, device=self.device)
        self.tm_mid_zeros = torch.zeros((nz_tm-1, ny-1), dtype=torch.complex128, device=self.device)

        self.tm_num_inner_z = nz_tm - 1
        self.tm_N = self.tm_num_inner_z * self.num_inner_y
        self.tm_rng = torch.arange(self.tm_N, device=self.device)

        iz_range_tm, iy_range_tm = torch.arange(0, self.tm_num_inner_z-1, device=self.device), torch.arange(0, self.num_inner_y, device=self.device)
        iz_g_tm, iy_g_tm = torch.meshgrid(iz_range_tm, iy_range_tm, indexing='ij')
        self.tm_row_idx, self.tm_col_idx = get_idx(iz_g_tm, iy_g_tm, self.tm_num_inner_z).flatten(), get_idx(iz_g_tm+1, iy_g_tm, self.tm_num_inner_z).flatten()

        iz_range_y_tm, iy_range_y_tm = torch.arange(0, self.tm_num_inner_z, device=self.device), torch.arange(0, self.num_inner_y-1, device=self.device)
        iz_gy_tm, iy_gy_tm = torch.meshgrid(iz_range_y_tm, iy_range_y_tm, indexing='ij')
        self.tm_row_idy, self.tm_col_idy = get_idx(iz_gy_tm, iy_gy_tm, self.tm_num_inner_z).flatten(), get_idx(iz_gy_tm, iy_gy_tm+1, self.tm_num_inner_z).flatten()

        self.tm_indices = torch.cat([
            torch.stack([self.tm_rng, self.tm_rng]),
            torch.stack([self.tm_row_idx, self.tm_col_idx]), torch.stack([self.tm_col_idx, self.tm_row_idx]),
            torch.stack([self.tm_row_idy, self.tm_col_idy]), torch.stack([self.tm_col_idy, self.tm_row_idy])
        ], dim=1)
        # --------------------------------------------------------------------------
        # Warm up torch.linalg.solve on the main thread so the first
        # ThreadPoolExecutor call does not hit lazy-init races
        # ("RuntimeError: lazy wrapper should be called at most once").
        _dummy_A = torch.eye(2, dtype=torch.complex128, device=self.device)
        _dummy_B = torch.ones((2, 1), dtype=torch.complex128, device=self.device)
        _ = torch.linalg.solve(_dummy_A, _dummy_B)
        # Warm-up complete.

    def forward(self, mode="TETM"):
        res = {}
        if "TE" in mode:
            rhoxy, phsxy, Zxy = self.solve_te()
            res['rhoxy'] = rhoxy
            res['phsxy'] = phsxy
            res['Zxy'] = Zxy
        if "TM" in mode:
            rhoyx, phsyx, Zyx = self.solve_tm()
            res['rhoyx'] = rhoyx
            res['phsyx'] = phsyx
            res['Zyx'] = Zyx
        return res

    def interp1d_torch(self, x_new, x_old, y_old):
        idxs = torch.searchsorted(x_old, x_new)
        idxs = torch.clamp(idxs, 1, len(x_old)-1)
        x_left = x_old[idxs-1]
        x_right = x_old[idxs]
        y_left = y_old[idxs-1]
        y_right = y_old[idxs]
        weight = (x_new - x_left) / (x_right - x_left + 1e-12)
        y_new = y_left + weight * (y_right - y_left)
        return y_new

    def solve_te(self):
        dy, dz = self.dy, self.dz
        sig, yn, ry, nza = self.sig, self.yn, self.ry, self.nza
        nf = self.nf
        
        Zxy_list = [None] * nf
        rhoxy_list = [None] * nf
        phsxy_list = [None] * nf

        def _compute_te_freq(kf):
            freq_val = self.freq[kf]
            ex = self.mt2dte_solver(freq_val, sig)
            hys, _ = self.mt2dhyhz(freq_val, dy, dz, sig, ex)
            exs = ex[nza, :]
            exr = self.interp1d_torch(ry, yn, exs)
            hyr = self.interp1d_torch(ry, yn, hys)
            Z, rho, phs = self.calc_impedance(freq_val, exr, hyr, mode='xy')
            return kf, Z, rho, phs

        max_workers = min(nf, 48)
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = [executor.submit(_compute_te_freq, kf) for kf in range(nf)]
            for future in concurrent.futures.as_completed(futures):
                kf, Z, rho, phs = future.result()
                Zxy_list[kf] = Z
                rhoxy_list[kf] = rho
                phsxy_list[kf] = phs

        rhoxy = torch.stack(rhoxy_list)
        phsxy = torch.stack(phsxy_list)
        Zxy = torch.stack(Zxy_list)
        return rhoxy, phsxy, Zxy

    def solve_tm(self):
        nza = self.nza
        dz = self.dz[nza:]
        sig = self.sig[nza:, :]
        dy, yn, ry = self.dy, self.yn, self.ry
        nf = self.nf
        
        Zyx_list = [None] * nf
        rhoyx_list = [None] * nf
        phsyx_list = [None] * nf

        def _compute_tm_freq(kf):
            freq_val = self.freq[kf]
            hx = self.mt2dtm_solver(freq_val, sig, dz)
            eys, _ = self.mt2deyez(freq_val, dy, dz, sig, hx)
            hxs = hx[0, :]
            hxr = self.interp1d_torch(ry, yn, hxs)
            eyr = self.interp1d_torch(ry, yn, eys)
            Z, rho, phs = self.calc_impedance(freq_val, eyr, hxr, mode='yx')
            return kf, Z, rho, phs

        max_workers = min(nf, 48)
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = [executor.submit(_compute_tm_freq, kf) for kf in range(nf)]
            for future in concurrent.futures.as_completed(futures):
                kf, Z, rho, phs = future.result()
                Zyx_list[kf] = Z
                rhoyx_list[kf] = rho
                phsyx_list[kf] = phs

        rhoyx = torch.stack(rhoyx_list)
        phsyx = torch.stack(phsyx_list)
        Zyx = torch.stack(Zyx_list)
        return rhoyx, phsyx, Zyx

    def mt2dhyhz(self, freq, dy, dz, sig, ex):
        omega = 2.0 * np.pi * freq
        kk = self.nza
        delz = dz[kk]
        sigma_surf = sig[kk, :] 
        
        factor_hz = 1.0 / (1j * omega * self.miu)
        ex_surf = ex[kk, :]
        dy_total = dy[:-1] + dy[1:]
        
        mid_part = -factor_hz * (ex_surf[2:] - ex_surf[:-2]) / dy_total
        left_val = -factor_hz * (ex_surf[1] - ex_surf[0]) / dy[0]
        right_val = -factor_hz * (ex_surf[-1] - ex_surf[-2]) / dy[-1]
        
        hzs = torch.cat([left_val.view(1), mid_part, right_val.view(1)])

        sig_node = torch.zeros_like(ex[0])
        sig_node[1:-1] = (sigma_surf[:-1] + sigma_surf[1:]) / 2.0
        sig_node[0] = sigma_surf[0]
        sig_node[-1] = sigma_surf[-1]
        
        term1_node = 1.0 / (1j * omega * self.miu * delz)
        term2_node = (3.0 / 8.0) * sig_node * delz
        term3_node = (1.0 / 8.0) * sig_node * delz
        
        c0_node = -term1_node + term2_node
        c1_node =  term1_node + term3_node
        
        hys = c0_node * ex[kk, :] + c1_node * ex[kk+1, :]
        return hys, hzs

    def mt2deyez(self, freq, dy, dz, sig, hx):
        kk = 0 
        delz = dz[kk]
        sigma_surf = sig[kk, :]
        dHx_dz = (hx[kk+1, :] - hx[kk, :]) / delz
        
        sig_main = sigma_surf
        sig_last = sigma_surf[-1].view(1)
        sig_use = torch.cat([sig_main, sig_last])
        
        eys = - (1.0/sig_use) * dHx_dz
        return eys, None

    def calc_impedance(self, freq, E_field, H_field, mode='xy'):
        omega = 2.0 * np.pi * freq
        if not torch.is_complex(E_field): E_field = E_field.to(torch.complex128)
        if not torch.is_complex(H_field): H_field = H_field.to(torch.complex128)

        Z = E_field / H_field
        rho = torch.abs(Z)**2 / (omega * self.miu)
        
        phs_raw = torch.atan2(Z.imag, Z.real) * 180.0 / np.pi
        phs_0_180 = torch.remainder(phs_raw, 180.0)
        phs = torch.minimum(phs_0_180, 180.0 - phs_0_180)
        return Z, rho.to(torch.float64), phs.to(torch.float64)

    def mt2dte_solver(self, freq, sig):
        ny, nz = self.ny - 1, self.nz - 1
        omega = 2.0 * np.pi * freq
        
        sigc = (sig[:-1,:-1]*self.w1 + sig[:-1,1:]*self.w2 + sig[1:,:-1]*self.w3 + sig[1:,1:]*self.w4)/(self.area*4.0)
        mtx1 = 1j * omega * self.miu * sigc * self.area - self.te_val
        
        diag_val = mtx1.T.flatten()
        values = torch.cat([diag_val, self.te_val_z, self.te_val_z, self.te_val_y, self.te_val_y])
        
        coef = self.coef.clone()
        coef[1:nz, 0] = (self.dzck / self.dy0[0, 0]).to(torch.complex128)
        coef[1:nz, ny] = (self.dzck / self.dy0[0, -1]).to(torch.complex128)
        coef[0, 1:ny] = (self.dycj / self.dz0[0, 0]).to(torch.complex128)
        coef[nz, 1:ny] = (self.dycj / self.dz0[-1, 0]).to(torch.complex128)
        
        ex_l = self.mt1dte_solver(freq, self.dz, sig[:, 0])
        ex_r = self.mt1dte_solver(freq, self.dz, sig[:, -1])
        
        top_row_inner = (ex_l[0,0] + (ex_r[0,0] - ex_l[0,0]) * self.steps[1:-1]).view(1, -1)
        bot_row_inner = (ex_l[-1,0] + (ex_r[-1,0] - ex_l[-1,0]) * self.steps[1:-1]).view(1, -1)
        
        col_inner = torch.cat([top_row_inner, self.mid_zeros, bot_row_inner], dim=0)
        ex1d = torch.cat([ex_l, col_inner, ex_r], dim=1)
        
        coef_base = ex1d * coef
        rhs_center = coef_base[1:nz, 1:ny]
        
        term_top = coef_base[0, 1:ny].view(1, -1)
        pad_top = F.pad(term_top, (0, 0, 0, (nz-1)-1)) 
        term_bot = coef_base[nz, 1:ny].view(1, -1)
        pad_bot = F.pad(term_bot, (0, 0, (nz-1)-1, 0)) 
        term_left = coef_base[1:nz, 0].view(-1, 1)
        pad_left = F.pad(term_left, (0, (ny-1)-1, 0, 0)) 
        term_right = coef_base[1:nz, ny].view(-1, 1)
        pad_right = F.pad(term_right, ((ny-1)-1, 0, 0, 0)) 
        
        rhs = rhs_center + pad_top + pad_bot + pad_left + pad_right
        b_vec = -rhs.T.flatten()
        
        ex_flat = complex_sparse_solve(self.te_indices, values, b_vec, self.N)
        ex_inner = ex_flat.view(ny-1, nz-1).T
        
        row_top = ex1d[0:1, :]
        row_bot = ex1d[nz:nz+1, :]
        col_left = ex1d[1:nz, 0:1]
        col_right = ex1d[1:nz, ny:ny+1]
        middle_layer = torch.cat([col_left, ex_inner, col_right], dim=1)
        ex_full = torch.cat([row_top, middle_layer, row_bot], dim=0)
        
        return ex_full

    def mt2dtm_solver(self, freq, sig, dz_tm):
        ny, nz_tm = self.ny - 1, self.nz - 1 - self.nza
        omega = 2.0 * np.pi * freq
        rho = 1.0 / sig
        
        r_tl, r_tr = rho[:-1, :-1], rho[:-1, 1:]
        r_bl, r_br = rho[1:, :-1], rho[1:, 1:]
        
        term_A = (r_tl * self.tm_dy0[:-1, :-1] + r_tr * self.tm_dy0[:-1, 1:]) / self.tm_w1
        term_B = (r_bl * self.tm_dy0[:-1, :-1] + r_br * self.tm_dy0[:-1, 1:]) / self.tm_w2
        term_C = (r_tl * self.tm_dz0[:-1, :-1] + r_bl * self.tm_dz0[1:, :-1]) / self.tm_w3
        term_D = (r_tr * self.tm_dz0[:-1, :-1] + r_br * self.tm_dz0[1:, :-1]) / self.tm_w4
        
        mtx1 = 1j * omega * self.miu * self.tm_dyc * self.tm_dzc - term_A - term_B - term_C - term_D
        
        diag_val = mtx1.T.flatten()
        val_z = term_B[:-1, :].flatten().to(torch.complex128)
        val_y = term_D[:, :-1].flatten().to(torch.complex128)

        values = torch.cat([diag_val, val_z, val_z, val_y, val_y])

        coef = self.tm_coef.clone()
        coef[1:nz_tm, 0] = term_C[:, 0].to(torch.complex128)
        coef[1:nz_tm, ny] = term_D[:, -1].to(torch.complex128)
        coef[0, 1:ny] = term_A[0, :].to(torch.complex128)
        coef[nz_tm, 1:ny] = term_B[-1, :].to(torch.complex128)
        
        hx_l = self.mt1dtm_solver(freq, dz_tm, sig[:, 0])
        hx_r = self.mt1dtm_solver(freq, dz_tm, sig[:, -1])
        
        top_row_inner = (hx_l[0,0] + (hx_r[0,0] - hx_l[0,0]) * self.steps[1:-1]).view(1, -1)
        bot_row_inner = (hx_l[-1,0] + (hx_r[-1,0] - hx_l[-1,0]) * self.steps[1:-1]).view(1, -1)
        
        col_inner = torch.cat([top_row_inner, self.tm_mid_zeros, bot_row_inner], dim=0)
        hx1d = torch.cat([hx_l, col_inner, hx_r], dim=1)
        
        coef_base = hx1d * coef
        rhs_center = coef_base[1:nz_tm, 1:ny]
        
        term_top = coef_base[0, 1:ny].view(1, -1)
        pad_top = F.pad(term_top, (0, 0, 0, (nz_tm-1)-1))
        term_bot = coef_base[nz_tm, 1:ny].view(1, -1)
        pad_bot = F.pad(term_bot, (0, 0, (nz_tm-1)-1, 0))
        term_left = coef_base[1:nz_tm, 0].view(-1, 1)
        pad_left = F.pad(term_left, (0, (ny-1)-1, 0, 0))
        term_right = coef_base[1:nz_tm, ny].view(-1, 1)
        pad_right = F.pad(term_right, ((ny-1)-1, 0, 0, 0))
        
        rhs = rhs_center + pad_top + pad_bot + pad_left + pad_right
        b_vec = -rhs.T.flatten()
        
        hx_flat = complex_sparse_solve(self.tm_indices, values, b_vec, self.tm_N)
        hx_inner = hx_flat.view(ny-1, nz_tm-1).T
        
        row_top = hx1d[0:1, :]
        row_bot = hx1d[nz_tm:nz_tm+1, :]
        col_left = hx1d[1:nz_tm, 0:1]
        col_right = hx1d[1:nz_tm, ny:ny+1]
        middle_layer = torch.cat([col_left, hx_inner, col_right], dim=1)
        hx_full = torch.cat([row_top, middle_layer, row_bot], dim=0)
        
        return hx_full

    # --------------------------------------------------------------------------
    # Lightweight 1D boundary-condition solver (dense torch)
    # --------------------------------------------------------------------------
    def mt1dte_solver(self, freq, dz, sig):
        omega = 2.0 * np.pi * freq
        nz = len(sig)
        last_sig = sig[-1]
        skin_depth = torch.sqrt(2.0 / (last_sig * omega * self.miu))
        dz_ext = torch.cat([dz, skin_depth.view(1)])
        sig_ext = torch.cat([sig, last_sig.view(1)])
        
        term1 = 1j * omega * self.miu * (sig_ext[:-1]*dz_ext[:-1] + sig_ext[1:]*dz_ext[1:])
        term2 = -2.0 / dz_ext[:-1] - 2.0 / dz_ext[1:]
        diag = term1 + term2
        
        # 1D FD tridiagonal: off-diagonals have length nz - 1
        off_upper = (2.0 / dz_ext[1:-1]).to(torch.complex128)
        off_lower = off_upper.clone() 
        
        # Build a dense tridiagonal matrix
        A_dense = torch.diag(diag) + torch.diag(off_upper, 1) + torch.diag(off_lower, -1)
        
        val0 = (-2.0 / dz_ext[0]).view(1, 1).to(torch.complex128)
        zeros_rest = torch.zeros((nz-1, 1), dtype=torch.complex128, device=self.device)
        rhs = torch.cat([val0, zeros_rest], dim=0)
        
        # Native torch solve
        res = torch.linalg.solve(A_dense, rhs)
        return torch.cat([torch.tensor([[1.0]], device=self.device, dtype=torch.complex128), res], dim=0)

    def mt1dtm_solver(self, freq, dz, sig):
        omega = 2.0 * np.pi * freq
        nz = len(sig)
        last_sig = sig[-1]
        skin_depth = torch.sqrt(2.0 / (last_sig * omega * self.miu))
        dz_ext = torch.cat([dz, skin_depth.view(1)])
        sig_ext = torch.cat([sig, last_sig.view(1)])
        
        term1 = 1j * omega * self.miu * (dz_ext[:-1] + dz_ext[1:])
        term2 = -2.0 / (dz_ext[:-1]*sig_ext[:-1]) - 2.0 / (dz_ext[1:]*sig_ext[1:])
        diag = term1 + term2
        
        # Off-diagonals must have length nz - 1
        off_upper = (2.0 / (dz_ext[1:-1]*sig_ext[1:-1])).to(torch.complex128)
        off_lower = off_upper.clone()
        
        # Build a dense tridiagonal matrix
        A_dense = torch.diag(diag) + torch.diag(off_upper, 1) + torch.diag(off_lower, -1)
        
        val0 = (-2.0 / (dz_ext[0] * sig_ext[0])).view(1, 1).to(torch.complex128)
        zeros_rest = torch.zeros((nz-1, 1), dtype=torch.complex128, device=self.device)
        rhs = torch.cat([val0, zeros_rest], dim=0)
        
        # Native torch solve
        res = torch.linalg.solve(A_dense, rhs)
        return torch.cat([torch.tensor([[1.0]], device=self.device, dtype=torch.complex128), res], dim=0)
