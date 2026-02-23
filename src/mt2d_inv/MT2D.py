#总场法二维正演，全自动微分
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.autograd import Function
import matplotlib.pyplot as plt

# Ensure default dtype double for stability
torch.set_default_dtype(torch.float64)

# ---------------------------
# Utilities: real-block solver
# ---------------------------
def _real_block_from_complex(A: torch.Tensor) -> torch.Tensor:
    if not torch.is_complex(A):
        A = A.to(torch.complex128)
    Ar = A.real
    Ai = A.imag
    top = torch.cat([Ar, -Ai], dim=-1)
    bot = torch.cat([Ai,  Ar], dim=-1)
    return torch.cat([top, bot], dim=-2)

def _vec_to_real(b: torch.Tensor) -> torch.Tensor:
    if not torch.is_complex(b):
        b = b.to(torch.complex128)
    br = b.real.reshape(-1)
    bi = b.imag.reshape(-1)
    return torch.cat([br, bi], dim=0)

def _real_to_complex_vec(x_real: torch.Tensor) -> torch.Tensor:
    N2 = x_real.shape[-1]
    N = N2 // 2
    xr = x_real[:N]
    xi = x_real[N:]
    return xr + 1j * xi

def complex_solve_block(A_complex: torch.Tensor, b_complex: torch.Tensor) -> torch.Tensor:
    if not torch.is_complex(A_complex):
        A_complex = A_complex.to(torch.complex128)
    if not torch.is_complex(b_complex):
        b_complex = b_complex.to(torch.complex128)

    if A_complex.ndim != 2 or A_complex.shape[0] != A_complex.shape[1]:
        raise ValueError("complex_solve_block expects a square 2D complex matrix A.")

    A_block = _real_block_from_complex(A_complex)
    b_real = _vec_to_real(b_complex)
    x_real = torch.linalg.solve(A_block, b_real)
    x_complex = _real_to_complex_vec(x_real)

    return x_complex.reshape(b_complex.shape)

# ---------------------------
# Helper: Safe Sparse Matrix Construction
# ---------------------------
def make_sparse_A(indices, values, size, device):
    """
    Construct dense matrix A from sparse indices/values to avoid in-place assignment A[...] = val
    """
    A_sp = torch.sparse_coo_tensor(indices, values, size=size, device=device)
    return A_sp.to_dense()

# ==========================================
# 主模型类
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
        # 简单的线性插值，支持梯度
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
        
        Zxy_list, rhoxy_list, phsxy_list = [], [], []

        for kf in range(nf):
            freq_val = self.freq[kf]
            # 1. 求解 Ex 场
            ex = self.mt2dte_solver(freq_val, dy, dz, sig)
            
            # 2. 计算磁场 Hy (已应用高频修正)
            hys, _ = self.mt2dhyhz(freq_val, dy, dz, sig, ex)
            
            # 3. 提取地表 Ex
            exs = ex[nza, :]
            
            # 4. 插值到测点
            exr = self.interp1d_torch(ry, yn, exs)
            hyr = self.interp1d_torch(ry, yn, hys)
            
            # 5. 计算阻抗
            Z, rho, phs = self.calc_impedance(freq_val, exr, hyr, mode='xy')

            Zxy_list.append(Z)
            rhoxy_list.append(rho)
            phsxy_list.append(phs)

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
        
        Zyx_list, rhoyx_list, phsyx_list = [], [], []

        for kf in range(nf):
            freq_val = self.freq[kf]
            hx = self.mt2dtm_solver(freq_val, dy, dz, sig)
            eys, _ = self.mt2deyez(freq_val, dy, dz, sig, hx)
            hxs = hx[0, :]
            hxr = self.interp1d_torch(ry, yn, hxs)
            eyr = self.interp1d_torch(ry, yn, eys)
            
            Z, rho, phs = self.calc_impedance(freq_val, eyr, hxr, mode='yx')
            Zyx_list.append(Z)
            rhoyx_list.append(rho)
            phsyx_list.append(phs)

        rhoyx = torch.stack(rhoyx_list)
        phsyx = torch.stack(phsyx_list)
        Zyx = torch.stack(Zyx_list)
        return rhoyx, phsyx, Zyx

    # --------------------------------------------------------------------------
    # 核心修正函数: MT2D TE 模式磁场计算
    # --------------------------------------------------------------------------
    def mt2dhyhz(self, freq, dy, dz, sig, ex):
        """
        Calculates Hy and Hz from Ex.
        [Updated] Uses robust integral coefficients for Hy to fix high-frequency errors.
        """
        omega = 2.0 * np.pi * freq
        kk = self.nza
        delz = dz[kk]
        
        # 获取地表下第一层介质的电导率 (row index kk)
        # sig 维度 (nz-1, ny-1), Ex 维度 (nz, ny)
        sigma_surf = sig[kk, :] 
        
        # 1. 计算 Hz (保持原有的水平差分逻辑，影响较小)
        factor_hz = 1.0 / (1j * omega * self.miu)
        ex_surf = ex[kk, :]
        dy_total = dy[:-1] + dy[1:]
        
        mid_part = -factor_hz * (ex_surf[2:] - ex_surf[:-2]) / dy_total
        left_val = -factor_hz * (ex_surf[1] - ex_surf[0]) / dy[0]
        right_val = -factor_hz * (ex_surf[-1] - ex_surf[-2]) / dy[-1]
        
        hzs = torch.cat([left_val.view(1), mid_part, right_val.view(1)])

        # 2. 计算 Hy (核心修正：积分形式系数)
        # 原始逻辑: dEx_dz = (ex[kk+1] - ex[kk])/delz; hy = -factor * dEx_dz
        # 新逻辑: 考虑网格内指数衰减的系数 c0, c1
        
        term1 = 1.0 / (1j * omega * self.miu * delz)
        # 引入电导率修正项 (3/8 和 1/8 是线性电流密度近似下的积分系数)
        term2 = (3.0 / 8.0) * sigma_surf * delz
        term3 = (1.0 / 8.0) * sigma_surf * delz
        
        c0 = -term1 + term2
        c1 =  term1 + term3
        
        # c0, c1 维度为 [ny-1]，ex 需要对应列
        # 注意: ex 的列数是 ny (节点), sigma 的列数是 ny-1 (单元)
        # 我们计算出的 Hy 是定义在单元顶部的 "垂直方向磁场"? 
        # 不，通常 Hy 定义在地表节点上。
        # 但这里的 Robust 公式通常针对 1D 柱状体假设。
        # 为了匹配维度，我们假设 sigma_surf 对应的 Ex 节点是 ex[kk, :-1] 和 ex[kk, 1:] 之间的区域?
        # 不，标准 FD 中 Hy 和 Ex 在水平位置是对齐的 (节点对齐)。
        # 在 coarse grid fix 中，通常使用节点下方的 sigma。
        # 这里的 sigma_surf 长度为 ny-1 (单元中心)。
        # 为了简单且有效，我们将 sigma_surf 扩展到节点位置 (简单平均) 或者直接计算单元中心的 Hy 再插值。
        # 但根据参考代码，它是直接操作的。
        # 参考代码逻辑: hy_robust = c0 * ex_0 + c1 * ex_1
        # 这里 ex_0 和 ex_1 是垂直方向的两个点 (Top, Bottom of first layer)
        # 所以 c0, c1 应该是针对每个水平位置 y 计算的。
        # sigma_surf 是 [ny-1]。Ex 是 [ny]。
        # 为了保证维度匹配，我们需要把 sigma 插值到节点，或者把 Hy 计算在单元中心。
        # 考虑到 interp1d_torch 后面会再次插值，这里我们在**节点**处计算 Hy 最方便。
        # 我们采用简单的节点电导率平均：
        
        sig_node = torch.zeros_like(ex[0])
        sig_node[1:-1] = (sigma_surf[:-1] + sigma_surf[1:]) / 2.0
        sig_node[0] = sigma_surf[0]
        sig_node[-1] = sigma_surf[-1]
        
        # 重新计算节点位置的系数
        term1_node = 1.0 / (1j * omega * self.miu * delz) # 标量
        term2_node = (3.0 / 8.0) * sig_node * delz
        term3_node = (1.0 / 8.0) * sig_node * delz
        
        c0_node = -term1_node + term2_node
        c1_node =  term1_node + term3_node
        
        # ex[kk, :] 是地表，ex[kk+1, :] 是地下第一层节点
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
        phs = -torch.atan2(Z.imag, Z.real) * 180.0 / np.pi
        return Z, rho.to(torch.float64), phs.to(torch.float64)

    # --------------------------------------------------------------------------
    # 下方为 Solver 辅助部分 (TE / TM / 1D) - 保持原始逻辑结构，确保矩阵构建稳定
    # --------------------------------------------------------------------------
    def mt2dte_solver(self, freq, dy, dz, sig):
        omega = 2.0 * np.pi * freq
        ny, nz = len(dy), len(dz)

        # ... Geometry setup ...
        dy0 = dy.view(1, -1).repeat(nz, 1)
        dz0 = dz.view(-1, 1).repeat(1, ny)
        dyc = (dy0[:-1, :-1] + dy0[:-1, 1:]) / 2.0
        dzc = (dz0[:-1, :-1] + dz0[1:, :-1]) / 2.0
        w1, w2 = dy0[:-1, :-1]*dz0[:-1, :-1], dy0[:-1, 1:]*dz0[:-1, :-1]
        w3, w4 = dy0[:-1, :-1]*dz0[1:, :-1], dy0[:-1, 1:]*dz0[1:, :-1]
        area = (w1+w2+w3+w4)/4.0
        sigc = (sig[:-1,:-1]*w1 + sig[:-1,1:]*w2 + sig[1:,:-1]*w3 + sig[1:,1:]*w4)/(area*4.0)
        
        val = dzc/dy0[:-1,:-1] + dzc/dy0[:-1,1:] + dyc/dz0[:-1,:-1] + dyc/dz0[1:,:-1]
        mtx1 = 1j * omega * self.miu * sigc * area - val
        
        num_inner_z, num_inner_y = nz - 1, ny - 1
        N = num_inner_z * num_inner_y

        # Build Matrix A using Sparse construction
        diag_val = mtx1.T.flatten()
        rng = torch.arange(N, device=self.device)
        
        term_z = dyc[1:, :] / dz0[1:-1, :-1]
        def get_idx(iz, iy): return iy * num_inner_z + iz
        
        iz_g, iy_g = torch.meshgrid(torch.arange(1, num_inner_z, device=self.device),
                                    torch.arange(num_inner_y, device=self.device), indexing='ij')
        row_idx, col_idx = get_idx(iz_g, iy_g).flatten(), get_idx(iz_g-1, iy_g).flatten()
        val_z = term_z.flatten().to(dtype=torch.complex128)

        term_y = dzc[:, 1:] / dy0[:-1, 1:-1]
        iz_range_y, iy_range_y = torch.arange(num_inner_z, device=self.device), torch.arange(1, num_inner_y, device=self.device)
        iz_gy, iy_gy = torch.meshgrid(iz_range_y, iy_range_y, indexing='ij')
        row_idy, col_idy = get_idx(iz_gy, iy_gy).flatten(), get_idx(iz_gy, iy_gy-1).flatten()
        val_y = term_y.flatten().to(dtype=torch.complex128)

        indices = torch.cat([
            torch.stack([rng, rng]),
            torch.stack([row_idx, col_idx]), torch.stack([col_idx, row_idx]),
            torch.stack([row_idy, col_idy]), torch.stack([col_idy, row_idy])
        ], dim=1)
        values = torch.cat([diag_val, val_z, val_z, val_y, val_y])
        
        A = make_sparse_A(indices, values, (N, N), self.device)

        # Constants for RHS
        coef = torch.zeros((nz+1, ny+1), dtype=torch.complex128, device=self.device)
        dzck = (dz0[:-1, 0] + dz0[1:, 0]) / 2.0
        dycj = (dy0[0, :-1] + dy0[0, 1:]) / 2.0
        coef[1:nz, 0] = (dzck / dy0[0, 0]).to(torch.complex128)
        coef[1:nz, ny] = (dzck / dy0[0, -1]).to(torch.complex128)
        coef[0, 1:ny] = (dycj / dz0[0, 0]).to(torch.complex128)
        coef[nz, 1:ny] = (dycj / dz0[-1, 0]).to(torch.complex128)
        
        # Build ex1d
        ex_l = self.mt1dte_solver(freq, dz, sig[:, 0])
        ex_r = self.mt1dte_solver(freq, dz, sig[:, -1])
        
        steps = torch.linspace(0, 1, ny+1, device=self.device, dtype=torch.complex128)
        top_row_inner = (ex_l[0,0] + (ex_r[0,0] - ex_l[0,0]) * steps[1:-1]).view(1, -1)
        bot_row_inner = (ex_l[-1,0] + (ex_r[-1,0] - ex_l[-1,0]) * steps[1:-1]).view(1, -1)
        mid_zeros = torch.zeros((nz-1, ny-1), dtype=torch.complex128, device=self.device)
        col_inner = torch.cat([top_row_inner, mid_zeros, bot_row_inner], dim=0)
        ex1d = torch.cat([ex_l, col_inner, ex_r], dim=1)
        
        # RHS Calculation
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
        
        # Solve
        b_vec = -rhs.T.flatten()
        ex_flat = complex_solve_block(A, b_vec)
        ex_inner = ex_flat.view(ny-1, nz-1).T
        
        # Assemble ex_full
        row_top = ex1d[0:1, :]
        row_bot = ex1d[nz:nz+1, :]
        col_left = ex1d[1:nz, 0:1]
        col_right = ex1d[1:nz, ny:ny+1]
        middle_layer = torch.cat([col_left, ex_inner, col_right], dim=1)
        ex_full = torch.cat([row_top, middle_layer, row_bot], dim=0)
        
        return ex_full

    def mt2dtm_solver(self, freq, dy, dz, sig):
        omega = 2.0 * np.pi * freq
        ny, nz = len(dy), len(dz)

        dy0 = dy.view(1, -1).repeat(nz, 1)
        dz0 = dz.view(-1, 1).repeat(1, ny)
        dyc = (dy0[:-1, :-1] + dy0[:-1, 1:]) / 2.0
        dzc = (dz0[:-1, :-1] + dz0[1:, :-1]) / 2.0
        w1, w2 = 2 * dz0[:-1, :-1], 2 * dz0[1:, :-1]
        w3, w4 = 2 * dy0[:-1, :-1], 2 * dy0[:-1, 1:]
        
        rho = 1.0 / sig
        r_tl, r_tr = rho[:-1, :-1], rho[:-1, 1:]
        r_bl, r_br = rho[1:, :-1], rho[1:, 1:]
        term_A = (r_tl * dy0[:-1, :-1] + r_tr * dy0[:-1, 1:]) / w1
        term_B = (r_bl * dy0[:-1, :-1] + r_br * dy0[:-1, 1:]) / w2
        term_C = (r_tl * dz0[:-1, :-1] + r_bl * dz0[1:, :-1]) / w3
        term_D = (r_tr * dz0[:-1, :-1] + r_br * dz0[1:, :-1]) / w4
        
        mtx1 = 1j * omega * self.miu * dyc * dzc - term_A - term_B - term_C - term_D
        num_inner_z, num_inner_y = nz - 1, ny - 1
        N = num_inner_z * num_inner_y

        diag_val = mtx1.T.flatten()
        rng = torch.arange(N, device=self.device)
        def get_idx(iz, iy): return iy * num_inner_z + iz

        iz_range, iy_range = torch.arange(0, num_inner_z-1, device=self.device), torch.arange(0, num_inner_y, device=self.device)
        iz_g, iy_g = torch.meshgrid(iz_range, iy_range, indexing='ij')
        row_idx, col_idx = get_idx(iz_g, iy_g).flatten(), get_idx(iz_g+1, iy_g).flatten()
        val_z = term_B[:-1, :].flatten().to(torch.complex128)

        iz_range_y, iy_range_y = torch.arange(0, num_inner_z, device=self.device), torch.arange(0, num_inner_y-1, device=self.device)
        iz_gy, iy_gy = torch.meshgrid(iz_range_y, iy_range_y, indexing='ij')
        row_idy, col_idy = get_idx(iz_gy, iy_gy).flatten(), get_idx(iz_gy, iy_gy+1).flatten()
        val_y = term_D[:, :-1].flatten().to(torch.complex128)

        indices = torch.cat([
            torch.stack([rng, rng]),
            torch.stack([row_idx, col_idx]), torch.stack([col_idx, row_idx]),
            torch.stack([row_idy, col_idy]), torch.stack([col_idy, row_idy])
        ], dim=1)
        values = torch.cat([diag_val, val_z, val_z, val_y, val_y])
        A = make_sparse_A(indices, values, (N, N), self.device)

        # Coef - constants
        coef = torch.zeros((nz+1, ny+1), dtype=torch.complex128, device=self.device)
        coef[1:nz, 0] = term_C[:, 0].to(torch.complex128); coef[1:nz, ny] = term_D[:, -1].to(torch.complex128)
        coef[0, 1:ny] = term_A[0, :].to(torch.complex128); coef[nz, 1:ny] = term_B[-1, :].to(torch.complex128)
        
        hx_l = self.mt1dtm_solver(freq, dz, sig[:, 0])
        hx_r = self.mt1dtm_solver(freq, dz, sig[:, -1])
        steps = torch.linspace(0, 1, ny+1, device=self.device, dtype=torch.complex128)
        
        top_row_inner = (hx_l[0,0] + (hx_r[0,0] - hx_l[0,0]) * steps[1:-1]).view(1, -1)
        bot_row_inner = (hx_l[-1,0] + (hx_r[-1,0] - hx_l[-1,0]) * steps[1:-1]).view(1, -1)
        mid_zeros = torch.zeros((nz-1, ny-1), dtype=torch.complex128, device=self.device)
        
        col_inner = torch.cat([top_row_inner, mid_zeros, bot_row_inner], dim=0)
        hx1d = torch.cat([hx_l, col_inner, hx_r], dim=1)
        
        coef_base = hx1d * coef
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
        hx_flat = complex_solve_block(A, b_vec)
        hx_inner = hx_flat.view(ny-1, nz-1).T
        
        row_top = hx1d[0:1, :]
        row_bot = hx1d[nz:nz+1, :]
        col_left = hx1d[1:nz, 0:1]
        col_right = hx1d[1:nz, ny:ny+1]
        middle_layer = torch.cat([col_left, hx_inner, col_right], dim=1)
        hx_full = torch.cat([row_top, middle_layer, row_bot], dim=0)
        
        return hx_full

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
        off = 2.0 / dz_ext[1:-1]
        
        rng = torch.arange(nz, device=self.device)
        rng_off = torch.arange(nz-1, device=self.device)
        indices = torch.cat([
            torch.stack([rng, rng]),
            torch.stack([rng_off, rng_off+1]), torch.stack([rng_off+1, rng_off])
        ], dim=1)
        values = torch.cat([diag, off.to(torch.complex128), off.to(torch.complex128)])
        A = make_sparse_A(indices, values, (nz, nz), self.device)
        
        val0 = (-2.0 / dz_ext[0]).view(1, 1)
        zeros_rest = torch.zeros((nz-1, 1), dtype=torch.complex128, device=self.device)
        rhs = torch.cat([val0, zeros_rest], dim=0)
        
        res = complex_solve_block(A, rhs)
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
        off = 2.0 / (dz_ext[1:-1]*sig_ext[1:-1])
        
        rng = torch.arange(nz, device=self.device)
        rng_off = torch.arange(nz-1, device=self.device)
        indices = torch.cat([
            torch.stack([rng, rng]),
            torch.stack([rng_off, rng_off+1]), torch.stack([rng_off+1, rng_off])
        ], dim=1)
        values = torch.cat([diag, off.to(torch.complex128), off.to(torch.complex128)])
        A = make_sparse_A(indices, values, (nz, nz), self.device)
        
        val0 = (-2.0 / (dz_ext[0] * sig_ext[0])).view(1, 1)
        zeros_rest = torch.zeros((nz-1, 1), dtype=torch.complex128, device=self.device)
        rhs = torch.cat([val0, zeros_rest], dim=0)
        
        res = complex_solve_block(A, rhs)
        return torch.cat([torch.tensor([[1.0]], device=self.device, dtype=torch.complex128), res], dim=0)
