# -*- coding: utf-8 -*-
#customized for unformatted edi data
import os
import glob
import numpy as np
import torch
from scipy.interpolate import interp1d
import re

# 尝试导入 mtpy 用于计算走向，如果失败则使用简易计算
try:
    from mtpy.analysis.geometry import strike_angle
    HAS_MTPY = True
except ImportError:
    HAS_MTPY = False
    print("Warning: 未检测到 mtpy，将跳过自动走向计算，使用默认值。")

# ================= 配置区域 =================
edi_dir = r'./1china'             # 你的数据文件夹路径
output_file = r'mtdata_china.pt'  # 输出结果文件名

# 频率设置 (请确保覆盖你数据的频率范围)
# 根据你提供的文件示例：最高频约 0.068 (14s)，这看起来像是长周期数据(LMT)？
# 如果你的数据是从 0.068Hz 到更低，请调整下面的范围！
# 这里我暂时设置为 100Hz 到 0.0001Hz 的宽范围，脚本会自动截取。
target_freqs = np.logspace(np.log10(10000), np.log10(0.1), num=40) 

DEFAULT_STRIKE = 45.0 #这里取一个先验值，后续会根据数据自动计算
# ===========================================

class CustomMT:
    """模拟 MT 对象，用于存储解析后的数据"""
    def __init__(self, lat, lon, freqs, z_array, z_err_array=None):
        self.lat = lat
        self.lon = lon
        self.frequency = np.array(freqs)
        self.Z = z_array         # Shape: (N, 2, 2)
        self.Z_err = z_err_array # Shape: (N, 2, 2)

    def rotate(self, angle):
        """简单的 Z 张量旋转 (角度为度，顺时针)"""
        # 构造旋转矩阵 R
        rad = np.deg2rad(angle)
        c = np.cos(rad)
        s = np.sin(rad)
        R = np.array([[c, s], [-s, c]]) # 2x2
        
        # Z_rot = R * Z * R.T
        # 对每个频点循环操作 (或者利用广播机制)
        new_Z = np.zeros_like(self.Z, dtype=complex)
        new_Err = np.zeros_like(self.Z_err, dtype=float) if self.Z_err is not None else None
        
        for i in range(len(self.frequency)):
            z_mat = self.Z[i]
            # Z' = R Z R^T
            new_Z[i] = R @ z_mat @ R.T
            
            # 误差旋转比较复杂，简单近似为不旋转误差，或者假设各向同性
            if new_Err is not None:
                # 严格来说误差传播需要协方差矩阵，这里简化处理：
                # 旋转后的误差通常与原误差量级相当
                new_Err[i] = self.Z_err[i] 

        self.Z = new_Z
        if new_Err is not None:
            self.Z_err = new_Err

def parse_dms(dms_str):
    """解析 047:00:32.4000 格式的经纬度"""
    # 移除可能的空白
    dms_str = dms_str.strip()
    try:
        parts = dms_str.split(':')
        d = float(parts[0])
        m = float(parts[1])
        s = float(parts[2])
        
        sign = 1
        if d < 0 or dms_str.startswith('-'):
            sign = -1
            d = abs(d)
        
        return sign * (d + m/60.0 + s/3600.0)
    except:
        return 0.0

def read_custom_edi(file_path):
    """
    升级版解析器：专门适配 WinGLink/Geosystem 格式
    能识别 >ZXXR, >ZXXVAR, >ZXX.VAR 等多种变体
    """
    with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
        lines = f.readlines()

    data_map = {} 
    lat, lon = 0.0, 0.0
    
    current_key = None
    current_values = []
    
    # 1. 逐行扫描
    for line in lines:
        line = line.strip()
        if not line: continue
        
        # 解析头信息
        if line.startswith('LAT='):
            lat = parse_dms(line.split('=')[1])
            continue
        if line.startswith('LONG='):
            lon = parse_dms(line.split('=')[1])
            continue
            
        # 解析数据块头 >KEY
        if line.startswith('>'):
            # 保存上一个块
            if current_key:
                data_map[current_key] = np.array(current_values)
            
            # 提取 KEY
            # 例子: >TXVAR.EXP ROT=TROT //41  -> KEY应为 TXVAR
            # 例子: >ZXYR //41 -> KEY应为 ZXYR
            
            # 去掉开头的 '>'
            temp = line[1:].strip()
            # 去掉 // 后的注释
            temp = temp.split('//')[0].strip()
            # 去掉空格后的参数 (如 ROT=TROT)
            temp = temp.split()[0].strip()
            
            # 统一格式：去掉点，转大写
            # ZXX.VAR -> ZXXVAR
            # ZXXR -> ZXXR
            current_key = temp.replace('.', '').upper() 
            current_values = []
        else:
            # 数据行
            if current_key:
                try:
                    # WinGLink 格式有时会在数字间用莫名其妙的字符，但通常是空格
                    vals = [float(x) for x in line.split()]
                    current_values.extend(vals)
                except:
                    pass
    
    # 保存最后一个块
    if current_key:
        data_map[current_key] = np.array(current_values)
        
    # 2. 组装数据
    # WinGLink 经常用 >FREQ 作为频率
    if 'FREQ' not in data_map:
        # 尝试找找有没有叫 FREQUENCIES 的
        if 'FREQUENCIES' in data_map:
            freqs = data_map['FREQUENCIES']
        else:
            print(f"  [警告] {os.path.basename(file_path)} 未找到频率块 >FREQ，跳过。")
            return None # 返回 None 让主程序跳过
    else:
        freqs = data_map['FREQ']

    n_freq = len(freqs)
    Z = np.zeros((n_freq, 2, 2), dtype=complex)
    Z_err = np.zeros((n_freq, 2, 2), dtype=float)
    has_err = False

    # 遍历四个分量 XX, XY, YX, YY
    for idx_i, c1 in enumerate(['X', 'Y']):
        for idx_j, c2 in enumerate(['X', 'Y']):
            comp = f'{c1}{c2}' # XX, XY...
            
            # --- 读取阻抗 (Impedance) ---
            # 常见的 Key: ZXXR / ZXXI
            real_key = f'Z{comp}R'
            imag_key = f'Z{comp}I'
            
            if real_key in data_map and imag_key in data_map:
                r = data_map[real_key][:n_freq]
                i = data_map[imag_key][:n_freq]
                Z[:, idx_i, idx_j] = r + 1j * i
            else:
                # 没找到阻抗数据，可能是坏点或空文件
                pass

            # --- 读取误差 (Variance/Error) ---
            # WinGLink 常见 Key: ZXXVAR, ZXX.VAR, VARXX
            # 我们的 current_key 处理已经把点去掉了，所以找 ZXXVAR 即可
            
            # 尝试1: ZXXVAR (最常见)
            var_key = f'Z{comp}VAR' 
            # 尝试2: VARXX (某些旧格式)
            var_key_2 = f'VAR{comp}'
            
            if var_key in data_map:
                var_val = data_map[var_key][:n_freq]
                Z_err[:, idx_i, idx_j] = np.sqrt(np.abs(var_val)) # 方差开根号 -> 标准差
                has_err = True
            elif var_key_2 in data_map:
                var_val = data_map[var_key_2][:n_freq]
                Z_err[:, idx_i, idx_j] = np.sqrt(np.abs(var_val))
                has_err = True
    
    if not has_err:
        Z_err = None # 标记为无误差，后续会用 10% 填充
        
    return CustomMT(lat, lon, freqs, Z, Z_err)

def project_stations(lats, lons, azimuth):
    """投影坐标"""
    lat0, lon0 = lats[0], lons[0]
    d_lat = (lats - lat0) * 111.0
    d_lon = (lons - lon0) * 111.0 * np.cos(np.deg2rad(lat0))
    rad = np.deg2rad(azimuth)
    return d_lon * np.sin(rad) + d_lat * np.cos(rad)

# ================= 主流程 =================

# 1. 扫描文件
files = sorted(glob.glob(os.path.join(edi_dir, '*.edi')))
print(f"找到 {len(files)} 个文件。")

mt_objects = []
valid_files = []

# 2. 读取所有文件
for f in files:
    try:
        mt = read_custom_edi(f)
        # 简单检查数据有效性
        if len(mt.frequency) > 0:
            mt_objects.append(mt)
            valid_files.append(f)
            # print(f"读取成功: {os.path.basename(f)} | Lat: {mt.lat:.4f}")
    except Exception as e:
        print(f"读取失败 {os.path.basename(f)}: {e}")

if not mt_objects:
    raise ValueError("没有成功读取任何 EDI 文件，请检查文件内容格式。")

# 3. 自动计算走向 (使用 mtpy 逻辑如果可用)
calculated_strike = DEFAULT_STRIKE
if HAS_MTPY:
    try:
        print("正在计算全局构造走向...")
        all_strikes = []
        for mt in mt_objects:
            # mtpy.analysis.geometry.strike_angle 需要 Z 数组
            # 阈值: skew=5, eccentricity=0.1
            s = strike_angle(mt.Z, 5, 0.1)
            s = np.atleast_1d(s)
            s = s[~np.isnan(s)]
            all_strikes.extend(s % 180)
        
        if all_strikes:
            hist, bins = np.histogram(all_strikes, bins=36, range=(0, 180))
            peak_strike = (bins[np.argmax(hist)] + bins[np.argmax(hist)+1]) / 2.0
            calculated_strike = peak_strike
            print(f"统计计算出的最佳走向: {calculated_strike:.2f} 度")
        else:
            print("走向计算返回空值，使用默认设置。")
    except Exception as e:
        print(f"走向计算出错 ({e})，使用默认设置。")
else:
    print(f"使用默认走向: {calculated_strike} 度")

# 4. 数据旋转、插值与打包
data_list = []
error_list = []
station_locs = []

print("正在插值与打包数据...")
rotation_angle = calculated_strike + 90 # 旋转到剖面方向 (X轴垂直走向)

for mt in mt_objects:
    # 旋转
    mt.rotate(rotation_angle)
    
    # 插值
    orig_freq = mt.frequency
    Z_orig = mt.Z
    
    # 构造误差 (如果缺失，给 10% floor)
    if mt.Z_err is None:
        Z_err_orig = np.abs(Z_orig) * 0.1
    else:
        Z_err_orig = mt.Z_err

    z_interp = np.zeros((len(target_freqs), 2, 2), dtype=complex)
    z_err_interp = np.zeros((len(target_freqs), 2, 2), dtype=float)
    
    # 注意：你的数据频率可能是降序或升序，插值前最好排序，但 interp1d 通常能处理
    # 最好传入 log10(freq)
    log_orig_f = np.log10(orig_freq)
    log_target_f = np.log10(target_freqs)
    
    # 简单的范围检查，避免插值全为 NaN
    # 如果目标频率超出数据范围太远，fill_value="extrapolate" 可能会导致数值爆炸
    # 建议反演时 target_freqs 不要超出实测范围太多
    
    for i in range(2):
        for j in range(2):
            # Z 插值
            vals = Z_orig[:, i, j]
            fr = interp1d(log_orig_f, vals.real, bounds_error=False, fill_value="extrapolate")
            fi = interp1d(log_orig_f, vals.imag, bounds_error=False, fill_value="extrapolate")
            z_interp[:, i, j] = fr(log_target_f) + 1j * fi(log_target_f)
            
            # Err 插值
            evals = Z_err_orig[:, i, j]
            fe = interp1d(log_orig_f, evals, bounds_error=False, fill_value="extrapolate")
            err_res = fe(log_target_f)
            err_res[err_res < 0] = 0 # 误差不能为负
            z_err_interp[:, i, j] = err_res
            
    data_list.append(z_interp)
    error_list.append(z_err_interp)
    station_locs.append([mt.lat, mt.lon])

# 转换为 Tensor
Z_tensor = np.moveaxis(np.array(data_list), 0, 1) # (N_freq, N_sta, 2, 2)
Z_err_tensor = np.moveaxis(np.array(error_list), 0, 1)

# 投影测点
lats = np.array([x[0] for x in station_locs])
lons = np.array([x[1] for x in station_locs])
stations_dist = project_stations(lats, lons, rotation_angle)
stations_dist -= stations_dist.min()

# 误差传播计算 Rho/Phs 误差
rho_err_dict = {}
phs_err_dict = {}
modes = {'xy': (0, 1), 'yx': (1, 0)}

# 预计算常数
omega = 2 * np.pi * target_freqs[:, None]
mu0 = 4 * np.pi * 1e-7

for mode, (i, j) in modes.items():
    Z_comp = Z_tensor[:, :, i, j]
    Z_err_comp = Z_err_tensor[:, :, i, j]
    abs_Z = np.abs(Z_comp)
    mask = abs_Z > 1e-12
    
    # Log Rho Std
    sigma_log_rho = np.zeros_like(Z_err_comp)
    sigma_log_rho[mask] = (2.0 * Z_err_comp[mask]) / (abs_Z[mask] * np.log(10))
    
    # Phase Std (Norm)
    sigma_phs_norm = np.zeros_like(Z_err_comp)
    sigma_phs_norm[mask] = (Z_err_comp[mask] / abs_Z[mask]) * (180 / np.pi) / 90.0
    
    rho_err_dict[f'rho{mode}'] = torch.tensor(sigma_log_rho)
    phs_err_dict[f'phs{mode}'] = torch.tensor(sigma_phs_norm)

# 保存
output_data = {
    "freqs": torch.tensor(target_freqs, dtype=torch.float64),
    "stations": torch.tensor(stations_dist, dtype=torch.float64),
    "Z_obs": torch.tensor(Z_tensor, dtype=torch.complex128),
    "Z_err": torch.tensor(Z_err_tensor, dtype=torch.float64),
    "noise_std_dict": {**rho_err_dict, **phs_err_dict},
    "calculated_strike": calculated_strike,
    "station_lats": lats,
    "station_lons": lons
}

torch.save(output_data, output_file)
print("-" * 30)
print(f"处理成功！共处理 {len(valid_files)} 个测点。")
print(f"频率范围: {target_freqs.max():.4f} Hz ~ {target_freqs.min():.5f} Hz")
print(f"计算/使用的走向: {calculated_strike:.2f} 度")
print(f"数据已保存至: {output_file}")