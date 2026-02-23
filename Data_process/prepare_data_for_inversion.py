# -*- coding: utf-8 -*-
# Customized for unformatted EDI data
import os
import glob
import numpy as np
import torch
from scipy.interpolate import interp1d
import re

# Try importing mtpy for strike; fallback to simple calc if unavailable
try:
    from mtpy.analysis.geometry import strike_angle
    HAS_MTPY = True
except ImportError:
    HAS_MTPY = False
    print("Warning: mtpy not found; skipping strike calc, using default.")

# ================= Configuration =================
edi_dir = r'./1china'             # Path to your EDI data folder
output_file = r'mtdata_china.pt'  # Output file name

# Frequency setup (ensure coverage of your data range)
# For long-period data (LMT) ~0.068 Hz, adjust range accordingly
target_freqs = np.logspace(np.log10(10000), np.log10(0.1), num=40)

DEFAULT_STRIKE = 45.0  # Prior value; auto-calculated from data if mtpy available
# ============================================

class CustomMT:
    """Simulated MT object for parsed data storage"""
    def __init__(self, lat, lon, freqs, z_array, z_err_array=None):
        self.lat = lat
        self.lon = lon
        self.frequency = np.array(freqs)
        self.Z = z_array         # Shape: (N, 2, 2)
        self.Z_err = z_err_array # Shape: (N, 2, 2)

    def rotate(self, angle):
        """Simple Z tensor rotation (degrees, clockwise)"""
        # Build rotation matrix R
        rad = np.deg2rad(angle)
        c = np.cos(rad)
        s = np.sin(rad)
        R = np.array([[c, s], [-s, c]])  # 2x2

        # Z_rot = R * Z * R.T
        # Loop over each frequency (or use broadcasting)
        new_Z = np.zeros_like(self.Z, dtype=complex)
        new_Err = np.zeros_like(self.Z_err, dtype=float) if self.Z_err is not None else None
        
        for i in range(len(self.frequency)):
            z_mat = self.Z[i]
            # Z' = R Z R^T
            new_Z[i] = R @ z_mat @ R.T
            
            # Error rotation is complex; approximate by keeping errors unchanged
            if new_Err is not None:
                # Strictly error propagation needs covariance; simplified:
                # rotated error magnitude is similar to original
                new_Err[i] = self.Z_err[i] 

        self.Z = new_Z
        if new_Err is not None:
            self.Z_err = new_Err

def parse_dms(dms_str):
    """Parse DMS format (e.g. 047:00:32.4000) for lat/lon"""
    # Strip whitespace
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
    Parser for WinGLink/Geosystem EDI format.
    Recognizes >ZXXR, >ZXXVAR, >ZXX.VAR variants.
    """
    with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
        lines = f.readlines()

    data_map = {} 
    lat, lon = 0.0, 0.0
    
    current_key = None
    current_values = []
    
    # 1. Line-by-line scan
    for line in lines:
        line = line.strip()
        if not line: continue
        
        # Parse header
        if line.startswith('LAT='):
            lat = parse_dms(line.split('=')[1])
            continue
        if line.startswith('LONG='):
            lon = parse_dms(line.split('=')[1])
            continue
            
        # Parse block header >KEY
        if line.startswith('>'):
            # Save previous block
            if current_key:
                data_map[current_key] = np.array(current_values)
            
            # Extract KEY (e.g. >TXVAR.EXP -> TXVAR, >ZXYR -> ZXYR)

            # Remove leading '>'
            temp = line[1:].strip()
            # Remove // comment
            temp = temp.split('//')[0].strip()
            # Remove space-separated params (e.g. ROT=TROT)
            temp = temp.split()[0].strip()
            
            # Normalize: remove dots, uppercase (ZXX.VAR -> ZXXVAR)
            current_key = temp.replace('.', '').upper() 
            current_values = []
        else:
            # Data line
            if current_key:
                try:
                    # WinGLink format may use odd separators; usually space
                    vals = [float(x) for x in line.split()]
                    current_values.extend(vals)
                except:
                    pass
    
    # Save last block
    if current_key:
        data_map[current_key] = np.array(current_values)

    # 2. Assemble data (WinGLink often uses >FREQ for frequency)
    if 'FREQ' not in data_map:
        # Try FREQUENCIES
        if 'FREQUENCIES' in data_map:
            freqs = data_map['FREQUENCIES']
        else:
            print(f"  [Warning] {os.path.basename(file_path)} no >FREQ block found, skipping.")
            return None  # Skip this file
    else:
        freqs = data_map['FREQ']

    n_freq = len(freqs)
    Z = np.zeros((n_freq, 2, 2), dtype=complex)
    Z_err = np.zeros((n_freq, 2, 2), dtype=float)
    has_err = False

    # Loop over XX, XY, YX, YY
    for idx_i, c1 in enumerate(['X', 'Y']):
        for idx_j, c2 in enumerate(['X', 'Y']):
            comp = f'{c1}{c2}' # XX, XY...
            
            # Read impedance
            # Common keys: ZXXR / ZXXI
            real_key = f'Z{comp}R'
            imag_key = f'Z{comp}I'
            
            if real_key in data_map and imag_key in data_map:
                r = data_map[real_key][:n_freq]
                i = data_map[imag_key][:n_freq]
                Z[:, idx_i, idx_j] = r + 1j * i
            else:
                pass  # No impedance data (bad/empty)

            # Read variance/error
            # WinGLink: ZXXVAR, ZXX.VAR, VARXX
            # Try ZXXVAR first
            var_key = f'Z{comp}VAR' 
            # Try VARXX (older format)
            var_key_2 = f'VAR{comp}'
            
            if var_key in data_map:
                var_val = data_map[var_key][:n_freq]
                Z_err[:, idx_i, idx_j] = np.sqrt(np.abs(var_val))  # sqrt(var) -> std
                has_err = True
            elif var_key_2 in data_map:
                var_val = data_map[var_key_2][:n_freq]
                Z_err[:, idx_i, idx_j] = np.sqrt(np.abs(var_val))
                has_err = True
    
    if not has_err:
        Z_err = None  # No error; use 10% floor later
        
    return CustomMT(lat, lon, freqs, Z, Z_err)

def project_stations(lats, lons, azimuth):
    """Project station coordinates"""
    lat0, lon0 = lats[0], lons[0]
    d_lat = (lats - lat0) * 111.0
    d_lon = (lons - lon0) * 111.0 * np.cos(np.deg2rad(lat0))
    rad = np.deg2rad(azimuth)
    return d_lon * np.sin(rad) + d_lat * np.cos(rad)

# ================= Main flow =================

# 1. Scan files
files = sorted(glob.glob(os.path.join(edi_dir, '*.edi')))
print(f"Found {len(files)} files.")

mt_objects = []
valid_files = []

# 2. Read all files
for f in files:
    try:
        mt = read_custom_edi(f)
        # Basic validity check
        if len(mt.frequency) > 0:
            mt_objects.append(mt)
            valid_files.append(f)
            # print(f"Read OK: {os.path.basename(f)} | Lat: {mt.lat:.4f}")
    except Exception as e:
        print(f"Failed to read {os.path.basename(f)}: {e}")

if not mt_objects:
    raise ValueError("No EDI files read successfully; check file format.")

# 3. Auto-compute strike (mtpy if available)
calculated_strike = DEFAULT_STRIKE
if HAS_MTPY:
    try:
        print("Computing global structure strike...")
        all_strikes = []
        for mt in mt_objects:
            # strike_angle requires Z array; thresholds: skew=5, eccentricity=0.1
            s = strike_angle(mt.Z, 5, 0.1)
            s = np.atleast_1d(s)
            s = s[~np.isnan(s)]
            all_strikes.extend(s % 180)
        
        if all_strikes:
            hist, bins = np.histogram(all_strikes, bins=36, range=(0, 180))
            peak_strike = (bins[np.argmax(hist)] + bins[np.argmax(hist)+1]) / 2.0
            calculated_strike = peak_strike
            print(f"Computed best strike: {calculated_strike:.2f} deg")
        else:
            print("Strike calculation returned empty; using default.")
    except Exception as e:
        print(f"Strike error ({e}); using default.")
else:
    print(f"Using default strike: {calculated_strike} deg")

# 4. Rotate, interpolate, pack data
data_list = []
error_list = []
station_locs = []

print("Interpolating and packing data...")
rotation_angle = calculated_strike + 90  # Rotate to profile (X perpendicular to strike)

for mt in mt_objects:
    # Rotate
    mt.rotate(rotation_angle)
    
    # Interpolate
    orig_freq = mt.frequency
    Z_orig = mt.Z
    
    # Build error (10% floor if missing)
    if mt.Z_err is None:
        Z_err_orig = np.abs(Z_orig) * 0.1
    else:
        Z_err_orig = mt.Z_err

    z_interp = np.zeros((len(target_freqs), 2, 2), dtype=complex)
    z_err_interp = np.zeros((len(target_freqs), 2, 2), dtype=float)
    
    # Note: freq may be ascending/descending; use log10(freq) for interp
    log_orig_f = np.log10(orig_freq)
    log_target_f = np.log10(target_freqs)
    
    # Avoid all-NaN: if target_freqs far outside data range, extrapolate may blow up
    
    for i in range(2):
        for j in range(2):
            # Z interpolation
            vals = Z_orig[:, i, j]
            fr = interp1d(log_orig_f, vals.real, bounds_error=False, fill_value="extrapolate")
            fi = interp1d(log_orig_f, vals.imag, bounds_error=False, fill_value="extrapolate")
            z_interp[:, i, j] = fr(log_target_f) + 1j * fi(log_target_f)
            
            # Error interpolation
            evals = Z_err_orig[:, i, j]
            fe = interp1d(log_orig_f, evals, bounds_error=False, fill_value="extrapolate")
            err_res = fe(log_target_f)
            err_res[err_res < 0] = 0  # Error must be non-negative
            z_err_interp[:, i, j] = err_res
            
    data_list.append(z_interp)
    error_list.append(z_err_interp)
    station_locs.append([mt.lat, mt.lon])

# Convert to tensors
Z_tensor = np.moveaxis(np.array(data_list), 0, 1) # (N_freq, N_sta, 2, 2)
Z_err_tensor = np.moveaxis(np.array(error_list), 0, 1)

# Project stations
lats = np.array([x[0] for x in station_locs])
lons = np.array([x[1] for x in station_locs])
stations_dist = project_stations(lats, lons, rotation_angle)
stations_dist -= stations_dist.min()

# Error propagation for Rho/Phs
rho_err_dict = {}
phs_err_dict = {}
modes = {'xy': (0, 1), 'yx': (1, 0)}

# Precompute constants
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

# Save
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
print(f"Done. Processed {len(valid_files)} stations.")
print(f"Frequency range: {target_freqs.max():.4f} Hz ~ {target_freqs.min():.5f} Hz")
print(f"Strike used: {calculated_strike:.2f} deg")
print(f"Data saved to: {output_file}")
