"""Export tensors for 2D inversion."""
from __future__ import annotations

from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple, Union

import numpy as np


import torch

class ExportMixin:
    def export_data_dict_for_2d_inversion(
        self,
        mt_objects: Optional[Sequence["PrepareData.CustomMT"]] = None,
        *,
        sort_by: str = "profile_pos_m",
        freq_rtol: float = 1e-6,  # 相对容差
        freq_atol: float = 1e-8,  # 绝对容差（新增/保留，专门对付低频浮点误差）
        device: Optional[str] = None,
        dtype=None,
        save_to_self: bool = True,
    ):
        """
        提取全局频率并集，使用 NaN 填充缺失数据，生成绝对规则的稠密张量。
        废弃了强制频率对齐，尊重数据的物理真实缺失。
        使用 rtol 和 atol 联合控制相近频点的合并。
        """
        try:
            import torch
        except Exception as e:
            raise ImportError("Requires torch") from e

        if mt_objects is None:
            mt_objects = self.mt_objects
        if not mt_objects:
            raise RuntimeError("mt_objects is empty; call load_mt_objects() first")

        need_derived = any(getattr(mt, "rho", None) is None or getattr(mt, "phs", None) is None for mt in mt_objects)
        if need_derived:
            self.compute_rho_phase_all(mt_objects)

        # 1. 排序台站
        mts = list(mt_objects)
        sort_key = sort_by.strip().lower() if sort_by else "none"
        if sort_key in {"profile_pos_m", "profile", "pos", "position"}:
            mts = sorted(mts, key=lambda m: float(getattr(m, "profile_pos_m", float("inf"))))
        elif sort_key in {"station_id", "id"}:
            mts = sorted(mts, key=lambda m: int(str(getattr(m, "station_id", 10**9))) if str(getattr(m, "station_id", "")).isdigit() else 10**9)

        stations = np.array([getattr(mt, "profile_pos_m", np.nan) for mt in mts], dtype=float)
        if not np.isfinite(stations).all():
            raise ValueError("Some stations have invalid profile_pos_m. Call assign_profile_pos_m() first.")

        # Station IDs aligned with `stations` ordering (after sorting).
        station_ids = []
        for j, mt in enumerate(mts):
            sid = getattr(mt, "station_id", None)
            if sid is None or str(sid).strip() == "":
                sid = f"S{j + 1}"
            station_ids.append(str(sid))

        # 2. 构建全局频率轴 (Union) 并使用 atol + rtol 联合去重
        all_freqs = np.concatenate([np.asarray(getattr(mt, "frequency"), dtype=float) for mt in mts])
        all_freqs = np.sort(all_freqs)
        
        # 核心合并逻辑：差值 > (atol + rtol * 当前频率) 才认为是不同的频点
        is_unique = np.append([True], np.diff(all_freqs) > (freq_atol + freq_rtol * all_freqs[:-1]))
        freqs_global = all_freqs[is_unique]
        
        n_freq = len(freqs_global)
        n_stn = len(mts)

        # 3. 初始化全 NaN 矩阵
        rhoxy = np.full((n_freq, n_stn), np.nan, dtype=float)
        phsxy = np.full((n_freq, n_stn), np.nan, dtype=float)
        rhoyx = np.full((n_freq, n_stn), np.nan, dtype=float)
        phsyx = np.full((n_freq, n_stn), np.nan, dtype=float)
        zxy = np.full((n_freq, n_stn), np.nan + 1j * np.nan, dtype=np.complex128)
        zyx = np.full((n_freq, n_stn), np.nan + 1j * np.nan, dtype=np.complex128)
        zxy_err = np.full((n_freq, n_stn), np.nan, dtype=float)
        zyx_err = np.full((n_freq, n_stn), np.nan, dtype=float)
        any_has_err = False

        # 4. 对号入座
        for j, mt in enumerate(mts):
            f_stn = np.asarray(getattr(mt, "frequency"), dtype=float)
            idx_global = np.array([np.abs(freqs_global - f).argmin() for f in f_stn])
            
            rho = np.asarray(getattr(mt, "rho", None), dtype=float)
            phs = np.asarray(getattr(mt, "phs", None), dtype=float)
            z_arr = np.asarray(getattr(mt, "Z", None), dtype=np.complex128)
            
            if rho is not None and rho.size > 0:
                rhoxy[idx_global, j] = rho[:, 0, 1]
                rhoyx[idx_global, j] = rho[:, 1, 0]
            if phs is not None and phs.size > 0:
                phsxy[idx_global, j] = phs[:, 0, 1]
                phsyx[idx_global, j] = phs[:, 1, 0]
            if z_arr is not None and z_arr.size > 0:
                zxy[idx_global, j] = z_arr[:, 0, 1]
                zyx[idx_global, j] = z_arr[:, 1, 0]
                
            z_err = getattr(mt, "Z_err", None)
            if z_err is not None:
                any_has_err = True
                z_err = np.asarray(z_err, dtype=float)
                zxy_err[idx_global, j] = z_err[:, 0, 1]
                zyx_err[idx_global, j] = z_err[:, 1, 0]

        # 5. 转换为 Torch Tensor
        if dtype is None:
            dtype = torch.float64
        freqs_t = torch.as_tensor(freqs_global, dtype=dtype, device=device)
        stations_t = torch.as_tensor(stations, dtype=dtype, device=device)
        
        obs_data = {
            "rhoxy": torch.as_tensor(rhoxy, dtype=dtype, device=device),
            "phsxy": torch.as_tensor(phsxy, dtype=dtype, device=device),
            "rhoyx": torch.as_tensor(rhoyx, dtype=dtype, device=device),
            "phsyx": torch.as_tensor(phsyx, dtype=dtype, device=device),
        }

        # Impedance & std-dev for error propagation on inverter side (require Zxy, Zyx + delta for calculate_data_errors_2d)
        data_std = {}
        torch_complex_dtype = self._torch_complex_dtype_from_float(dtype)
        data_std["Zxy"] = torch.as_tensor(zxy, dtype=torch_complex_dtype, device=device)
        data_std["Zyx"] = torch.as_tensor(zyx, dtype=torch_complex_dtype, device=device)
        if any_has_err:
            # Use EDI Z_err where available; in union+NaN mode may contain NaN at missing points (inverter masks)
            data_std["delta_zxy_real"] = torch.as_tensor(zxy_err, dtype=dtype, device=device)
            data_std["delta_zxy_imag"] = torch.as_tensor(zxy_err, dtype=dtype, device=device)
            data_std["delta_zyx_real"] = torch.as_tensor(zyx_err, dtype=dtype, device=device)
            data_std["delta_zyx_imag"] = torch.as_tensor(zyx_err, dtype=dtype, device=device)
        else:
            # No Z_err in EDI: use default 1% relative error for propagation (inverter noise_floor can further clip)
            zxy_abs = np.abs(zxy)
            zyx_abs = np.abs(zyx)
            delta_default = 0.01
            data_std["delta_zxy_real"] = torch.as_tensor(
                np.where(np.isfinite(zxy_abs), delta_default * zxy_abs, np.nan), dtype=dtype, device=device
            )
            data_std["delta_zxy_imag"] = data_std["delta_zxy_real"]
            data_std["delta_zyx_real"] = torch.as_tensor(
                np.where(np.isfinite(zyx_abs), delta_default * zyx_abs, np.nan), dtype=dtype, device=device
            )
            data_std["delta_zyx_imag"] = data_std["delta_zyx_real"]

        data_dict = {"obs_data": obs_data, "data_std": data_std, "station_ids": station_ids}

        if save_to_self:
            self.inv_freqs_torch = freqs_t
            self.inv_stations_torch = stations_t
            self.inv_data_dict = data_dict
            self.inv_station_ids = station_ids

        print(f"[PrepareData] Exported Tensor shape: Frequencies={n_freq}, Stations={n_stn}")
        return freqs_t, stations_t, data_dict

    def _sorted_mts_for_export(
        self,
        mt_objects: Optional[Sequence["PrepareData.CustomMT"]] = None,
        sort_by: str = "profile_pos_m",
    ) -> List["PrepareData.CustomMT"]:
        """Same station ordering as :meth:`export_data_dict_for_2d_inversion`."""
        if mt_objects is None:
            mt_objects = self.mt_objects
        mts = list(mt_objects)
        sort_key = sort_by.strip().lower() if sort_by else "none"
        if sort_key in {"profile_pos_m", "profile", "pos", "position"}:
            mts.sort(key=lambda m: float(getattr(m, "profile_pos_m", float("inf"))))
        elif sort_key in {"station_id", "id"}:
            mts.sort(
                key=lambda m: int(str(getattr(m, "station_id", 10**9)))
                if str(getattr(m, "station_id", "")).isdigit()
                else 10**9
            )
        return mts

    def get_station_lon_lat_sorted(
        self,
        sort_by: str = "profile_pos_m",
        mt_objects: Optional[Sequence["PrepareData.CustomMT"]] = None,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Profile positions and geographic coordinates per station (after ``sort_by``).

        Returns
        -------
        stations_m, lon, lat : ndarray
            Shapes (n_stn,). Same order as columns from ``export_data_dict_for_2d_inversion``.
        """
        mts = self._sorted_mts_for_export(mt_objects=mt_objects, sort_by=sort_by)
        if not mts:
            raise RuntimeError("mt_objects is empty")
        stations = np.array([float(getattr(mt, "profile_pos_m", np.nan)) for mt in mts], dtype=float)
        if not np.isfinite(stations).all():
            raise ValueError("Invalid profile_pos_m; call assign_profile_pos_m() first.")
        lon = np.array([float(getattr(mt, "lon", np.nan)) for mt in mts], dtype=float)
        lat = np.array([float(getattr(mt, "lat", np.nan)) for mt in mts], dtype=float)
        if not (np.isfinite(lon).all() and np.isfinite(lat).all()):
            raise ValueError("Some stations have invalid lat/lon (check EDI headers).")
        return stations, lon, lat

    @staticmethod
    def _yn_edges_to_numpy(yn: Union[np.ndarray, Any]) -> np.ndarray:
        try:
            import torch

            if torch.is_tensor(yn):
                return yn.detach().cpu().numpy().astype(np.float64).reshape(-1)
        except ImportError:
            pass
        return np.asarray(yn, dtype=np.float64).reshape(-1)

    def export_mt_object_to_txt(
        mt_object: Any,
        txt_path: Union[str, Path],
        *,
        sort_by_frequency: bool = True,
        include_derived: bool = True,
        include_errors: bool = True,
        float_format: str = ".6e",
        encoding: str = "utf-8",
        overwrite: bool = True,
    ) -> Path:
        """把单个 MT 对象导出为 txt（便于检查/复现）。

        兼容两类对象：
        - 本模块的 `PrepareData.CustomMT`（推荐）
        - 外部库对象（如 mtpy 风格）：支持 `obj.Z.z` / `obj.Z.z_err` 取值

        输出为“带注释头 + 表格数据”的纯文本。表格列会尽量覆盖：
        - freq_hz, period_s
        - Z(2x2) 的实部/虚部（按 xx,xy,yx,yy 顺序展开）
        - 可选：Z_err(2x2)
        - 可选：rho/phs 及其误差、噪声归一化 std-dev
        """

        def _maybe_get(obj: Any, names: Sequence[str]) -> Any:
            for n in names:
                if hasattr(obj, n):
                    return getattr(obj, n)
            return None

        def _as_array(x: Any, *, dtype=None) -> Optional[np.ndarray]:
            if x is None:
                return None
            try:
                return np.asarray(x, dtype=dtype)
            except Exception:
                return None

        txt_path = Path(txt_path)
        if txt_path.exists() and (not overwrite):
            raise FileExistsError(f"Target txt already exists: {txt_path}")
        txt_path.parent.mkdir(parents=True, exist_ok=True)

        # -------- meta --------
        station_id = _maybe_get(mt_object, ["station_id", "station", "name", "id"])
        lat = _maybe_get(mt_object, ["lat", "latitude"])
        lon = _maybe_get(mt_object, ["lon", "long", "longitude"])
        profile_pos_m = _maybe_get(mt_object, ["profile_pos_m", "profile_pos", "profile_position_m"])

        # -------- frequency --------
        freqs = _maybe_get(mt_object, ["frequency", "freq", "freqs", "frequencies"])
        if freqs is None:
            # mtpy sometimes stores period; try to invert
            period = _maybe_get(mt_object, ["period", "periods"])
            period = _as_array(period, dtype=float)
            if period is not None and period.ndim == 1 and period.size > 0:
                freqs = 1.0 / period
        freqs = _as_array(freqs, dtype=float)

        # -------- impedance tensor + err --------
        Z_raw = _maybe_get(mt_object, ["Z", "z"])
        Z_arr: Optional[np.ndarray] = None
        Z_err_arr: Optional[np.ndarray] = None
        if isinstance(Z_raw, np.ndarray) or np.isscalar(Z_raw):
            Z_arr = _as_array(Z_raw)
        else:
            # mtpy: mt.Z.z, mt.Z.z_err
            Z_arr = _as_array(_maybe_get(Z_raw, ["z", "Z", "impedance"]))
            Z_err_arr = _as_array(_maybe_get(Z_raw, ["z_err", "Z_err", "zerr", "error"]))

        if Z_arr is None:
            Z_arr = _as_array(_maybe_get(mt_object, ["Z", "z_array", "z_tensor"]))
        if Z_err_arr is None:
            Z_err_arr = _as_array(_maybe_get(mt_object, ["Z_err", "z_err", "ZERR"]))

        if freqs is None or freqs.ndim != 1 or freqs.size == 0:
            raise ValueError("mt_object has no valid frequency axis (expected 1D non-empty `frequency`)")
        if Z_arr is None:
            raise ValueError("mt_object has no valid impedance tensor (expected `Z` with shape (n,2,2))")
        Z_arr = np.asarray(Z_arr)
        if Z_arr.ndim != 3 or Z_arr.shape[1:] != (2, 2):
            raise ValueError(f"Z must have shape (n,2,2), got {Z_arr.shape}")
        if Z_arr.shape[0] != freqs.shape[0]:
            raise ValueError(f"frequency length {freqs.shape[0]} doesn't match Z length {Z_arr.shape[0]}")

        if Z_err_arr is not None:
            Z_err_arr = np.asarray(Z_err_arr, dtype=float)
            if Z_err_arr.shape != Z_arr.shape:
                Z_err_arr = None

        # -------- derived (rho/phs + errors/noise std) --------
        rho = _as_array(_maybe_get(mt_object, ["rho", "apparent_resistivity"]))
        phs = _as_array(_maybe_get(mt_object, ["phs", "phase"]))
        rho_err = _as_array(_maybe_get(mt_object, ["rho_err", "rho_error"]))
        phs_err = _as_array(_maybe_get(mt_object, ["phs_err", "phs_error"]))
        rho_noise_std_log10 = _as_array(_maybe_get(mt_object, ["rho_noise_std_log10", "rho_noise_std"]))
        phs_noise_std_norm = _as_array(_maybe_get(mt_object, ["phs_noise_std_norm", "phs_noise_std"]))

        rho_ok = isinstance(rho, np.ndarray) and rho.shape == Z_arr.shape
        phs_ok = isinstance(phs, np.ndarray) and phs.shape == Z_arr.shape
        rho_err_ok = isinstance(rho_err, np.ndarray) and rho_err.shape == Z_arr.shape
        phs_err_ok = isinstance(phs_err, np.ndarray) and phs_err.shape == Z_arr.shape
        rn_ok = isinstance(rho_noise_std_log10, np.ndarray) and rho_noise_std_log10.shape == Z_arr.shape
        pn_ok = isinstance(phs_noise_std_norm, np.ndarray) and phs_noise_std_norm.shape == Z_arr.shape

        if include_derived and ((not rho_ok) or (not phs_ok) or (include_errors and ((not rho_err_ok) or (not phs_err_ok) or (not rn_ok) or (not pn_ok)))):
            rho_calc, phs_calc, rho_err_calc, phs_err_calc, rn_calc, pn_calc = PrepareData.impedance_to_rho_phase(
                freqs_hz=freqs,
                Z=Z_arr,
                Z_err=Z_err_arr if include_errors else None,
            )
            if (not rho_ok) and rho_calc is not None:
                rho = rho_calc
            if (not phs_ok) and phs_calc is not None:
                phs = phs_calc
            if include_errors and (not rho_err_ok) and (rho_err_calc is not None):
                rho_err = rho_err_calc
            if include_errors and (not phs_err_ok) and (phs_err_calc is not None):
                phs_err = phs_err_calc
            if include_errors and (not rn_ok) and (rn_calc is not None):
                rho_noise_std_log10 = rn_calc
            if include_errors and (not pn_ok) and (pn_calc is not None):
                phs_noise_std_norm = pn_calc

        # -------- optional sort by frequency --------
        order = np.arange(freqs.size)
        if sort_by_frequency:
            order = np.argsort(freqs)
            freqs = freqs[order]
            Z_arr = Z_arr[order]
            if Z_err_arr is not None:
                Z_err_arr = Z_err_arr[order]
            if isinstance(rho, np.ndarray) and rho.shape[:1] == (order.size,):
                rho = rho[order]
            if isinstance(phs, np.ndarray) and phs.shape[:1] == (order.size,):
                phs = phs[order]
            if isinstance(rho_err, np.ndarray) and rho_err.shape[:1] == (order.size,):
                rho_err = rho_err[order]
            if isinstance(phs_err, np.ndarray) and phs_err.shape[:1] == (order.size,):
                phs_err = phs_err[order]
            if isinstance(rho_noise_std_log10, np.ndarray) and rho_noise_std_log10.shape[:1] == (order.size,):
                rho_noise_std_log10 = rho_noise_std_log10[order]
            if isinstance(phs_noise_std_norm, np.ndarray) and phs_noise_std_norm.shape[:1] == (order.size,):
                phs_noise_std_norm = phs_noise_std_norm[order]

        # -------- write --------
        def _fmt(x: float) -> str:
            if x is None:
                return "nan"
            try:
                xf = float(x)
            except Exception:
                return "nan"
            if not np.isfinite(xf):
                return "nan"
            return format(xf, float_format)

        def _flatten_2x2(M: np.ndarray) -> tuple[float, float, float, float]:
            return (float(M[0, 0]), float(M[0, 1]), float(M[1, 0]), float(M[1, 1]))

        cols: list[str] = [
            "freq_hz",
            "period_s",
            "Zxx_real",
            "Zxx_imag",
            "Zxy_real",
            "Zxy_imag",
            "Zyx_real",
            "Zyx_imag",
            "Zyy_real",
            "Zyy_imag",
        ]
        if include_errors and (Z_err_arr is not None):
            cols += ["Zxx_err", "Zxy_err", "Zyx_err", "Zyy_err"]
        if include_derived and isinstance(rho, np.ndarray) and rho.shape == Z_arr.shape:
            cols += ["rho_xx", "rho_xy", "rho_yx", "rho_yy"]
        if include_derived and isinstance(phs, np.ndarray) and phs.shape == Z_arr.shape:
            cols += ["phs_xx_deg", "phs_xy_deg", "phs_yx_deg", "phs_yy_deg"]
        if include_errors and isinstance(rho_err, np.ndarray) and rho_err.shape == Z_arr.shape:
            cols += ["rho_err_xx", "rho_err_xy", "rho_err_yx", "rho_err_yy"]
        if include_errors and isinstance(phs_err, np.ndarray) and phs_err.shape == Z_arr.shape:
            cols += ["phs_err_xx_deg", "phs_err_xy_deg", "phs_err_yx_deg", "phs_err_yy_deg"]
        if include_errors and isinstance(rho_noise_std_log10, np.ndarray) and rho_noise_std_log10.shape == Z_arr.shape:
            cols += [
                "rho_noise_std_log10_xx",
                "rho_noise_std_log10_xy",
                "rho_noise_std_log10_yx",
                "rho_noise_std_log10_yy",
            ]
        if include_errors and isinstance(phs_noise_std_norm, np.ndarray) and phs_noise_std_norm.shape == Z_arr.shape:
            cols += [
                "phs_noise_std_norm_xx",
                "phs_noise_std_norm_xy",
                "phs_noise_std_norm_yx",
                "phs_noise_std_norm_yy",
            ]

        with open(txt_path, "w", encoding=encoding, newline="\n") as f:
            f.write("# export_mt_object_to_txt\n")
            if station_id is not None:
                f.write(f"# station_id: {station_id}\n")
            if lat is not None:
                f.write(f"# lat: {lat}\n")
            if lon is not None:
                f.write(f"# lon: {lon}\n")
            if profile_pos_m is not None:
                f.write(f"# profile_pos_m: {profile_pos_m}\n")
            f.write(f"# n_freq: {int(freqs.size)}\n")
            f.write("# columns: " + "\t".join(cols) + "\n")
            f.write("\t".join(cols) + "\n")

            for i in range(freqs.size):
                freq = float(freqs[i])
                period_s = 1.0 / freq if np.isfinite(freq) and freq != 0 else float("nan")
                z = Z_arr[i]
                zxx, zxy, zyx, zyy = z[0, 0], z[0, 1], z[1, 0], z[1, 1]
                row: list[str] = [
                    _fmt(freq),
                    _fmt(period_s),
                    _fmt(zxx.real),
                    _fmt(zxx.imag),
                    _fmt(zxy.real),
                    _fmt(zxy.imag),
                    _fmt(zyx.real),
                    _fmt(zyx.imag),
                    _fmt(zyy.real),
                    _fmt(zyy.imag),
                ]

                if include_errors and (Z_err_arr is not None):
                    ze = Z_err_arr[i]
                    exx, exy, eyx, eyy = _flatten_2x2(ze)
                    row += [_fmt(exx), _fmt(exy), _fmt(eyx), _fmt(eyy)]

                if include_derived and isinstance(rho, np.ndarray) and rho.shape == Z_arr.shape:
                    r = rho[i]
                    rxx, rxy, ryx, ryy = _flatten_2x2(r)
                    row += [_fmt(rxx), _fmt(rxy), _fmt(ryx), _fmt(ryy)]

                if include_derived and isinstance(phs, np.ndarray) and phs.shape == Z_arr.shape:
                    p = phs[i]
                    pxx, pxy, pyx, pyy = _flatten_2x2(p)
                    row += [_fmt(pxx), _fmt(pxy), _fmt(pyx), _fmt(pyy)]

                if include_errors and isinstance(rho_err, np.ndarray) and rho_err.shape == Z_arr.shape:
                    re = rho_err[i]
                    rexx, rexy, reyx, reyy = _flatten_2x2(re)
                    row += [_fmt(rexx), _fmt(rexy), _fmt(reyx), _fmt(reyy)]

                if include_errors and isinstance(phs_err, np.ndarray) and phs_err.shape == Z_arr.shape:
                    pe = phs_err[i]
                    pexx, pexy, peyx, peyy = _flatten_2x2(pe)
                    row += [_fmt(pexx), _fmt(pexy), _fmt(peyx), _fmt(peyy)]

                if include_errors and isinstance(rho_noise_std_log10, np.ndarray) and rho_noise_std_log10.shape == Z_arr.shape:
                    rn = rho_noise_std_log10[i]
                    rnxx, rnxy, rnyx, rnyy = _flatten_2x2(rn)
                    row += [_fmt(rnxx), _fmt(rnxy), _fmt(rnyx), _fmt(rnyy)]

                if include_errors and isinstance(phs_noise_std_norm, np.ndarray) and phs_noise_std_norm.shape == Z_arr.shape:
                    pn = phs_noise_std_norm[i]
                    pnxx, pnxy, pnyx, pnyy = _flatten_2x2(pn)
                    row += [_fmt(pnxx), _fmt(pnxy), _fmt(pnyx), _fmt(pnyy)]

                f.write("\t".join(row) + "\n")

        return txt_path
