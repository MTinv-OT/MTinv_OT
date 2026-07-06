"""EDI parsing and impedance transforms."""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple, Union

import numpy as np



class EdiMixin:
    @staticmethod
    def parse_dms(dms_str: str) -> float:
        """DMS to decimal."""

        dms_str = str(dms_str).strip()
        parts = dms_str.split(":")
        d = float(parts[0])
        m = float(parts[1])
        s = float(parts[2])
        sign = 1
        if d < 0:
            sign = -1
            d = abs(d)
        return sign * (d + m / 60 + s / 3600)

    @staticmethod
    def _module_dir() -> Path:
        return Path(__file__).resolve().parent.parent

    # -------------------------- EDI parser --------------------------

    @classmethod
    def find_edi_files(cls, edi_dir: str) -> Tuple[str, list[str]]:
        """Resolve EDI directory robustly (works even if cwd isn't the notebook folder)."""

        base = cls._module_dir()
        candidates = [
            Path(edi_dir),
            base / edi_dir,
            Path("AKBST-AMT-L08"),
            base / "AKBST-AMT-L08",
            Path("./AKBST-AMT-L08"),
            Path("edi"),
            Path("./edi"),
            Path("MTinv_OT") / "Data-preprocess" / "1" / "AKBST-AMT-L08",
            Path("Data-preprocess") / "1" / "AKBST-AMT-L08",
        ]
        for c in candidates:
            if c.is_dir():
                fs = sorted(c.glob("*.edi"))
                if fs:
                    return str(c), [str(f) for f in fs]
        return str(edi_dir), []

    @classmethod
    def read_custom_edi(
        cls,
        file_path: str,
        *,
        freq_min_hz: Optional[float] = None,
        freq_max_hz: Optional[float] = None,
    ) -> "PrepareData.CustomMT":
        with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
            lines = f.readlines()

        data_map: dict[str, np.ndarray] = {}
        lat, lon = 0.0, 0.0
        station_id: Optional[str] = None
        current_key: Optional[str] = None
        current_vals: list[float] = []

        for line in lines:
            line = line.strip()
            if not line:
                continue
            if line.startswith("LAT="):
                lat = cls.parse_dms(line.split("=")[1])
                continue
            if line.startswith("LONG="):
                lon = cls.parse_dms(line.split("=")[1])
                continue
            if line.startswith("DATAID="):
                raw = line.split("=", 1)[1].strip().strip('"\'')
                if raw:
                    station_id = raw
                continue

            if line.startswith(">"):
                if current_key:
                    data_map[current_key] = np.array(current_vals)
                key = line[1:].split("//")[0].strip()
                key = key.split()[0]
                key = key.replace(".", "").upper()
                current_key = key
                current_vals = []
            else:
                if current_key:
                    try:
                        vals = [float(x) for x in line.split()]
                        current_vals.extend(vals)
                    except Exception:
                        pass

        if current_key:
            data_map[current_key] = np.array(current_vals)

        # Fallback 1: filename stem (e.g. CAF054.edi -> CAF054, B01.edi -> B01)
        # Fallback 2: done in load_mt_objects using index (S1, S2, ...) if still None
        if station_id is None:
            station_id = Path(file_path).stem

        # CAFE land EDI (DATAID / filename ``CAF*``): ``Z..VAR`` blocks store **std-dev** of Re/Im.
        # CAFE offshore ``B*`` and other datasets (e.g. China): ``Z..VAR`` stores **variance** → sqrt.
        sid_up = str(station_id).strip().upper()
        z_var_blocks_are_std = sid_up.startswith("CAF")

        freqs = data_map["FREQ"]
        n = len(freqs)
        Z = np.zeros((n, 2, 2), dtype=complex)
        Z_err = np.zeros((n, 2, 2), dtype=float)
        has_err = False

        for i, c1 in enumerate(["X", "Y"]):
            for j, c2 in enumerate(["X", "Y"]):
                comp = c1 + c2
                rkey = "Z" + comp + "R"
                ikey = "Z" + comp + "I"
                if rkey in data_map and ikey in data_map:
                    Z[:, i, j] = data_map[rkey][:n] + 1j * data_map[ikey][:n]
                vkey = "Z" + comp + "VAR"
                if vkey in data_map:
                    v = np.abs(np.asarray(data_map[vkey][:n], dtype=float))
                    if z_var_blocks_are_std:
                        Z_err[:, i, j] = v
                    else:
                        Z_err[:, i, j] = np.sqrt(v)
                    has_err = True

        Z_err_out = Z_err if has_err else None

        # Optional: Tipper T (n_freq, 2) complex: Tzx, Tzy (Hz = Tzx*Hx + Tzy*Hy)
        # EDI block names: TXR.EXP/TXI.EXP/TYR.EXP/TYI.EXP or TZXR/TZXI/TZYR/TZYI
        T_out = None
        for txr, txi, tyr, tyi in [
            ("TXREXP", "TXIEXP", "TYREXP", "TYIEXP"),
            ("TZXR", "TZXI", "TZYR", "TZYI"),
        ]:
            if txr in data_map and txi in data_map and tyr in data_map and tyi in data_map:
                tzx = np.asarray(data_map[txr][:n], dtype=float) + 1j * np.asarray(data_map[txi][:n], dtype=float)
                tzy = np.asarray(data_map[tyr][:n], dtype=float) + 1j * np.asarray(data_map[tyi][:n], dtype=float)
                T_out = np.stack([tzx, tzy], axis=1)
                break

        # Optional: trim frequency band early (applies to frequency/Z/Z_err/T).
        if freq_min_hz is not None or freq_max_hz is not None:
            fmin = float(freq_min_hz) if freq_min_hz is not None else None
            fmax = float(freq_max_hz) if freq_max_hz is not None else None
            mask = np.ones(freqs.shape[0], dtype=bool)
            if fmin is not None:
                mask &= freqs >= fmin
            if fmax is not None:
                mask &= freqs <= fmax
            if not np.any(mask):
                raise ValueError(
                    f"No frequencies left after band selection while reading EDI: {file_path}. "
                    f"freq_min_hz={freq_min_hz}, freq_max_hz={freq_max_hz}."
                )
            freqs = freqs[mask]
            Z = Z[mask]
            if Z_err_out is not None:
                Z_err_out = Z_err_out[mask]
            if T_out is not None:
                T_out = T_out[mask]
           
        return cls.CustomMT(
            lat=float(lat),
            lon=float(lon),
            frequency=np.array(freqs),
            Z=Z,
            Z_err=Z_err_out,
            station_id=station_id,
            T=T_out,
        )

    @classmethod
    def _mv_per_km_per_nt_to_ohm_scale(cls) -> float:
        """Convert impedance from (mV/km)/nT to ohm.

        EDI commonly uses E in mV/km and B in nT:
        - E[V/m] = E[mV/km] * 1e-6
        - H[A/m] = B[T]/mu0 = (B[nT]*1e-9)/mu0
        So Z[ohm] = (E/H) = Z[(mV/km)/nT] * (1e-6) / (1e-9/mu0) = Z * (mu0 * 1e3)
        """

        return cls._mu0() * 1e3

    @classmethod
    def _normalize_edi_impedance_unit(cls, unit: str) -> str:
        u = str(unit).strip().lower()
        u = u.replace(" ", "")
        u = u.replace("\\", "/")
        u = u.replace("per", "/")
        # common aliases
        if u in {"omega", "ohms", "ohm", "si"}:
            return "ohm"
        if u in {"mv/km/nt", "mvkmnt", "mv/km/nT".lower(), "mvperkmpernt", "mv/km/nanoTesla".lower()}:
            return "mv/km/nt"
        if u in {"auto", "detect"}:
            return "auto"
        return u

    def _maybe_scale_impedance_units_inplace(self, mt: "PrepareData.CustomMT") -> bool:
        """Scale mt.Z and mt.Z_err so that impedance is in ohms.

        Returns True if scaling was applied.
        """

        unit = self._normalize_edi_impedance_unit(self.edi_impedance_unit)
        if unit not in {"auto", "ohm", "mv/km/nt"}:
            raise ValueError(
                f"Unknown edi_impedance_unit={self.edi_impedance_unit!r}. "
                "Use 'auto', 'ohm', or 'mv/km/nt'."
            )

        z = np.asarray(getattr(mt, "Z", None))
        if z is None or z.size == 0:
            return False

        scale = 1.0
        applied_unit = "ohm"

        if unit == "ohm":
            scale = 1.0
        elif unit == "mv/km/nt":
            scale = float(self._mv_per_km_per_nt_to_ohm_scale())
            applied_unit = "mv/km/nt->ohm"
        else:
            # Heuristic detection: if |Z| is extremely large, it's almost surely (mV/km)/nT.
            zabs_max = float(np.nanmax(np.abs(z)))
            if np.isfinite(zabs_max) and zabs_max > 1e3:
                scale = float(self._mv_per_km_per_nt_to_ohm_scale())
                applied_unit = "auto(mv/km/nt->ohm)"

        if scale == 1.0:
            setattr(mt, "impedance_unit", "ohm")
            return False

        mt.Z = np.asarray(mt.Z, dtype=np.complex128) * scale
        if getattr(mt, "Z_err", None) is not None:
            mt.Z_err = np.asarray(mt.Z_err, dtype=float) * abs(scale)
        setattr(mt, "impedance_unit", applied_unit)
        return True

    # -------------------------- impedance -> rho/phase (+ errors) --------------------------

    @staticmethod
    def _mu0() -> float:
        return float(4e-7 * np.pi)

    @classmethod
    def impedance_to_rho_phase(
        cls,
        freqs_hz: np.ndarray,
        Z: np.ndarray,
        Z_err: Optional[np.ndarray] = None,
        *,
        eps: float = 1e-8,
        eps_rho: float = 1e-6,
        eps_z: float = 1e-12,
        max_noise_std: float = 1.0,
        err_component_scale: float = 1.0,
    ) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray], Optional[np.ndarray], Optional[np.ndarray], Optional[np.ndarray]]:
        """把阻抗张量 Z 转成视电阻率 rho 与相位 phs，并可选传播误差。

        公式（与用户给出的 torch 版本一致）：
        - rho = |Z|^2 / (omega * mu0)
        - phs_raw = atan2(Im(Z), Re(Z)) [deg]  (range: (-180, 180])
        - phs = fold(phs_raw) into principal phase [0, 90]

        误差传播（假设 Re/Im 独立，且两者 std-dev 相同）：
        - d(rho) = sqrt( (∂rho/∂Zr * σr)^2 + (∂rho/∂Zi * σi)^2 )
        - d(phi) = sqrt( (∂phi/∂Zr * σr)^2 + (∂phi/∂Zi * σi)^2 )

        返回：rho, phs, rho_err, phs_err, rho_noise_std_log10, phs_noise_std_norm

        Notes
        -----
        - `Z_err` 来自 EDI 的 `Z..VAR`：在 ``read_custom_edi`` 中，``CAF*`` 台站按块内为 **std-dev**
          直接使用；其余台站按 **方差** 读入并已 ``sqrt`` 为 std-dev。此处一律按“每个复阻抗
          分量的 std-dev”做误差传播。若仍需缩放，可用 ``err_component_scale``。
        """

        freqs_hz = np.asarray(freqs_hz, dtype=float)
        Z = np.asarray(Z)
        if Z.ndim != 3 or Z.shape[1:] != (2, 2):
            raise ValueError(f"Z must have shape (n_freq,2,2), got {Z.shape}")
        if freqs_hz.ndim != 1 or freqs_hz.shape[0] != Z.shape[0]:
            raise ValueError("freqs_hz must be 1D and match Z.shape[0]")

        omega = 2.0 * np.pi * freqs_hz  # (n,)
        denom = omega * cls._mu0()      # (n,)
        denom_3d = denom[:, None, None]

        rho = (np.abs(Z) ** 2) / denom_3d
        phs_raw = np.degrees(np.arctan2(Z.imag, Z.real))
        # Fold to principal phase in [0, 90] to match inversion normalization (phase/90)
        phs_0_180 = np.mod(phs_raw, 180.0)
        phs = np.minimum(phs_0_180, 180.0 - phs_0_180)

        if Z_err is None:
            return rho, phs, None, None, None, None

        Z_err = np.asarray(Z_err, dtype=float)
        if Z_err.shape != Z.shape:
            raise ValueError(f"Z_err must have same shape as Z, got {Z_err.shape} vs {Z.shape}")

        sigma = np.maximum(Z_err, 0.0) * float(err_component_scale)
        sigma_r = sigma
        sigma_i = sigma

        Zr = Z.real
        Zi = Z.imag

        # ---------- rho std-dev ----------
        dRho_dZr = 2.0 * Zr / denom_3d
        dRho_dZi = 2.0 * Zi / denom_3d
        rho_err = np.sqrt((dRho_dZr * sigma_r) ** 2 + (dRho_dZi * sigma_i) ** 2)

        # ---------- phi std-dev (avoid 1/|Z|^2 explosion) ----------
        Zabs = np.abs(Z)
        Zabs_safe = np.maximum(Zabs, float(eps_z))
        dPhi_dZr = -Zi / (Zabs_safe**2)
        dPhi_dZi = Zr / (Zabs_safe**2)
        phi_err_rad = np.sqrt((dPhi_dZr * sigma_r) ** 2 + (dPhi_dZi * sigma_i) ** 2)
        phs_err = phi_err_rad * (180.0 / np.pi)

        # ---------- Dimensionless noise std-dev (for chi^2 / RMS) ----------
        rho_obs_safe = np.maximum(rho, float(eps_rho))
        rho_noise_std_log10 = np.clip(
            rho_err / (rho_obs_safe * np.log(10.0)),
            float(eps),
            float(max_noise_std),
        )
        phs_noise_std_norm = np.clip(
            phs_err / 90.0,
            float(eps),
            float(max_noise_std),
        )

        return rho, phs, rho_err, phs_err, rho_noise_std_log10, phs_noise_std_norm

    def compute_rho_phase(
        self,
        mt: "PrepareData.CustomMT",
        *,
        err_component_scale: float = 1.0,
    ) -> None:
        rho, phs, rho_err, phs_err, rho_noise_std_log10, phs_noise_std_norm = self.impedance_to_rho_phase(
            freqs_hz=mt.frequency,
            Z=mt.Z,
            Z_err=mt.Z_err,
            err_component_scale=err_component_scale,
        )
        mt.rho = rho
        mt.phs = phs
        mt.rho_err = rho_err
        mt.phs_err = phs_err
        mt.rho_noise_std_log10 = rho_noise_std_log10
        mt.phs_noise_std_norm = phs_noise_std_norm

    def compute_rho_phase_all(
        self,
        mt_objects: Optional[Sequence["PrepareData.CustomMT"]] = None,
        *,
        err_component_scale: float = 1.0,
    ) -> None:
        if mt_objects is None:
            mt_objects = self.mt_objects
        for mt in mt_objects:
            self.compute_rho_phase(mt, err_component_scale=err_component_scale)

    # -------------------------- export for inversion (torch) --------------------------

    @staticmethod
    def _torch_complex_dtype_from_float(dtype):
        """Map float dtype -> complex dtype (torch)."""

        try:
            import torch  # type: ignore[import-untyped]
        except Exception as e:
            raise ImportError("export_data_dict_for_2d_inversion() requires torch") from e

        if dtype is None:
            return torch.complex128
        if dtype == torch.float32:
            return torch.complex64
        return torch.complex128



