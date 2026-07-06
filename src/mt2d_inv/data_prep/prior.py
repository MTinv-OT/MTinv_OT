"""GMT prior grid helpers."""
from __future__ import annotations

from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple, Union

import numpy as np



class PriorMixin:
    def build_prior_lon_lat_at_y_centers(
        self,
        yn: Union[np.ndarray, Any],
        sort_by: str = "profile_pos_m",
        mt_objects: Optional[Sequence["PrepareData.CustomMT"]] = None,
        extrapolate: str = "clamp",
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Interpolate station lon/lat onto horizontal **cell centers** of ``yn`` (meters along profile).

        Use the same ``sort_by`` as ``export_data_dict_for_2d_inversion`` so columns match
        :class:`mt2d_inv.inversion.base.MT2DInverter` ``yn`` / ``stations``.

        Parameters
        ----------
        yn
            Horizontal grid **node** positions (m), length ``ny+1``; centers are ``(yn[:-1]+yn[1:])/2``.
        extrapolate
            ``clamp`` (default): outside the station span, hold edge station lon/lat.
            ``linear``: use ``numpy.interp`` default (constant edge value in numpy).

        Returns
        -------
        lon_centers, lat_centers : ndarray
            Length ``len(yn) - 1``, ready for ``prior_options['lon']`` / ``['lat']``.
        """
        yn_np = self._yn_edges_to_numpy(yn)
        if yn_np.size < 2:
            raise ValueError("yn must have at least 2 nodes")
        y_centers = 0.5 * (yn_np[:-1] + yn_np[1:])
        st, lon_s, lat_s = self.get_station_lon_lat_sorted(sort_by=sort_by, mt_objects=mt_objects)
        if st.size < 2:
            lon_c = np.full(y_centers.shape, float(lon_s[0]), dtype=np.float64)
            lat_c = np.full(y_centers.shape, float(lat_s[0]), dtype=np.float64)
            return lon_c, lat_c
        if extrapolate == "clamp":
            lon_c = np.interp(y_centers, st, lon_s, left=float(lon_s[0]), right=float(lon_s[-1]))
            lat_c = np.interp(y_centers, st, lat_s, left=float(lat_s[0]), right=float(lat_s[-1]))
        elif extrapolate == "linear":
            lon_c = np.interp(y_centers, st, lon_s)
            lat_c = np.interp(y_centers, st, lat_s)
        else:
            raise ValueError("extrapolate must be 'clamp' or 'linear'")
        return lon_c.astype(np.float64), lat_c.astype(np.float64)

    def build_prior_options(
        self,
        yn: Union[np.ndarray, Any],
        sort_by: str = "profile_pos_m",
        mt_objects: Optional[Sequence["PrepareData.CustomMT"]] = None,
        extrapolate: str = "clamp",
        **prior_fields: Any,
    ) -> Dict[str, Any]:
        """
        Build a ``prior_options`` dict for :meth:`mt2d_inv.inversion.base.MT2DInverter.initialize_model`
        (``use_prior_model=True``).

        Inserts ``lon`` and ``lat`` at each horizontal cell center (see ``build_prior_lon_lat_at_y_centers``).
        Pass GMT paths and conductivities as keywords, e.g.::

            prior = prep.build_prior_options(
                inv.yn,
                sediment_grd=\".../sedthick_world_v2.grd\",
                sigma_sediment=1.0/5.0,
                sigma_background=1e-2,
            )

        Notes
        -----
        Standard ``.grd`` files are **2D** (lon × lat) with one scalar per node (e.g. thickness or
        interface depth in m), not (lon, lat, depth, conductivity).
        """
        lon_c, lat_c = self.build_prior_lon_lat_at_y_centers(
            yn, sort_by=sort_by, mt_objects=mt_objects, extrapolate=extrapolate
        )
        out: Dict[str, Any] = {"lon": lon_c, "lat": lat_c}
        for k, v in prior_fields.items():
            if v is not None:
                out[k] = v
        return out

