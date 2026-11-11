"""
Step 4 — Ancillary layers per field: elevation (USGS 3DEP), soils (SSURGO via
Soil Data Access), irrigation flag (center-pivot shape heuristic from fields.py,
optionally overridden by a user-supplied layer such as LANID).

All functions are per-field and cached to data/ancillary/*.parquet.
"""
from __future__ import annotations

import json
import time

import numpy as np
import pandas as pd
import requests

from . import config as C

EPQS = "https://epqs.nationalmap.gov/v1/json"
SDA = "https://sdmdataaccess.sc.egov.usda.gov/Tabular/post.rest"


# ----------------------------------------------------------------------------
# Elevation
# ----------------------------------------------------------------------------
def elevation_point(lon: float, lat: float) -> float:
    r = requests.get(EPQS, params={"x": lon, "y": lat, "units": "Meters", "wkid": 4326,
                                   "includeDate": "false"}, timeout=30)
    r.raise_for_status()
    v = r.json().get("value")
    return float(v) if v not in (None, "", "-1000000") else np.nan


def elevation_fields(fields: pd.DataFrame, use_py3dep: bool = True) -> pd.DataFrame:
    """Elevation (m) and slope (%) per field. With py3dep installed we sample a
    10 m DEM over the polygon (mean elev + mean slope); otherwise centroid elev only."""
    p = C.ANC_DIR / "elevation.parquet"
    cache = pd.read_parquet(p) if p.exists() else pd.DataFrame(columns=["field_id", "elev_m", "slope_pct"])
    todo = fields[~fields["field_id"].isin(cache["field_id"])]
    rows = []
    if use_py3dep:
        try:
            import py3dep
            for _, f in todo.iterrows():
                try:
                    dem = py3dep.get_dem(f.geometry, resolution=10, crs="EPSG:4326")
                except Exception:
                    dem = None
                if dem is not None:
                    arr = dem.values.astype("float32")
                    gy, gx = np.gradient(arr, 10.0)
                    slope_pct = float(np.nanmean(np.hypot(gx, gy)) * 100)
                    rows.append({"field_id": f["field_id"], "elev_m": float(np.nanmean(arr)), "slope_pct": slope_pct})
                    continue
                rows.append({"field_id": f["field_id"], "elev_m": elevation_point(f["lon"], f["lat"]), "slope_pct": np.nan})
        except ImportError:
            use_py3dep = False
    if not use_py3dep:
        for _, f in todo.iterrows():
            try:
                e = elevation_point(f["lon"], f["lat"])
            except Exception:
                e = np.nan
            rows.append({"field_id": f["field_id"], "elev_m": e, "slope_pct": np.nan})
            time.sleep(0.05)
    out = pd.concat([cache, pd.DataFrame(rows)], ignore_index=True) if rows else cache
    out.to_parquet(p, index=False)
    return out


# ----------------------------------------------------------------------------
# Soils (SSURGO via Soil Data Access)
# ----------------------------------------------------------------------------
def _sda(sql: str) -> pd.DataFrame:
    r = requests.post(SDA, data=json.dumps({"query": sql, "format": "JSON+COLUMNNAME"}),
                      headers={"Content-Type": "application/json"}, timeout=120)
    r.raise_for_status()
    tbl = r.json().get("Table", [])
    if not tbl:
        return pd.DataFrame()
    return pd.DataFrame(tbl[1:], columns=tbl[0])


def soils_fields(fields: pd.DataFrame, batch: int = 50) -> pd.DataFrame:
    """Dominant map unit at each field centroid with muaggatt + valu1 attributes:
    drainage class, hydrologic group, AWC 0-100 cm, slope, NCCPI (corn/cotton/soy)."""
    p = C.ANC_DIR / "soils.parquet"
    cache = pd.read_parquet(p) if p.exists() else pd.DataFrame(columns=["field_id"])
    todo = fields[~fields["field_id"].isin(cache["field_id"])]
    rows = []
    for i in range(0, len(todo), batch):
        chunk = todo.iloc[i:i + batch]
        for _, f in chunk.iterrows():
            sql = f"""
            SELECT TOP 1 m.mukey, m.muname, a.drclassdcd, a.hydgrpdcd, a.aws0100wta, a.slopegraddcp,
                   a.taxorderdcd, a.taxpartsizedcd, v.nccpi3corn, v.nccpi3cot, v.nccpi3soy, v.nccpi3all
            FROM mapunit m
            JOIN muaggatt a ON a.mukey = m.mukey
            LEFT JOIN valu1 v ON v.mukey = m.mukey
            WHERE m.mukey IN (SELECT * FROM SDA_Get_Mukey_from_intersection_with_WktWgs84('point({f['lon']} {f['lat']})'))
            """
            try:
                df = _sda(sql)
            except Exception as e:
                df = pd.DataFrame()
            rec = {"field_id": f["field_id"]}
            if len(df):
                rec.update(df.iloc[0].to_dict())
            rows.append(rec)
        time.sleep(0.2)
    out = pd.concat([cache, pd.DataFrame(rows)], ignore_index=True) if rows else cache
    for c in ("aws0100wta", "slopegraddcp", "nccpi3corn", "nccpi3cot", "nccpi3soy", "nccpi3all"):
        if c in out:
            out[c] = pd.to_numeric(out[c], errors="coerce")
    out.to_parquet(p, index=False)
    return out


# ----------------------------------------------------------------------------
# Irrigation
# ----------------------------------------------------------------------------
def irrigation_fields(fields: pd.DataFrame, lanid_raster: str | None = None) -> pd.DataFrame:
    """Irrigated flag. Default = center-pivot geometry heuristic from fields.py.
    If a LANID (or other) irrigation raster path is given, use majority within the field."""
    out = fields[["field_id", "pivot_like"]].rename(columns={"pivot_like": "irrigated"}).copy()
    out["irrigation_source"] = "pivot_shape"
    if lanid_raster:
        try:
            import rasterio
            from rasterstats import zonal_stats
            zs = zonal_stats(fields.geometry, lanid_raster, stats=["mean"], nodata=0)
            frac = np.array([z["mean"] or 0 for z in zs])
            out["irrigated"] = frac > 0.5
            out["irrigation_source"] = "raster"
        except Exception as e:  # pragma: no cover
            print("irrigation raster failed, keeping shape heuristic:", e)
    return out


def all_ancillary(fields: pd.DataFrame, do_elev=True, do_soil=True, lanid=None) -> pd.DataFrame:
    anc = irrigation_fields(fields, lanid)
    if do_elev:
        anc = anc.merge(elevation_fields(fields), on="field_id", how="left")
    if do_soil:
        anc = anc.merge(soils_fields(fields), on="field_id", how="left")
    anc.to_parquet(C.ANC_DIR / "ancillary_merged.parquet", index=False)
    return anc
