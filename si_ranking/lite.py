"""
Lite ranking workflow — the core product.

    boundary (upload / draw / click CSB field)
      -> buffer around it
      -> most recent clear Sentinel-2 scene(s) over the buffer   (on demand, nothing pre-ingested)
      -> VI per field (NDVI / GNDVI / NDRE), inward-buffered polygons
      -> virtual reference = 95th percentile of same-crop fields in the buffer
      -> SI = VI / reference, rank + percentile

Scenes are walked backwards from the "as of" date. A field takes its value from
the first (most recent) scene in which it has enough clear pixels, so every
field is tagged with its own observation date. We stop when nearly all fields
have a value or we have looked back `max_days`.
"""
from __future__ import annotations

import datetime as dt
from dataclasses import dataclass

import geopandas as gpd
import numpy as np
import pandas as pd

from . import config as C


@dataclass
class LiteResult:
    table: pd.DataFrame          # one row per field: vi, obs_date, SI, rank, percentile, cohort flag
    reference: float
    target_id: str
    scenes_used: list[str]
    vi: str


def _load_bands(item, bbox, bands: list[str], resolution: int):
    import odc.stac
    ds = odc.stac.load([item], bands=bands, bbox=bbox, resolution=resolution, chunks=None,
                       groupby="solar_day", resampling={"scl": "nearest"}, fail_on_error=False)
    return ds.isel(time=0)


def latest_vi_all(fields: gpd.GeoDataFrame, as_of: dt.date | None = None, max_days: int = 45,
                  min_pix: int = C.MIN_PIXELS_10M, coverage_target: float = 0.97,
                  collection: str = C.STAC_COLLECTION, log=print, progress=None) -> tuple[pd.DataFrame, list[str]]:
    """Most-recent clear look per field, ALL indices at once (NDVI, GNDVI, NDRE from the same scene and
    the same pixels, red-edge resampled to 10 m) -> switching index never changes the compared fields.
    Uses the fast engine (60 m cloud screen, parallel scenes, per-scene field cache)."""
    from . import scenes
    as_of = as_of or dt.date.today()
    return scenes.latest(fields, as_of, max_days, dict(C.VI_ENGINE), kind=C.VI_KIND, min_pix=min_pix,
                         coverage_target=coverage_target, progress=progress, log=log)


def pick_vi(obs: pd.DataFrame, vi: str) -> pd.DataFrame:
    """View of latest_vi_all output in the single-index schema (vi_median / vi_p10 / vi_p90)."""
    out = obs[["field_id", "n_pix", "obs_date", "scene_id"]].copy()
    out["vi_median"] = obs[vi]
    out["vi_p10"] = obs[f"{vi}_p10"] if f"{vi}_p10" in obs else np.nan
    out["vi_p90"] = obs[f"{vi}_p90"] if f"{vi}_p90" in obs else np.nan
    return out


def latest_vi(fields: gpd.GeoDataFrame, vi: str, as_of: dt.date | None = None, max_days: int = 45, **kw):
    """Back-compatible single-index call (same scenes/pixels as latest_vi_all)."""
    obs, used = latest_vi_all(fields, as_of, max_days, **kw)
    return pick_vi(obs, vi), used


def make_target(geom, crop: str, field_id: str = "MY_FIELD") -> gpd.GeoDataFrame:
    """Wrap a user polygon (WGS84 shapely geometry) in the same schema as CSB fields."""
    g = gpd.GeoDataFrame({"field_id": [field_id], "crop_latest": [crop]}, geometry=[geom], crs="EPSG:4326")
    ga = g.to_crs(C.EQUAL_AREA_CRS)
    g["acres"] = ga.geometry.area.values / 1e4 * 2.47105
    from .fields import inner_buffer
    g["geometry_inner"] = gpd.GeoSeries(inner_buffer(ga.geometry), crs=C.EQUAL_AREA_CRS).to_crs("EPSG:4326").values
    cen = ga.geometry.centroid.to_crs("EPSG:4326")
    g["lon"], g["lat"] = cen.x.values, cen.y.values
    pp = float(4 * np.pi * ga.geometry.area.iloc[0] / (ga.geometry.length.iloc[0] ** 2))
    g["circularity"] = pp
    g["pivot_like"] = bool(pp > 0.80 and g["acres"].iloc[0] > 20)
    g["rotation"] = "user"; g["dist_km"] = 0.0
    return g


def rank_buffer(fields: gpd.GeoDataFrame, target_id: str, crop: str, vi: str,
                as_of: dt.date | None = None, q: float = C.VIRTUAL_REF_QUANTILE,
                max_days: int = 45, log=print) -> LiteResult:
    """fields: CSB fields in the buffer (+ the target row). Only same-crop fields form the cohort."""
    cohort_fields = fields[(fields["crop_latest"] == crop) | (fields["field_id"] == target_id)].copy()
    min_ac = C.CROP_PARAMS[crop].min_acres_10m      # same field set for every index
    cohort_fields = cohort_fields[(cohort_fields["acres"] >= min_ac) | (cohort_fields["field_id"] == target_id)]
    cohort_fields = cohort_fields[~cohort_fields["geometry_inner"].is_empty].reset_index(drop=True)
    log(f"{len(cohort_fields)} {crop} fields >= {min_ac:.0f} ac in buffer (incl. target)")

    obs, used = latest_vi(cohort_fields, vi, as_of, max_days=max_days, log=log)
    t = cohort_fields.drop(columns=["geometry", "geometry_inner"]).merge(obs, on="field_id", how="left")
    v = t["vi_median"].astype(float)
    ref = float(np.nanquantile(v.dropna(), q)) if v.notna().sum() >= 3 else float("nan")
    t["vi_ref"] = ref
    t["SI"] = t["vi_median"] / ref if ref > 0 else np.nan          # a ratio needs a positive reference
    t["rank"] = t["vi_median"].rank(ascending=False, method="first").astype("Int64")
    t["n_ranked"] = int(v.notna().sum())
    t["percentile"] = t["vi_median"].rank(pct=True) * 100
    t = t.sort_values("rank")
    return LiteResult(t, ref, target_id, used, vi)


def score(table: pd.DataFrame, cohort_mask: pd.Series, target_id: str, q: float = C.VIRTUAL_REF_QUANTILE,
          ref_mask: pd.Series | None = None) -> tuple[pd.DataFrame, float]:
    """Virtual reference = quantile q of VI over `ref_mask` fields (default: the cohort); SI / rank / percentile
    are computed within `cohort_mask`. Non-cohort fields keep their VI and SI but get no rank."""
    t = table.copy()
    coh = t[cohort_mask & t["vi_median"].notna()]
    refset = t[(ref_mask if ref_mask is not None else cohort_mask) & t["vi_median"].notna() & t["field_id"].ne(target_id)]
    if len(refset) < 3:                     # reference = the best of the OTHER fields (never includes my own field)
        refset = coh[coh["field_id"] != target_id]
    ref = float(np.nanquantile(refset["vi_median"], q)) if len(refset) >= 3 else float("nan")
    t["in_ref"] = (ref_mask if ref_mask is not None else cohort_mask).values & t["vi_median"].notna().values
    t["vi_ref"] = ref
    t["SI"] = t["vi_median"] / ref if ref > 0 else np.nan          # a ratio needs a positive reference
    t["in_cohort"] = cohort_mask.values
    t["rank"] = pd.array([pd.NA] * len(t), dtype="Int64")
    t.loc[coh.index, "rank"] = coh["vi_median"].rank(ascending=False, method="first").astype("Int64")
    t["percentile"] = np.nan
    t.loc[coh.index, "percentile"] = coh["vi_median"].rank(pct=True) * 100
    t["n_ranked"] = len(coh)
    return t.sort_values(["in_cohort", "vi_median"], ascending=[False, False]), ref
