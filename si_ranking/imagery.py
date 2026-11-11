"""
Step 2 — Sentinel-2 L2A retrieval and per-field vegetation-index time series.

Source: Earth Search STAC (AWS, open, no auth). For every scene touching the
area of interest we read only the window covering the fields (COG range
requests), mask clouds with the SCL band, compute NDVI / GNDVI / NDRE, and
reduce to per-field statistics (median, p10, p90, n valid pixels) with a
rasterized field-ID grid. Results are cached as parquet so the app and the
planting-date module never touch imagery directly.

Usage:
    python -m si_ranking.cli ingest --lon -83.5 --lat 31.45 --radius-km 15 \
        --start 2026-02-01 --end 2026-11-01
"""
from __future__ import annotations

import datetime as dt
import warnings
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd

from . import config as C

try:  # heavy deps imported lazily so the rest of the package works without them
    import pystac_client
    import odc.stac  # noqa: F401
    import rasterio
    from rasterio import features
    HAVE_IMAGERY = True
except Exception:  # pragma: no cover
    HAVE_IMAGERY = False


def _ts_path(aoi_tag: str) -> Path:
    return C.TS_DIR / f"vi_{aoi_tag}.parquet"


def aoi_tag_for(lon: float, lat: float, radius_km: float) -> str:
    return f"{lon:.3f}_{lat:.3f}_{int(radius_km)}km".replace("-", "m")


from functools import lru_cache


@lru_cache(maxsize=4)
def _client(url):
    """One catalogue connection per process (opening it costs a round trip)."""
    return pystac_client.Client.open(url)


def search_scenes(bbox, start: str, end: str, collection: str = C.STAC_COLLECTION,
                  max_cloud: float = C.MAX_CLOUD_COVER):
    cat = _client(C.STAC_URL)
    q = cat.search(collections=[collection], bbox=bbox, datetime=f"{start}/{end}",
                   query={"eo:cloud_cover": {"lt": max_cloud}})
    items = list(q.items())
    items.sort(key=lambda it: it.datetime)
    print(f"{len(items)} scenes in {collection} {start}..{end} (<{max_cloud}% cloud)")
    return items


def _boa_offset(item) -> float:
    """Return value to subtract from DN before /10000 (Sentinel-2 PB>=4.00 adds +1000)."""
    p = item.properties
    if p.get("earthsearch:boa_offset_applied", False):
        return 0.0
    pb = p.get("s2:processing_baseline") or p.get("sentinel:processing_baseline") or "0"
    try:
        return 1000.0 if float(pb) >= 4.0 else 0.0
    except ValueError:
        return 0.0


def _load_item(item, bbox, resolution: int):
    """Load required bands for one item over bbox as an xarray Dataset (EPSG of the item)."""
    import odc.stac
    bands = ["red", "green", "nir", "nir08", "rededge1", "scl"]
    ds = odc.stac.load([item], bands=bands, bbox=bbox, resolution=resolution,
                       chunks=None, groupby="solar_day", resampling={"scl": "nearest"},
                       fail_on_error=False)
    return ds.isel(time=0)


def _rasterize_fields(fields_inner: gpd.GeoSeries, ds) -> np.ndarray:
    """Burn 1-based field index into the dataset grid; 0 = background."""
    transform = ds.odc.geobox.transform
    shape = (ds.sizes["y"], ds.sizes["x"])
    crs = ds.odc.geobox.crs
    geoms = fields_inner.to_crs(str(crs))
    shapes = [(geom, i + 1) for i, geom in enumerate(geoms.values) if not geom.is_empty]
    return features.rasterize(shapes, out_shape=shape, transform=transform, fill=0,
                              dtype="int32", all_touched=False)


def _zonal(values: np.ndarray, labels: np.ndarray, n: int, valid: np.ndarray):
    """Per-label median/p10/p90/count using sorting; returns arrays of length n."""
    med = np.full(n, np.nan); p10 = np.full(n, np.nan); p90 = np.full(n, np.nan); cnt = np.zeros(n, int)
    m = valid & (labels > 0) & np.isfinite(values)
    lab = labels[m]; val = values[m]
    if lab.size == 0:
        return med, p10, p90, cnt
    order = np.argsort(lab, kind="stable")
    lab = lab[order]; val = val[order]
    bounds = np.flatnonzero(np.diff(lab)) + 1
    starts = np.r_[0, bounds]; ends = np.r_[bounds, lab.size]
    for s, e in zip(starts, ends):
        k = lab[s] - 1
        v = val[s:e]
        cnt[k] = v.size
        if v.size:
            med[k], p10[k], p90[k] = np.percentile(v, [50, 10, 90])
    return med, p10, p90, cnt


def ingest(fields: gpd.GeoDataFrame, start: str, end: str, aoi_tag: str,
           resolution: int = 10, collection: str = C.STAC_COLLECTION,
           force: bool = False) -> pd.DataFrame:
    """Compute per-field VI stats for every clear scene; append to the parquet cache."""
    if not HAVE_IMAGERY:
        raise RuntimeError("Install imagery deps: pip install pystac-client odc-stac rasterio rioxarray")
    out_path = _ts_path(aoi_tag)
    cache = pd.read_parquet(out_path) if (out_path.exists() and not force) else pd.DataFrame()
    done = set(cache["scene_id"]) if len(cache) else set()

    bbox = tuple(fields.total_bounds)
    items = search_scenes(bbox, start, end, collection)
    n = len(fields)
    fid = fields["field_id"].values
    rows = []
    labels = None
    for it in items:
        if it.id in done:
            continue
        try:
            ds = _load_item(it, bbox, resolution)
        except Exception as e:  # pragma: no cover
            warnings.warn(f"skip {it.id}: {e}")
            continue
        if labels is None or labels.shape != (ds.sizes["y"], ds.sizes["x"]):
            labels = _rasterize_fields(fields["geometry_inner"], ds)
        scl = ds["scl"].values
        clear = np.isin(scl, C.SCL_CLEAR)
        off = _boa_offset(it)
        b = {k: (ds[k].values.astype("float32") - off) / 10000.0 for k in ("red", "green", "nir", "nir08", "rededge1")}
        valid = clear & (b["nir"] > 0) & (b["red"] > 0)
        rec = {"scene_id": it.id, "date": pd.Timestamp(it.datetime).tz_localize(None).normalize(),
               "cloud_cover": it.properties.get("eo:cloud_cover")}
        for vi, (a, c) in C.VI_DEFS.items():
            v = (b[a] - b[c]) / (b[a] + b[c] + 1e-6)
            med, p10, p90, cnt = _zonal(v, labels, n, valid)
            for k in range(n):
                if cnt[k] >= C.MIN_PIXELS_10M * (resolution / 10) ** -2:
                    rows.append({**rec, "field_id": fid[k], "vi": vi, "median": med[k],
                                 "p10": p10[k], "p90": p90[k], "n_pix": int(cnt[k])})
        print(f"  {it.datetime.date()} {it.id}  clear={clear.mean():.0%}")
    new = pd.DataFrame(rows)
    allts = pd.concat([cache, new], ignore_index=True) if len(cache) else new
    if len(allts):
        allts = allts.drop_duplicates(["scene_id", "field_id", "vi"])
        allts.to_parquet(out_path, index=False)
        print(f"Time series: {len(allts):,} rows -> {out_path}")
    return allts


def load_timeseries(aoi_tag: str) -> pd.DataFrame:
    p = _ts_path(aoi_tag)
    if not p.exists():
        return pd.DataFrame(columns=["scene_id", "date", "field_id", "vi", "median", "p10", "p90", "n_pix"])
    return pd.read_parquet(p)


def latest_observation(ts: pd.DataFrame, vi: str, as_of: dt.date | None = None,
                       window_days: int = 20) -> pd.DataFrame:
    """Latest clear value per field for a VI (max within the last `window_days` before as_of)."""
    t = ts[ts["vi"] == vi].copy()
    if as_of is not None:
        t = t[t["date"] <= pd.Timestamp(as_of)]
    if t.empty:
        return t
    last = t["date"].max()
    t = t[t["date"] >= last - pd.Timedelta(days=window_days)]
    # max-value composite per field over the window (classic cloud-robust reduce)
    idx = t.groupby("field_id")["median"].idxmax()
    return t.loc[idx].reset_index(drop=True)
