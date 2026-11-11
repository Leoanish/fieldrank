"""
Static field profile: terrain, soil and irrigation for every CSB field.

    terrain     Copernicus GLO-30 DEM from Earth Search (the same catalogue as Sentinel-2);
                fallback USGS 3DEP via Planetary Computer
                -> elevation median / mean / p05 / p95 / range, slope median / mean / p90 (%)
    soil        SSURGO map-unit polygons from Soil Data Access, clipped to the fields on the
                server and intersected exactly with each field (fallback: gNATSGO raster);
                horizon data 0-30 cm depth-weighted, component-% weighted, area-weighted
                -> sand / silt / clay %, organic matter %, AWC, water storage 0-30 cm,
                   USDA texture class + texture group, dominant drainage class, hydrologic group
    irrigation  LANID-US (Xie & Lark 2021, 30 m, 1997-2017) if the clipped rasters exist in
                data/ancillary/lanid/, combined with the centre-pivot shape flag
                -> lanid_recent (share irrigated 2015-17), lanid_freq (years of 21),
                   lanid_ever, irrigated, irr_source

Everything is computed on the field's inward-buffered polygon (edges, ditches and tree
lines removed), cached per field_id in data/ancillary/field_profile.parquet and only
computed for fields not yet cached -> a buffer that was seen before costs nothing.
"""
from __future__ import annotations

import hashlib
import json
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd

from . import config as C

PROFILE_VERSION = "p2"
PROFILE_PATH = C.ANC_DIR / "field_profile.parquet"
MUKEY_PATH = C.ANC_DIR / "mukey_soil_0_30cm.parquet"          # cache of SDA answers
GNATSGO_TABLE = C.ANC_DIR / "gnatsgo_mukey_0_30cm.parquet"     # every US map unit, built once from Planetary Computer
SOILGRIDS = "/vsicurl?max_retry=3&retry_delay=1&list_dir=no&url=https://files.isric.org/soilgrids/latest/data/{v}/{v}_{d}cm_mean.vrt"
SOIL_COLS = ["mukey", "sand", "silt", "clay", "om", "awc", "aws_0_30cm", "drainage", "hydgrp"]
LANID_DIR = C.ANC_DIR / "lanid"
PC_STAC = "https://planetarycomputer.microsoft.com/api/stac/v1"
SDA_URL = "https://sdmdataaccess.sc.egov.usda.gov/Tabular/post.rest"
GRID_CRS = "EPSG:5070"          # CONUS Albers: metres, equal area -> slope and areas are direct
RES = 30.0                      # DEM and LANID are 30 m products; soil is exact vector
TILE_DEG = 0.30                 # fields are processed in ~30 km tiles (one DEM read each)
SDA_BATCH = 60                  # fields per Soil Data Access spatial query
SOIL_DEPTH_CM = 30

TEXTURE_GROUP = {
    "sand": "coarse", "loamy sand": "coarse",
    "sandy loam": "moderately coarse",
    "loam": "medium", "silt loam": "medium", "silt": "medium",
    "sandy clay loam": "moderately fine", "clay loam": "moderately fine", "silty clay loam": "moderately fine",
    "sandy clay": "fine", "silty clay": "fine", "clay": "fine",
}
DRAINAGE_ORDER = ["Excessively drained", "Somewhat excessively drained", "Well drained", "Moderately well drained",
                  "Somewhat poorly drained", "Poorly drained", "Very poorly drained"]

NUM_COLS = ["elev_median", "elev_mean", "elev_p05", "elev_p95", "elev_range", "slope_median", "slope_mean", "slope_p90",
            "sand", "silt", "clay", "om", "awc", "aws_0_30cm", "dom_mukey_share", "n_mapunits",
            "lanid_recent", "lanid_freq", "lanid_ever"]
CAT_COLS = ["texture", "texture_group", "drainage", "hydgrp", "irr_source"]


# =============================================================================== helpers
def texture_class(sand: float, clay: float) -> str | None:
    """USDA soil texture triangle."""
    if sand is None or clay is None or not np.isfinite(sand) or not np.isfinite(clay):
        return None
    silt = 100.0 - sand - clay
    if silt + 1.5 * clay < 15:
        return "sand"
    if silt + 2 * clay < 30:
        return "loamy sand"
    if (7 <= clay < 20 and sand > 52) or (clay < 7 and silt < 50):
        return "sandy loam"
    if 7 <= clay < 27 and 28 <= silt < 50 and sand <= 52:
        return "loam"
    if silt >= 80 and clay < 12:
        return "silt"
    if (silt >= 50 and 12 <= clay < 27) or (50 <= silt < 80 and clay < 12):
        return "silt loam"
    if 20 <= clay < 35 and silt < 28 and sand > 45:
        return "sandy clay loam"
    if 27 <= clay < 40 and 20 < sand <= 45:
        return "clay loam"
    if 27 <= clay < 40 and sand <= 20:
        return "silty clay loam"
    if clay >= 35 and sand > 45:
        return "sandy clay"
    if clay >= 40 and silt >= 40:
        return "silty clay"
    return "clay"


def zonal_percentiles(values: np.ndarray, labels: np.ndarray, n: int, valid: np.ndarray, qs=(5, 10, 50, 90, 95)):
    """Per-label mean + percentiles for labels 1..n. Returns (mean[n], pct[n, len(qs)], count[n])."""
    mean = np.full(n, np.nan); pct = np.full((n, len(qs)), np.nan); cnt = np.zeros(n, int)
    m = valid & (labels > 0) & np.isfinite(values)
    lab = labels[m]; val = values[m]
    if lab.size == 0:
        return mean, pct, cnt
    o = np.argsort(lab, kind="stable"); lab = lab[o]; val = val[o]
    cut = np.flatnonzero(np.diff(lab)) + 1
    for s, e in zip(np.r_[0, cut], np.r_[cut, lab.size]):
        k = lab[s] - 1; v = val[s:e]
        cnt[k] = v.size; mean[k] = v.mean(); pct[k] = np.percentile(v, qs)
    return mean, pct, cnt


def _rasterize(geoms: gpd.GeoSeries, gbox, all_touched=False) -> np.ndarray:
    from rasterio import features
    shapes = [(g, i + 1) for i, g in enumerate(geoms.values) if g is not None and not g.is_empty]
    if not shapes:
        return np.zeros(gbox.shape, "int32")
    return features.rasterize(shapes, out_shape=gbox.shape, transform=gbox.transform, fill=0, dtype="int32", all_touched=all_touched)


def _geobox(bounds_5070, pad=60.0):
    from odc.geo.geobox import GeoBox
    x0, y0, x1, y1 = bounds_5070
    return GeoBox.from_bbox((x0 - pad, y0 - pad, x1 + pad, y1 + pad), crs=GRID_CRS, resolution=RES)


def _cache_id(row) -> str:
    """CSB fields keep their id; a user polygon gets a geometry hash so it can be cached too."""
    if row["field_id"] != "MY_FIELD":
        return str(row["field_id"])
    return "MY_" + hashlib.md5(row["geometry"].wkb).hexdigest()[:12]


# =============================================================================== Planetary Computer
_client_lock = threading.Lock()
_client = None


def _pc_client():
    global _client
    with _client_lock:
        if _client is None:
            import planetary_computer as pc
            import pystac_client
            _client = pystac_client.Client.open(PC_STAC, modifier=pc.sign_inplace)
    return _client


def _pc_items(collection: str, bbox_ll):
    return list(_pc_client().search(collections=[collection], bbox=bbox_ll).items())


def _load_mosaic(items, asset: str, gbox, resampling: str, dtype: str, nodata_out):
    """Load each item onto gbox and take the first valid value. Returns (array | None, first error text)."""
    import odc.stac
    out, err = None, ""
    for it in items:
        try:
            ds = odc.stac.load([it], bands=[asset], geobox=gbox, resampling=resampling, chunks=None, fail_on_error=True)
        except Exception as e:
            err = err or f"{type(e).__name__}: {e}"[:200]
            continue
        a = ds[asset].isel(time=0).values
        nd = ds[asset].attrs.get("nodata")
        a = a.astype(dtype)
        bad = ~np.isfinite(a) if np.issubdtype(a.dtype, np.floating) else np.zeros(a.shape, bool)
        if nd is not None:
            bad |= a == np.array(nd).astype(dtype)
        if out is None:
            out = np.full(a.shape, nodata_out, dtype)
        fill = (out == nodata_out) if not (isinstance(nodata_out, float) and np.isnan(nodata_out)) else np.isnan(out)
        take = fill & ~bad
        out[take] = a[take]
    return out, err


def load_dem(gbox, bbox_ll):
    """Returns (dem | None, source text, error text). Copernicus GLO-30 on Earth Search first
    (same host as Sentinel-2, no signing), USGS 3DEP on Planetary Computer as fallback."""
    errs = []
    try:
        import pystac_client
        items = list(pystac_client.Client.open(C.STAC_URL).search(collections=["cop-dem-glo-30"], bbox=bbox_ll).items())
        dem, err = _load_mosaic(items, "data", gbox, "bilinear", "float32", np.nan)
        if dem is not None and np.isfinite(dem).any():
            dem[(dem < -100) | (dem > 9000)] = np.nan
            return dem, "Copernicus GLO-30", ""
        errs.append(f"Copernicus: {err or f'{len(items)} tiles, no data'}")
    except Exception as e:
        errs.append(f"Copernicus: {type(e).__name__}: {e}"[:200])
    try:
        items = _pc_items("3dep-seamless", bbox_ll)
        dem, err = _load_mosaic(items, "data", gbox, "bilinear", "float32", np.nan)
        if dem is not None and np.isfinite(dem).any():
            dem[(dem < -100) | (dem > 9000)] = np.nan
            return dem, "USGS 3DEP", ""
        errs.append(f"3DEP: {err or 'no data'}")
    except Exception as e:
        errs.append(f"3DEP: {type(e).__name__}: {e}"[:200])
    return None, "", " | ".join(errs)


def load_mukey(gbox, bbox_ll):
    items = _pc_items("gnatsgo-rasters", bbox_ll)
    mk, err = _load_mosaic(items, "mukey", gbox, "nearest", "int64", 0)
    if mk is not None:
        mk[(mk < 0) | (mk >= 2147483647)] = 0
    return mk, err


# =============================================================================== soil attributes
_SDA_DOWN = {"flag": False, "why": ""}


def _sda(sql: str) -> pd.DataFrame:
    """POST a query to USDA Soil Data Access. After one network failure the server is skipped for this session."""
    import requests
    if _SDA_DOWN["flag"]:
        raise RuntimeError(f"Soil Data Access skipped ({_SDA_DOWN['why']})")
    try:
        r = requests.post(SDA_URL, data=json.dumps({"query": sql, "format": "JSON+COLUMNNAME"}),
                          headers={"Content-Type": "application/json"}, timeout=90)
    except requests.exceptions.ConnectionError as e:
        _SDA_DOWN.update(flag=True, why="not reachable from this network")
        raise RuntimeError("Soil Data Access not reachable from this network (DNS / connection)") from e
    if r.status_code != 200:
        raise RuntimeError(f"Soil Data Access HTTP {r.status_code}: {r.text[:160]}")
    tbl = r.json().get("Table", [])
    return pd.DataFrame(tbl[1:], columns=tbl[0]) if tbl else pd.DataFrame()


def summarize_horizons(h: pd.DataFrame, depth=SOIL_DEPTH_CM) -> pd.DataFrame:
    """Rows: mukey, cokey, comppct_r, hzdept_r, hzdepb_r, sand/silt/clay/om/awc, drclassdcd, hydgrpdcd.
    -> one row per mukey: 0-`depth` cm depth-weighted within each component, then component-% weighted."""
    cols = ["mukey", "sand", "silt", "clay", "om", "awc", "aws_0_30cm", "drainage", "hydgrp"]
    if h is None or h.empty:
        return pd.DataFrame(columns=cols)
    h = h.copy()
    for c in ["comppct_r", "hzdept_r", "hzdepb_r", "sandtotal_r", "silttotal_r", "claytotal_r", "om_r", "awc_r"]:
        h[c] = pd.to_numeric(h[c], errors="coerce") if c in h else np.nan
    h["thk"] = (h["hzdepb_r"].clip(upper=depth) - h["hzdept_r"].clip(lower=0)).clip(lower=0).fillna(0)
    props = {"sandtotal_r": "sand", "silttotal_r": "silt", "claytotal_r": "clay", "om_r": "om", "awc_r": "awc"}
    for src, dst in props.items():
        ok = h[src].notna() & (h["thk"] > 0)
        h[dst + "_w"] = np.where(ok, h[src] * h["thk"], 0.0)
        h[dst + "_t"] = np.where(ok, h["thk"], 0.0)
    g = h.groupby(["mukey", "cokey"])
    comp = g[[d + "_w" for d in props.values()] + [d + "_t" for d in props.values()]].sum()
    comp["pct"] = g["comppct_r"].first().fillna(0)
    for d in props.values():
        comp[d] = comp[d + "_w"] / comp[d + "_t"].replace(0, np.nan)
    comp["aws_0_30cm"] = comp["awc_w"].where(comp["awc_t"] > 0)          # cm of water in 0-30 cm
    comp = comp.reset_index()
    out = pd.DataFrame({"mukey": comp["mukey"].unique()}).set_index("mukey")
    for d in list(props.values()) + ["aws_0_30cm"]:
        w = comp["pct"].where(comp[d].notna() & (comp["pct"] > 0), 0.0)
        num = (comp[d].fillna(0) * w).groupby(comp["mukey"]).sum()
        den = w.groupby(comp["mukey"]).sum()
        out[d] = num / den.replace(0, np.nan)
    first = h.groupby("mukey")[["drclassdcd", "hydgrpdcd"]].first().rename(columns={"drclassdcd": "drainage", "hydgrpdcd": "hydgrp"})
    out = out.join(first).reset_index()
    return out[cols]


def _wkt_multipolygon(geoms) -> str:
    """Compact WKT (6 decimals ≈ 0.1 m) of simplified, valid field polygons."""
    from shapely import set_precision, make_valid
    from shapely.geometry import MultiPolygon, Polygon
    parts = []
    for g in geoms:
        if g is None or g.is_empty:
            continue
        g = make_valid(g.simplify(0.00003, preserve_topology=True))
        for p in getattr(g, "geoms", [g]):
            if isinstance(p, Polygon) and not p.is_empty:
                parts.append(p)
    mp = set_precision(MultiPolygon(parts), 1e-6)
    return mp.wkt


def soil_pieces_sda(inner_4326: gpd.GeoSeries, workers: int = 3) -> tuple[pd.DataFrame, str]:
    """Exact soil map-unit area inside each field.
    inner_4326: field interiors (index 0..n-1). Returns (DataFrame[lab, mukey, area_m2], error text)."""
    from shapely import wkt as _wkt
    idx = [i for i, g in enumerate(inner_4326.values) if g is not None and not g.is_empty]
    batches = [idx[i:i + SDA_BATCH] for i in range(0, len(idx), SDA_BATCH)]
    fields5 = gpd.GeoDataFrame({"lab": range(len(inner_4326))}, geometry=inner_4326.values, crs="EPSG:4326").to_crs(GRID_CRS)

    def one(b):
        w = _wkt_multipolygon(inner_4326.values[b])
        sql = ("SELECT P.mukey, P.mupolygongeo.STIntersection(geometry::STGeomFromText('" + w + "', 4326)).STAsText() AS geom "
               "FROM mupolygon AS P WHERE P.mupolygongeo.STIntersects(geometry::STGeomFromText('" + w + "', 4326)) = 1")
        df = _sda(sql)
        if df.empty:
            return pd.DataFrame(columns=["lab", "mukey", "area_m2"])
        g = gpd.GeoDataFrame({"mukey": df["mukey"].astype("int64")}, geometry=[_wkt.loads(t) for t in df["geom"]], crs="EPSG:4326").to_crs(GRID_CRS)
        fb = fields5.iloc[b]
        pairs = gpd.sjoin(g, fb, predicate="intersects", how="inner")
        if pairs.empty:
            return pd.DataFrame(columns=["lab", "mukey", "area_m2"])
        inter = pairs.geometry.values.intersection(fb.loc[pairs["index_right"]].geometry.values)
        return pd.DataFrame({"lab": pairs["lab"].values, "mukey": pairs["mukey"].values, "area_m2": gpd.GeoSeries(inter).area.values})

    out, err = [], ""
    with ThreadPoolExecutor(max_workers=max(1, min(workers, len(batches)))) as ex:
        for fu in [ex.submit(one, b) for b in batches]:
            try:
                out.append(fu.result())
            except Exception as e:
                err = err or f"{type(e).__name__}: {e}"[:200]
    res = pd.concat(out, ignore_index=True) if out else pd.DataFrame(columns=["lab", "mukey", "area_m2"])
    res = res[res["area_m2"] > 1].groupby(["lab", "mukey"], as_index=False)["area_m2"].sum()
    return res, err


def _pc_table(name: str, columns: list[str]) -> pd.DataFrame:
    """One gNATSGO table (parquet on Azure) from Planetary Computer; needs `adlfs`."""
    import planetary_computer as pc
    item = pc.sign(_pc_client().get_collection("gnatsgo-tables").get_item(name))
    asset = item.assets["data"]
    df = pd.read_parquet(asset.href, columns=columns, storage_options=asset.extra_fields.get("table:storage_options", {}))
    for c in ("mukey", "cokey"):
        if c in df:
            df[c] = pd.to_numeric(df[c], errors="coerce").astype("Int64")
    return df


def build_gnatsgo_table(log=print) -> pd.DataFrame:
    """Soil summary (0-30 cm) for EVERY US map unit, from the gNATSGO tables on Planetary Computer.
    One-time (≈ 1-3 min, ~100 MB read); saved to data/ancillary so every later lookup is local."""
    log("Building the national soil table from gNATSGO (one time)…")
    comp = _pc_table("component", ["mukey", "cokey", "comppct_r"])
    hz = _pc_table("chorizon", ["cokey", "hzdept_r", "hzdepb_r", "sandtotal_r", "silttotal_r", "claytotal_r", "om_r", "awc_r"])
    hz = hz[pd.to_numeric(hz["hzdept_r"], errors="coerce") < SOIL_DEPTH_CM]
    try:
        agg = _pc_table("muaggatt", ["mukey", "drclassdcd", "hydgrpdcd"])
    except Exception:
        agg = pd.DataFrame(columns=["mukey", "drclassdcd", "hydgrpdcd"])
    h = hz.merge(comp, on="cokey", how="inner").merge(agg, on="mukey", how="left")
    h["mukey"] = h["mukey"].astype("int64")
    out = summarize_horizons(h)
    out["mukey"] = out["mukey"].astype("int64")
    GNATSGO_TABLE.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(GNATSGO_TABLE, index=False)
    log(f"  saved {len(out):,} map units → {GNATSGO_TABLE.name}")
    return out


def _norm(df: pd.DataFrame) -> pd.DataFrame:
    df = df.reindex(columns=SOIL_COLS)
    df["mukey"] = pd.to_numeric(df["mukey"], errors="coerce").astype("int64")
    return df


def mukey_soil(mukeys, log=print, status: dict | None = None) -> pd.DataFrame:
    """Soil summary per map unit. Order: national gNATSGO table on disk → Soil Data Access (cached)
    → build the national table from Planetary Computer. Always returns SOIL_COLS."""
    status = status if status is not None else {}
    want = pd.Series(sorted({int(m) for m in mukeys if m}), dtype="int64")
    if GNATSGO_TABLE.exists():
        status["props"] = "gNATSGO tables"
        return _norm(pd.read_parquet(GNATSGO_TABLE)).merge(want.rename("mukey"), on="mukey", how="right")
    cache = _norm(pd.read_parquet(MUKEY_PATH)) if MUKEY_PATH.exists() else pd.DataFrame(columns=SOIL_COLS)
    need = sorted(set(want) - set(cache["mukey"].tolist()))
    new, failed = [], False
    for i in range(0, len(need), 500):
        chunk = need[i:i + 500]
        sql = f"""
        SELECT c.mukey, c.cokey, c.comppct_r, h.hzdept_r, h.hzdepb_r, h.sandtotal_r, h.silttotal_r, h.claytotal_r,
               h.om_r, h.awc_r, ma.drclassdcd, ma.hydgrpdcd
        FROM component c
        LEFT JOIN chorizon h ON h.cokey = c.cokey AND h.hzdept_r < {SOIL_DEPTH_CM}
        LEFT JOIN muaggatt ma ON ma.mukey = c.mukey
        WHERE c.mukey IN ({",".join(str(k) for k in chunk)})"""
        try:
            raw = _sda(sql)
        except Exception as e:
            status["sda_err"] = str(e)[:160]; failed = True
            break
        if not raw.empty:
            raw["mukey"] = raw["mukey"].astype("int64")
        s = summarize_horizons(raw) if not raw.empty else pd.DataFrame(columns=SOIL_COLS)
        new.append(pd.DataFrame({"mukey": chunk}).merge(_norm(s), on="mukey", how="left"))
    if new:
        cache = pd.concat([cache] + new, ignore_index=True).drop_duplicates("mukey", keep="last")
        cache.to_parquet(MUKEY_PATH, index=False)
    if failed and C.HOSTED:   # never build the national table on a small hosted machine — SoilGrids fills in
        status["pc_err"] = "Soil Data Access unreachable (hosted: national soil table not built)"
    elif failed:   # SDA unreachable -> same SSURGO data from Planetary Computer, built once
        try:
            tbl = build_gnatsgo_table(log)
            status["props"] = "gNATSGO tables"
            return _norm(tbl).merge(want.rename("mukey"), on="mukey", how="right")
        except Exception as e:
            status["pc_err"] = f"gNATSGO tables: {type(e).__name__}: {e}"[:200]
    else:
        status["props"] = "Soil Data Access"
    return _norm(cache).merge(want.rename("mukey"), on="mukey", how="right")


def soilgrids_fields(gbox, labels: np.ndarray, n: int) -> pd.DataFrame:
    """Global SoilGrids 2.0 (250 m) fallback: 0-30 cm depth-weighted sand / silt / clay (%), OM (%) per field."""
    import rasterio
    from rasterio.vrt import WarpedVRT
    from rasterio.enums import Resampling
    depths = [("0-5", 5), ("5-15", 10), ("15-30", 15)]
    out = {}
    for v in ("sand", "silt", "clay", "soc"):
        acc = np.zeros(gbox.shape, "float64"); wsum = np.zeros(gbox.shape, "float64")
        for d, w in depths:
            with rasterio.open(SOILGRIDS.format(v=v, d=d)) as src, WarpedVRT(
                    src, crs=GRID_CRS, transform=gbox.transform, width=gbox.shape[1], height=gbox.shape[0],
                    resampling=Resampling.bilinear) as vrt:
                a = vrt.read(1).astype("float64")
                nd = src.nodata
            ok = np.isfinite(a) & (a > 0) & ((a != nd) if nd is not None else True)
            acc[ok] += a[ok] * w; wsum[ok] += w
        val = np.where(wsum > 0, acc / np.maximum(wsum, 1e-9), np.nan)
        out[v] = zonal_percentiles(val, labels, n, np.isfinite(val), qs=(50,))[0]
    df = pd.DataFrame({"sand": out["sand"] / 10.0, "silt": out["silt"] / 10.0, "clay": out["clay"] / 10.0,   # g/kg -> %
                       "om": out["soc"] / 100.0 * 1.724})                                                 # dg/kg SOC -> % OM
    return df


# =============================================================================== LANID
def lanid_layers() -> dict[str, Path]:
    """Clipped LANID GeoTIFFs written by scripts/02_get_profile_layers.py."""
    if not LANID_DIR.exists():
        return {}
    found = {}
    for p in LANID_DIR.glob("*.tif"):
        k = p.stem.lower().replace("_ga", "")
        found[k] = p
    return found


def _read_onto(path: Path, gbox, resampling="nearest") -> np.ndarray | None:
    import rasterio
    from rasterio.warp import reproject, Resampling, transform_bounds
    from rasterio.windows import from_bounds
    with rasterio.open(path) as src:
        b = transform_bounds(GRID_CRS, src.crs, *gbox.boundingbox, densify_pts=21)
        win = from_bounds(*b, transform=src.transform).round_offsets().round_lengths()
        try:
            arr = src.read(1, window=win, boundless=True, fill_value=0)
        except Exception:
            return None
        if src.nodata is not None:
            arr = np.where(arr == src.nodata, 0, arr)
        dst = np.zeros(gbox.shape, "float32")
        reproject(arr.astype("float32"), dst, src_transform=src.window_transform(win), src_crs=src.crs,
                  dst_transform=gbox.transform, dst_crs=GRID_CRS, resampling=getattr(Resampling, resampling),
                  src_nodata=None, dst_nodata=None)
    return np.where(dst >= 0.5, dst, 0.0).astype("float32")     # GDAL nudges zeros to ~1e-45


# =============================================================================== per-tile work
def _tile_profile(tile: gpd.GeoDataFrame, log) -> tuple[pd.DataFrame, dict]:
    """Profile for one ~30 km tile of fields. Returns (DataFrame keyed by cache_id, diagnostics)."""
    diag = {"n": len(tile), "elev_src": "", "elev_err": "", "soil_src": "", "soil_err": "", "lanid": False}
    t5 = tile.to_crs(GRID_CRS)
    inner4 = gpd.GeoSeries(tile["geometry_inner"].values, crs="EPSG:4326")
    inner4 = gpd.GeoSeries([gi if (gi is not None and not gi.is_empty) else go for gi, go in zip(inner4.values, tile.geometry.values)], crs="EPSG:4326")
    inner = inner4.to_crs(GRID_CRS)
    gbox = _geobox(t5.total_bounds)
    bbox_ll = tuple(tile.total_bounds)
    n = len(tile)
    labels = _rasterize(inner, gbox)
    small = np.bincount(labels.ravel(), minlength=n + 1)[1:] < 4
    if small.any():                                             # make sure every field has pixels
        lab2 = _rasterize(inner, gbox, all_touched=True)
        for k in np.flatnonzero(small):
            labels[(lab2 == k + 1) & (labels == 0)] = k + 1
    out = pd.DataFrame({"cache_id": tile["cache_id"].values})
    one = np.ones(labels.shape, bool)

    # ---- terrain
    try:
        dem, src, err = load_dem(gbox, bbox_ll)
    except Exception as e:
        dem, src, err = None, "", f"{type(e).__name__}: {e}"[:200]
    diag["elev_src"], diag["elev_err"] = src, err
    if dem is not None:
        gy, gx = np.gradient(dem, RES)
        slope = np.hypot(gx, gy) * 100.0
        mn, pc, _ = zonal_percentiles(dem, labels, n, np.isfinite(dem))
        out["elev_mean"], out["elev_p05"], out["elev_median"], out["elev_p95"] = mn, pc[:, 0], pc[:, 2], pc[:, 4]
        out["elev_range"] = pc[:, 4] - pc[:, 0]
        smn, spc, _ = zonal_percentiles(slope, labels, n, np.isfinite(slope))
        out["slope_mean"], out["slope_median"], out["slope_p90"] = smn, spc[:, 2], spc[:, 3]
        out["dem_source"] = src

    # ---- soil: exact SSURGO polygons (SDA) -> fallback gNATSGO raster
    pairs = pd.DataFrame(columns=["lab", "mukey", "area_m2"])
    try:
        pairs, err = soil_pieces_sda(inner4.reset_index(drop=True))
        if len(pairs):
            diag["soil_src"] = "SSURGO polygons (Soil Data Access)"
        diag["soil_err"] = err
    except Exception as e:
        diag["soil_err"] = f"SDA: {type(e).__name__}: {e}"[:200]
    if pairs.empty:
        try:
            mk, err = load_mukey(gbox, bbox_ll)
            if mk is not None and (mk > 0).any():
                m = (labels > 0) & (mk > 0)
                pairs = pd.DataFrame({"lab": labels[m] - 1, "mukey": mk[m]}).value_counts().rename("area_m2").reset_index()
                pairs["area_m2"] *= RES * RES
                diag["soil_src"] = "gNATSGO raster"
            elif err:
                diag["soil_err"] += f" | gNATSGO: {err}"
        except Exception as e:
            diag["soil_err"] += f" | gNATSGO: {type(e).__name__}: {e}"[:200]
    if len(pairs):
        try:
            st_ = {}
            soil = mukey_soil(pairs["mukey"].unique(), log, st_)
            for k in ("sda_err", "pc_err"):
                if st_.get(k):
                    diag["soil_err"] += f" | {st_[k]}"
            if st_.get("props"):
                diag["soil_src"] += f" + {st_['props']}"
            pp = pairs.merge(soil, on="mukey", how="left")
            rows = []
            for lab, g in pp.groupby("lab"):
                r = {"lab": int(lab), "n_mapunits": len(g), "dom_mukey_share": g["area_m2"].max() / g["area_m2"].sum()}
                for c in ["sand", "silt", "clay", "om", "awc", "aws_0_30cm"]:
                    ok = g[c].notna()
                    r[c] = np.average(g.loc[ok, c], weights=g.loc[ok, "area_m2"]) if ok.any() else np.nan
                for c in ["drainage", "hydgrp"]:
                    ok = g[c].notna()
                    r[c] = g[ok].groupby(c)["area_m2"].sum().idxmax() if ok.any() else None
                rows.append(r)
            out = out.join(pd.DataFrame(rows).set_index("lab"), how="left")
        except Exception as e:
            diag["soil_err"] += f" | soil properties: {type(e).__name__}: {e}"[:200]
    if "sand" not in out or out["sand"].isna().all():          # last resort: global SoilGrids 250 m
        try:
            sg = soilgrids_fields(gbox, labels, n)
            if sg["sand"].notna().any():
                for c in sg:
                    out[c] = sg[c].values
                diag["soil_src"] = "SoilGrids 250 m (global)"
        except Exception as e:
            diag["soil_err"] += f" | SoilGrids: {type(e).__name__}: {e}"[:160]

    # ---- irrigation (LANID)
    layers = lanid_layers()
    if layers:
        try:
            recent = [layers[k] for k in sorted(layers) if k.startswith("lanid20")][-3:]
            acc = [(a >= 0.5).astype("float32") for a in (_read_onto(p, gbox) for p in recent) if a is not None]
            if acc:
                out["lanid_recent"] = zonal_percentiles(np.mean(acc, axis=0), labels, n, one, qs=(50,))[0]
                out["lanid_years"] = ",".join(p.stem[5:9] for p in recent)
            if "irrfreq" in layers:
                a = _read_onto(layers["irrfreq"], gbox)
                if a is not None:
                    out["lanid_freq"] = zonal_percentiles(a, labels, n, one, qs=(50,))[0]
            if "maxirrext" in layers:
                a = _read_onto(layers["maxirrext"], gbox)
                if a is not None:
                    out["lanid_ever"] = zonal_percentiles((a >= 0.5).astype("float32"), labels, n, one, qs=(50,))[0]
            diag["lanid"] = True
        except Exception as e:
            diag["lanid_err"] = f"{type(e).__name__}: {e}"[:200]
    return out, diag


def _finish(df: pd.DataFrame, pivot: pd.Series) -> pd.DataFrame:
    for c in NUM_COLS:
        if c not in df:
            df[c] = np.nan
    for c in ["drainage", "hydgrp"]:
        if c not in df:
            df[c] = None
    df["texture"] = [texture_class(s, c) for s, c in zip(df["sand"], df["clay"])]
    df["texture_group"] = df["texture"].map(TEXTURE_GROUP)
    piv = pivot.reindex(df.index).fillna(False).astype(bool).values
    has_lanid = df["lanid_recent"].notna().values
    lan = (df["lanid_recent"].fillna(0) >= 0.5) | ((df["lanid_ever"].fillna(0) >= 0.6) & (df["lanid_freq"].fillna(0) >= 5))
    df["irr_lanid"] = lan.values & has_lanid
    df["irrigated"] = df["irr_lanid"].values | piv
    df["irr_source"] = np.select([df["irr_lanid"].values & piv, df["irr_lanid"].values, piv, has_lanid],
                                 ["LANID + pivot", "LANID", "pivot shape", "dryland (LANID)"], "dryland (no pivot)")
    df["profile_version"] = PROFILE_VERSION
    return df


def field_profiles(fields: gpd.GeoDataFrame, log=print, progress=None, workers: int = 3, use_cache=True, cancel=None) -> pd.DataFrame:
    """Profile for every row of `fields` (needs field_id, geometry, geometry_inner, pivot_like).
    Returns DataFrame (field_id + profile columns); .attrs["diag"] says which sources answered."""
    progress = progress or (lambda f, m: None)
    f = fields.copy()
    f["cache_id"] = [_cache_id(r) for _, r in f.iterrows()]
    cache = pd.read_parquet(PROFILE_PATH) if (use_cache and PROFILE_PATH.exists()) else pd.DataFrame(columns=["cache_id"])
    if len(cache) and "profile_version" in cache:
        cache = cache[cache["profile_version"] == PROFILE_VERSION]
    todo = f[~f["cache_id"].isin(cache["cache_id"])]
    diag = {"n": len(f), "cached": len(f) - len(todo), "elev_src": set(), "soil_src": set(), "errors": [], "lanid": bool(lanid_layers())}
    progress(0.0, f"Field profiles · {len(f) - len(todo)} cached · {len(todo)} to compute")
    fresh = pd.DataFrame(columns=["cache_id"])
    if len(todo):
        cen = todo.geometry.representative_point()
        key = (np.floor(cen.x.values / TILE_DEG).astype(int).astype(str) + "_" + np.floor(cen.y.values / TILE_DEG).astype(int).astype(str))
        tiles = [g for _, g in todo.groupby(key)]
        done = []
        with ThreadPoolExecutor(max_workers=max(1, min(workers, len(tiles)))) as ex:
            futs = [ex.submit(_tile_profile, t, log) for t in tiles]
            for i, fu in enumerate(futs):
                if cancel is not None and cancel.is_set():          # the user pressed Stop
                    for f_ in futs[i:]:
                        f_.cancel()
                    break
                try:
                    df, d = fu.result()
                    done.append(df)
                    if d["elev_src"]: diag["elev_src"].add(d["elev_src"])
                    if d["soil_src"]: diag["soil_src"].add(d["soil_src"])
                    for k in ("elev_err", "soil_err", "lanid_err"):
                        if d.get(k):
                            diag["errors"].append(f"{k.split('_')[0]}: {d[k]}")
                except Exception as e:
                    diag["errors"].append(f"tile: {type(e).__name__}: {e}"[:220])
                progress((i + 1) / len(futs), f"Field profiles · tile {i + 1}/{len(futs)} · elevation, soil, irrigation")
        if done:
            fresh = pd.concat(done, ignore_index=True)
            piv = todo.set_index("cache_id")["pivot_like"] if "pivot_like" in todo else pd.Series(dtype=bool)
            fresh = _finish(fresh.set_index("cache_id"), piv).reset_index()
            keep = fresh["elev_median"].notna() & fresh["sand"].notna()      # incomplete rows are retried next time
            if use_cache and keep.any():
                PROFILE_PATH.parent.mkdir(parents=True, exist_ok=True)
                pd.concat([cache, fresh[keep]], ignore_index=True).drop_duplicates("cache_id", keep="last").to_parquet(PROFILE_PATH, index=False)
    allp = pd.concat([cache, fresh], ignore_index=True).drop_duplicates("cache_id", keep="last")
    out = f[["field_id", "cache_id"]].merge(allp, on="cache_id", how="left")
    out = _finish(out.set_index(f.index), f["pivot_like"] if "pivot_like" in f else pd.Series(False, index=f.index))
    out = out.drop(columns=["cache_id"]).reset_index(drop=True)
    diag["n_elev"] = int(out["elev_median"].notna().sum())
    diag["n_soil"] = int(out["sand"].notna().sum())
    diag["n_lanid"] = int(out["lanid_recent"].notna().sum())
    diag["elev_src"] = ", ".join(sorted(diag["elev_src"])) or ("cache" if diag["n_elev"] else "")
    diag["soil_src"] = ", ".join(sorted(diag["soil_src"])) or ("cache" if diag["n_soil"] else "")
    diag["errors"] = list(dict.fromkeys(diag["errors"]))[:6]
    out.attrs["diag"] = diag
    return out


# =============================================================================== matching
DEFAULT_MATCH = {"irrigation": "any", "elev_m": 0.0, "slope_pct": 0.0, "texture": "any", "om_pct": 0.0,
                 "aws_cm": 0.0, "drainage": False}


def profile_mask(T: pd.DataFrame, me: pd.Series, s: dict) -> tuple[pd.Series, list[tuple[str, int, str]]]:
    """Boolean mask of fields whose profile matches mine under settings `s`, plus a funnel
    [(filter label, fields dropped, rule text)] applied in order. Unknown values fail an active filter."""
    s = {**DEFAULT_MATCH, **(s or {})}
    ok = pd.Series(True, index=T.index)
    funnel = []

    def apply(label, cond, rule):
        nonlocal ok
        cond = cond.fillna(False).astype(bool)
        dropped = int((ok & ~cond).sum())
        ok &= cond
        funnel.append((label, dropped, rule))

    if s["irrigation"] == "same" and pd.notna(me.get("irrigated")):
        mine = bool(me["irrigated"])
        apply("Irrigation", T["irrigated"].astype("boolean") == mine, "irrigated" if mine else "dryland")
    if s["elev_m"] and pd.notna(me.get("elev_median")):
        apply("Elevation", (T["elev_median"] - me["elev_median"]).abs() <= s["elev_m"], f"{me['elev_median']:.0f} ± {s['elev_m']:.0f} m")
    if s["slope_pct"] and pd.notna(me.get("slope_median")):
        apply("Slope", (T["slope_median"] - me["slope_median"]).abs() <= s["slope_pct"], f"{me['slope_median']:.1f} ± {s['slope_pct']:.1f} %")
    if s["texture"] == "group" and me.get("texture_group"):
        apply("Soil texture", T["texture_group"] == me["texture_group"], f"{me['texture_group']} group")
    elif s["texture"] == "class" and me.get("texture"):
        apply("Soil texture", T["texture"] == me["texture"], str(me["texture"]))
    if s["om_pct"] and pd.notna(me.get("om")):
        apply("Organic matter", (T["om"] - me["om"]).abs() <= s["om_pct"], f"{me['om']:.2f} ± {s['om_pct']:.2f} %")
    if s["aws_cm"] and pd.notna(me.get("aws_0_30cm")):
        apply("Water storage 0–30 cm", (T["aws_0_30cm"] - me["aws_0_30cm"]).abs() <= s["aws_cm"], f"{me['aws_0_30cm']:.1f} ± {s['aws_cm']:.1f} cm")
    if s["drainage"] and me.get("drainage"):
        apply("Drainage", T["drainage"] == me["drainage"], str(me["drainage"]))
    return ok, funnel


# =============================================================================== automatic matching (dashboard toggles)
def elevation_tolerance(values: pd.Series) -> float:
    """'Similar elevation' = within half the local inter-quartile range, kept between 3 and 10 m."""
    v = pd.to_numeric(values, errors="coerce").dropna()
    if len(v) < 5:
        return 5.0
    return float(np.clip(0.5 * (v.quantile(0.75) - v.quantile(0.25)), 3.0, 10.0))


def auto_match(T: pd.DataFrame, me: pd.Series, base: pd.Series, elevation=False, soil=False, irrigation=False):
    """Apply the toggled profile filters on top of `base` (similar planting), in this order:
    irrigation -> elevation -> soil texture (the farmer's two most important matches first). Unknown values fail an active filter.
    Returns (final mask, stages [(key, label, n_after, rule)], reason Series for fields dropped here)."""
    is_me = T["field_id"].eq("MY_FIELD")
    cur = base.fillna(False).astype(bool) | is_me
    reason = pd.Series("", index=T.index)
    stages = []

    def step(key, label, cond, rule):
        nonlocal cur
        cond = cond.fillna(False).astype(bool) | is_me
        reason[cur & ~cond] = key
        cur = cur & cond
        stages.append((key, label, int((cur & ~is_me).sum()), rule))

    if irrigation:
        mine = bool(me.get("irrigated"))
        step("irrigation", "Same irrigation", T["irrigated"].astype("boolean") == mine, "irrigated" if mine else "dryland")
    if elevation:
        if pd.notna(me.get("elev_median")):
            tol = elevation_tolerance(T.loc[base.fillna(False) & ~is_me, "elev_median"])
            step("elevation", "Similar elevation", (T["elev_median"] - me["elev_median"]).abs() <= tol,
                 f"{me['elev_median']:.0f} m ± {tol:.0f} m")
        else:
            stages.append(("elevation", "Similar elevation", int((cur & ~is_me).sum()), "no elevation data for my field — skipped"))
    if soil:
        if me.get("texture_group"):
            step("soil", "Similar soil texture", T["texture_group"] == me["texture_group"], f"{me['texture_group']} ({me.get('texture') or '—'})")
        else:
            stages.append(("soil", "Similar soil texture", int((cur & ~is_me).sum()), "no soil data for my field — skipped"))
    return cur, stages, reason


MATCH_KEYS = ("planting", "irrigation", "elevation", "soil")


def match_flags(T: pd.DataFrame, me: pd.Series, base: pd.Series) -> pd.DataFrame:
    """For every field, does it match mine on each criterion — judged on its own, whatever the sidebar toggles are?
    Same rules as `auto_match` (elevation tolerance from the similar-planting fields). Unknown values = no match.
    Returns DataFrame[field_id, planting, irrigation, elevation, soil] (bool)."""
    is_me = T["field_id"].eq("MY_FIELD")
    out = pd.DataFrame({"field_id": T["field_id"].values}, index=T.index)
    out["planting"] = base.fillna(False).astype(bool) | is_me
    out["irrigation"] = (T["irrigated"].astype("boolean") == bool(me.get("irrigated"))).fillna(False).astype(bool) | is_me
    if pd.notna(me.get("elev_median")):
        tol = elevation_tolerance(T.loc[base.fillna(False) & ~is_me, "elev_median"])
        out["elevation"] = ((T["elev_median"] - me["elev_median"]).abs() <= tol).fillna(False).astype(bool) | is_me
    else:
        out["elevation"] = is_me.copy()
    out["soil"] = ((T["texture_group"] == me["texture_group"]) if isinstance(me.get("texture_group"), str) else pd.Series(False, index=T.index)).fillna(False).astype(bool) | is_me
    return out
