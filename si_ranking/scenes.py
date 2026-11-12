"""
Fast per-field statistics from Sentinel-2 L2A scenes (shared by the green-up step and the Rank tab).

Why it is fast
  1. GDAL tuned for cloud-optimised GeoTIFFs: no directory listings, HTTP/2 multiplexing,
     merged range requests, block cache.
  2. Cloud screen at 60 m first: the SCL mask is read from the COG overview (≈ 1/36 of the
     pixels of a 10 m read). Scenes that are cloudy over the pending fields stop there.
  3. One 10 m read per useful scene: SCL + all needed bands in a single call, bands in parallel.
  4. Several scenes in flight at once (thread pool), results merged in date order.
  5. Per-scene field cache (data/timeseries/scenes/): every field's statistic from every scene
     is stored once. A second farmer nearby, a new rank date or a slightly different buffer
     re-uses everything already read; only new (scene, field) pairs touch the network.
  6. Catalogue cache: STAC searches are done per calendar month on a coarse 0.25° box and kept
     (on disk once the month is over), so green-up, ranking, the scout check and the season view
     share one search; duplicate (re-processed) scenes of the same tile and day are dropped.
  7. Per-field cloud screen: a scene is skipped only if NONE of the pending fields is mostly clear
     at 60 m, and the 10 m read is cut down to the window around the clear fields.
"""
from __future__ import annotations

import datetime as dt
import hashlib
import json
import math
import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
import pandas as pd

from . import config as C

ENGINE_VERSION = "e2"     # e2: 30 m inward edge buffer (values differ from e1, so e1 caches are retired)
SCENE_DIR = C.TS_DIR / "scenes"
GDAL_ENV = {
    "GDAL_DISABLE_READDIR_ON_OPEN": "EMPTY_DIR",
    "CPL_VSIL_CURL_ALLOWED_EXTENSIONS": ".tif,.TIF,.tiff,.jp2",
    "GDAL_HTTP_MULTIPLEX": "YES",
    "GDAL_HTTP_VERSION": "2",
    "GDAL_HTTP_MERGE_CONSECUTIVE_RANGES": "YES",
    "GDAL_INGESTED_BYTES_AT_OPEN": "32768",
    "VSI_CACHE": "TRUE",
    "VSI_CACHE_SIZE": str((32 if C.HOSTED else 256) * 2 ** 20),     # small on the ~1 GB host
    "GDAL_CACHEMAX": "128" if C.HOSTED else "512",
    "GDAL_HTTP_MAX_RETRY": "4",
    "GDAL_HTTP_RETRY_DELAY": "1",
    "AWS_NO_SIGN_REQUEST": "YES",
}


def configure():
    for k, v in GDAL_ENV.items():
        os.environ.setdefault(k, v)
    try:
        import odc.stac
        odc.stac.configure_rio(cloud_defaults=True, **{k: v for k, v in GDAL_ENV.items() if k.startswith(("GDAL_HTTP", "VSI_"))})
    except Exception:
        pass


configure()


# ----------------------------------------------------------------------------- ids + cache
def cache_ids(fields) -> np.ndarray:
    """CSB ids as-is; a user polygon (MY_FIELD) gets a geometry hash, so its values are cached too."""
    out = []
    for fid, geom in zip(fields["field_id"].astype(str), fields["geometry"]):
        out.append("MY_" + hashlib.md5(geom.wkb).hexdigest()[:12] if fid == "MY_FIELD" else fid)
    return np.array(out, dtype=object)


_file_locks: dict[str, threading.Lock] = {}
_file_locks_lock = threading.Lock()


def _lock(path) -> threading.Lock:
    with _file_locks_lock:
        return _file_locks.setdefault(str(path), threading.Lock())


def _cache_path(kind: str, item_id: str):
    return SCENE_DIR / f"{ENGINE_VERSION}_{kind}" / f"{item_id}.parquet"


def _read_cache(kind, item_id) -> pd.DataFrame | None:
    p = _cache_path(kind, item_id)
    if not p.exists():
        return None
    try:
        return pd.read_parquet(p)
    except Exception:
        return None


def _write_cache(kind, item_id, df: pd.DataFrame):
    p = _cache_path(kind, item_id)
    p.parent.mkdir(parents=True, exist_ok=True)
    with _lock(p):
        old = _read_cache(kind, item_id)
        out = df if old is None else pd.concat([old, df], ignore_index=True).drop_duplicates("cache_id", keep="last")
        tmp = p.with_suffix(".tmp")
        out.to_parquet(tmp, index=False)
        os.replace(tmp, p)


# ----------------------------------------------------------------------------- raster helpers
def _load(item, bbox, bands, res, pool=None):
    import odc.stac
    ds = odc.stac.load([item], bands=bands, bbox=bbox, resolution=res, chunks=None, groupby="solar_day",
                       resampling={"scl": "nearest"}, fail_on_error=True, pool=pool)
    return ds.isel(time=0)


class _LabelStore:
    """Field-index rasters per output grid (S2 tiles in different UTM zones get their own).
    Memory-bounded: projected geometries are kept per CRS, rasters only for the last few grids,
    and 10 m reads rasterize just the fields of their own window."""
    MAX_GRIDS = 6

    def __init__(self, inner):
        self.inner = inner; self._d = {}; self._proj = {}; self._l = threading.Lock()
        ok = inner.notna() & ~inner.is_empty
        b = np.full((len(inner), 4), np.nan)
        if ok.any():
            b[ok.values] = inner[ok].bounds.values
        self.b = b                                                    # lon/lat bounds per field

    def bounds(self, idx, pad=0.002):
        """lon/lat box around the fields at positions idx (≈ 200 m padding)."""
        bb = self.b[np.asarray(idx, dtype=int)]
        bb = bb[np.isfinite(bb).all(axis=1)]
        if not len(bb):
            return None
        return (bb[:, 0].min() - pad, bb[:, 1].min() - pad, bb[:, 2].max() + pad, bb[:, 3].max() + pad)

    def windows(self, idx, res_m: float, max_px: int):
        """Split fields `idx` into spatial groups whose read window stays under `max_px` pixels at `res_m`.
        Each field lands (whole) in exactly one window, so a 1000-field search area never becomes one huge read."""
        idx = np.asarray(idx, dtype=int)
        bb = self.b[idx]
        okb = np.isfinite(bb).all(axis=1)
        idx, bb = idx[okb], bb[okb]
        if not len(idx):
            return []
        lat = float(np.nanmean(bb[:, [1, 3]]))
        mx, my = 111_320.0 * max(np.cos(np.radians(lat)), 0.2), 110_570.0       # metres per degree
        box_ = self.bounds(idx)
        px = ((box_[2] - box_[0]) * mx / res_m) * ((box_[3] - box_[1]) * my / res_m)
        if px <= max_px:
            return [idx]
        side = 0.8 * np.sqrt(max_px) * res_m                                     # cell side (m); leaves room for field overhang
        cx = ((bb[:, 0] + bb[:, 2]) / 2 - box_[0]) * mx // side
        cy = ((bb[:, 1] + bb[:, 3]) / 2 - box_[1]) * my // side
        key = cx.astype(np.int64) * 100_000 + cy.astype(np.int64)
        return [idx[key == k] for k in np.unique(key)]

    def _projected(self, crs):
        with self._l:
            g = self._proj.get(crs)
        if g is None:
            g = self.inner.to_crs(crs).values
            with self._l:
                self._proj[crs] = g
        return g

    def get(self, ds, all_touched=False, subset=None):
        from rasterio import features
        gb = ds.odc.geobox
        sub = None if subset is None else np.asarray(subset, dtype=int)
        key = (str(gb.crs), tuple(gb.transform)[:6], tuple(gb.shape), all_touched,
               None if sub is None else hashlib.md5(sub.tobytes()).hexdigest())
        with self._l:
            if key in self._d:
                return self._d[key]
        geoms = self._projected(str(gb.crs))
        pick = range(len(geoms)) if sub is None else sub
        shapes = [(geoms[i], int(i) + 1) for i in pick if geoms[i] is not None and not geoms[i].is_empty]
        dtype = "uint16" if len(geoms) < 65_000 else "int32"
        lab = features.rasterize(shapes, out_shape=tuple(gb.shape), transform=gb.transform, fill=0, dtype=dtype,
                                 all_touched=all_touched) if shapes else np.zeros(tuple(gb.shape), dtype)
        with self._l:
            self._d[key] = lab
            while len(self._d) > self.MAX_GRIDS:                                  # forget the oldest grids
                self._d.pop(next(iter(self._d)))
        return lab


def _zonal(values, labels, n, valid):
    from .imagery import _zonal as z
    return z(values, labels, n, valid)


def _boa_offset(item):
    from .imagery import _boa_offset as b
    return b(item)


# ----------------------------------------------------------------------------- one scene
def _scene(item, ids, bbox, indices: dict, kind: str, labels: _LabelStore, pending: np.ndarray,
           min_pix: int, min_clear: float):
    """Stats for the `pending` field indices in one scene.
    Returns (DataFrame[cache_id, n_pix, <idx>, <idx>_p10, <idx>_p90], status, clear_frac)."""
    cols = ["cache_id", "n_pix"] + [c for k in indices for c in (k, f"{k}_p10", f"{k}_p90")]
    want = ids[pending]
    cached = _read_cache(kind, item.id)
    have = cached[cached["cache_id"].isin(want)] if cached is not None else pd.DataFrame(columns=cols)
    todo = pending[~np.isin(want, have["cache_id"].values)]
    if len(todo) == 0:
        return have, "cached", np.nan

    # ---- 1. cloud screen at 60 m (overview read), per field
    dm = _load(item, bbox, ["scl"], 60)
    lab60 = labels.get(dm, all_touched=True)
    scl60 = dm["scl"].values
    n = len(ids)
    inpend = np.isin(lab60, todo + 1) & (scl60 > 0)       # pending fields inside this tile's footprint
    lp = np.where(inpend, lab60, 0).ravel()
    tot = np.bincount(lp, minlength=n + 1)
    clr = np.bincount(lp, weights=np.isin(scl60, C.SCL_CLEAR).ravel().astype("float64"), minlength=n + 1)
    frac = float(clr[1:].sum() / max(tot[1:].sum(), 1))
    fclear = clr[todo + 1] / np.maximum(tot[todo + 1], 1)
    worth = (tot[todo + 1] > 0) & (fclear >= 0.3)            # ≥ 30 % clear at 60 m → worth a 10 m read
    if not worth.any():
        new = pd.DataFrame({"cache_id": ids[todo], "n_pix": 0})
        for c in cols[2:]:
            new[c] = np.nan
        _write_cache(kind, item.id, new)
        return pd.concat([have, new], ignore_index=True), "cloudy", frac

    # ---- 2. 10 m reads in bounded windows around the clear fields: mask + all bands, field pixels only
    specs = {k: (v if len(v) == 3 else ("nd",) + tuple(v)) for k, v in indices.items()}
    bands = sorted({b for sp in specs.values() for b in sp[1:]})
    off = _boa_offset(item)
    med_ = {k: np.full(n, np.nan) for k in specs}; p10_ = {k: np.full(n, np.nan) for k in specs}; p90_ = {k: np.full(n, np.nan) for k in specs}
    cnt = np.zeros(n, int)
    for grp in labels.windows(todo[worth], 10, C.MAX_WINDOW_PX):
        wbox = labels.bounds(grp)
        ds = _load(item, wbox, ["scl"] + bands, 10, pool=min(4, len(bands) + 1))
        lab = labels.get(ds, subset=grp)
        keep = (lab > 0) & np.isin(ds["scl"].values, C.SCL_CLEAR)
        lab1 = lab[keep]
        R = {b: (ds[b].values[keep].astype("float32") - off) / 10000.0 for b in bands}
        del ds, keep, lab
        valid = np.ones(lab1.shape, bool)
        for b in bands:
            valid &= R[b] > 0
        for name, (op, a, b) in specs.items():
            v = (2.5 * (R[a] - R[b]) / (R[a] + 2.4 * R[b] + 1.0)) if op == "evi2" else (R[a] - R[b]) / (R[a] + R[b] + 1e-6)
            md, q1, q9, c = _zonal(v, lab1, n, valid)
            g_ = c > 0
            med_[name][g_], p10_[name][g_], p90_[name][g_] = md[g_], q1[g_], q9[g_]
            if name == next(iter(specs)):
                cnt[g_] = c[g_]
        del R, lab1, valid
    new = pd.DataFrame({"cache_id": ids[todo]})
    ok = cnt[todo] >= min_pix
    for name in specs:
        new[name] = np.where(ok, med_[name][todo], np.nan)
        new[f"{name}_p10"] = np.where(ok, p10_[name][todo], np.nan)
        new[f"{name}_p90"] = np.where(ok, p90_[name][todo], np.nan)
    new.insert(1, "n_pix", cnt[todo].astype(int))
    _write_cache(kind, item.id, new)
    return pd.concat([have, new], ignore_index=True), "read", frac


# ----------------------------------------------------------------------------- drivers
STAC_DIR = C.TS_DIR / "stac"
_SEARCH_MEM: dict = {}
_SEARCH_LOCK = threading.Lock()
_SEARCH_MEM_MAX = 120 if C.HOSTED else 400          # month searches kept in memory (oldest dropped; disk keeps finished months)


def _remember(key, items):
    with _SEARCH_LOCK:
        _SEARCH_MEM.pop(key, None)
        _SEARCH_MEM[key] = (time.time(), items)
        while len(_SEARCH_MEM) > _SEARCH_MEM_MAX:
            _SEARCH_MEM.pop(next(iter(_SEARCH_MEM)))
SNAP_DEG = 0.25


def _months(a: dt.date, b: dt.date):
    m = dt.date(a.year, a.month, 1)
    while m <= b:
        nxt = dt.date(m.year + (m.month == 12), m.month % 12 + 1, 1)
        yield m, nxt - dt.timedelta(days=1)
        m = nxt


def _month_items(sb, m0: dt.date, m1: dt.date, collection: str | None = None, searcher=None):
    """All scenes of one calendar month over the snapped box — memory, then disk (finished months), then network.
    Items are stored UNSIGNED (Planetary Computer links are signed only when a band is read), so they never expire."""
    collection = collection or C.STAC_COLLECTION
    if searcher is None:
        from .imagery import search_scenes
        searcher = lambda b, s_, e_: search_scenes(b, s_, e_, collection, max_cloud=100)   # noqa: E731
    key = f"{collection}_{sb[0]:.2f}_{sb[1]:.2f}_{sb[2]:.2f}_{sb[3]:.2f}_{m0:%Y-%m}"
    stable = m1 < dt.date.today() - dt.timedelta(days=7)          # month over: no new scenes will appear
    fresh = lambda h: h and (stable or time.time() - h[0] < 1800)   # noqa: E731
    with _SEARCH_LOCK:
        hit = _SEARCH_MEM.get(key)
        if not fresh(hit):                                         # any cached box of the same month that covers this one
            for k_, h_ in _SEARCH_MEM.items():
                b_ = _key_box(k_, collection, m0)
                if b_ and b_[0] <= sb[0] and b_[1] <= sb[1] and b_[2] >= sb[2] and b_[3] >= sb[3] and fresh(h_):
                    hit = h_; break
    if fresh(hit):
        return hit[1]
    path = STAC_DIR / f"{key}.json"
    if stable and not path.exists() and STAC_DIR.exists():
        for f_ in STAC_DIR.glob(f"{collection}_*_{m0:%Y-%m}.json"):
            b_ = _key_box(f_.stem, collection, m0)
            if b_ and b_[0] <= sb[0] and b_[1] <= sb[1] and b_[2] >= sb[2] and b_[3] >= sb[3]:
                path = f_; break
    if stable and path.exists():
        try:
            import pystac
            items = [pystac.Item.from_dict(d) for d in json.loads(path.read_text())]
            _remember(key, items)
            return items
        except Exception:
            pass
    items = searcher(sb, m0.isoformat(), m1.isoformat())
    _remember(key, items)
    if stable and items and all(hasattr(it, "to_dict") for it in items):
        try:
            STAC_DIR.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps([it.to_dict(transform_hrefs=False) for it in items]))
        except Exception:
            pass
    return items


def _key_box(key: str, collection: str, m0: dt.date):
    """(x0, y0, x1, y1) of a cache key of this collection and month, else None."""
    pre, suf = f"{collection}_", f"_{m0:%Y-%m}"
    if not (key.startswith(pre) and key.endswith(suf)):
        return None
    try:
        v = [float(x) for x in key[len(pre):-len(suf)].split("_")]
        return tuple(v) if len(v) == 4 else None
    except ValueError:
        return None


def refresh_recent(days: int = 45):
    """Forget catalogue results for recent months (Update button: pick up images published minutes ago)."""
    cut = (dt.date.today() - dt.timedelta(days=days)).strftime("%Y-%m")
    with _SEARCH_LOCK:
        for k in [k for k in _SEARCH_MEM if k.rsplit("_", 1)[-1] >= cut]:
            _SEARCH_MEM.pop(k, None)


def _search(bbox, start, end, collection: str | None = None, searcher=None, tile_of=None, prefer=None):
    """Scenes intersecting bbox in [start, end], one per tile and day (latest processing wins, or `prefer`).
    The per-field cloud screen decides what is usable, not the tile-wide cloud cover."""
    a, b = pd.Timestamp(start).date(), pd.Timestamp(end).date()
    sb = (math.floor(bbox[0] / SNAP_DEG) * SNAP_DEG, math.floor(bbox[1] / SNAP_DEG) * SNAP_DEG,
          math.ceil(bbox[2] / SNAP_DEG) * SNAP_DEG, math.ceil(bbox[3] / SNAP_DEG) * SNAP_DEG)
    best: dict = {}
    months = list(_months(a, b))
    with ThreadPoolExecutor(max_workers=min(4, len(months))) as ex:          # months in parallel
        chunks = list(ex.map(lambda mm: _month_items(sb, *mm, collection=collection, searcher=searcher), months))
    for items_ in chunks:
        for it in items_:
            d = pd.Timestamp(it.datetime).tz_localize(None).date() if pd.Timestamp(it.datetime).tzinfo else pd.Timestamp(it.datetime).date()
            if not (a <= d <= b):
                continue
            ib = getattr(it, "bbox", None)
            if not ib and getattr(it, "geometry", None):
                from shapely.geometry import shape
                ib = shape(it.geometry).bounds
            if ib and (ib[0] > bbox[2] or ib[2] < bbox[0] or ib[1] > bbox[3] or ib[3] < bbox[1]):
                continue
            pr = getattr(it, "properties", {}) or {}
            tile = (tile_of(it) if tile_of else
                    pr.get("grid:code") or pr.get("s2:mgrs_tile") or (it.id.split("_")[1] if "_" in it.id else it.id))
            k = (tile, d)
            better = (prefer(it) > prefer(best[k])) if (prefer and k in best) else (k not in best or it.id > best[k].id)
            if k not in best or better:
                best[k] = it
    return sorted(best.values(), key=lambda it: it.datetime)


def series(fields, start: dt.date, end: dt.date, indices: dict, kind: str, min_pix: int = C.MIN_PIXELS_10M,
           min_clear: float = 0.10, workers: int | None = None, progress=None, log=print):
    """Every clear look of every field between start and end.
    Returns (long DataFrame[field_id, date, scene_id, n_pix, <indices>...], scenes_found, scenes_with_data)."""
    progress = progress or (lambda f, m: None)
    workers = workers or C.WORKERS
    bbox = tuple(fields.total_bounds)
    ids = cache_ids(fields)
    fid = fields["field_id"].astype(str).values
    items = _search(bbox, start.isoformat(), end.isoformat())
    progress(0.02, f"Sentinel-2 · {len(items)} scenes {start} → {end} · screening clouds at 60 m, {workers} at a time")
    labels = _LabelStore(fields["geometry_inner"])
    allidx = np.arange(len(fields))
    out, used, n_cached, n_cloudy = [], 0, 0, 0
    m = dict(zip(ids, fid))
    ex = ThreadPoolExecutor(max_workers=workers)
    try:
        futs = {ex.submit(_scene, it, ids, bbox, indices, kind, labels, allidx, min_pix, min_clear): it for it in items}
        for i, fu in enumerate(as_completed(futs)):
            it = futs[fu]; date = pd.Timestamp(it.datetime).tz_localize(None).normalize()
            try:
                df, status, frac = fu.result()
            except Exception as e:
                log(f"  skip {it.id}: {e}"); continue
            first = next(iter(indices))
            good = df[df[first].notna()]
            if len(good):
                used += 1
            n_cached += status == "cached"; n_cloudy += status == "cloudy"
            g = good.assign(field_id=good["cache_id"].map(m), date=date, scene_id=it.id)
            out.append(g)
            tag = "from cache" if status == "cached" else (f"{frac:.0%} clear → skipped" if status == "cloudy" else f"{frac:.0%} clear")
            progress(0.05 + 0.93 * (i + 1) / max(len(items), 1), f"Sentinel-2 · {i + 1}/{len(items)} · {date.date()} · {tag} · {len(good)} fields")
    finally:      # Stop pressed (or any error): drop the queued scenes immediately; the few in flight finish into the cache
        ex.shutdown(wait=False, cancel_futures=True)
    log(f"Sentinel-2: {len(items)} scenes · {used} with data · {n_cloudy} cloudy · {n_cached} from cache")
    cols = ["field_id", "date", "scene_id", "n_pix"] + [c for k in indices for c in (k, f"{k}_p10", f"{k}_p90")]
    df = pd.concat(out, ignore_index=True)[cols] if out else pd.DataFrame(columns=cols)
    df = _numeric(df, cols[3:])
    return df.sort_values(["field_id", "date"]).reset_index(drop=True), len(items), used


def latest(fields, as_of: dt.date, max_days: int, indices: dict, kind: str, min_pix: int = C.MIN_PIXELS_10M,
           coverage_target: float = 0.97, min_clear: float = 0.10, workers: int | None = None, progress=None, log=print):
    """Newest clear look per field, walking back from `as_of` in batches of `workers` scenes.
    Returns (DataFrame[field_id, n_pix, obs_date, scene_id, <indices>...], scene ids used)."""
    progress = progress or (lambda f, m: None)
    workers = workers or C.WORKERS
    bbox = tuple(fields.total_bounds)
    ids = cache_ids(fields)
    fid = fields["field_id"].astype(str).values
    start = as_of - dt.timedelta(days=max_days)
    items = _search(bbox, start.isoformat(), as_of.isoformat())
    items.sort(key=lambda it: it.datetime, reverse=True)
    progress(0.02, f"Found {len(items)} scenes {start} → {as_of} · newest first, {workers} at a time")
    labels = _LabelStore(fields["geometry_inner"])
    n = len(fields)
    best: dict[int, dict] = {}
    pos = {c: k for k, c in enumerate(ids)}
    used = []
    first = next(iter(indices))
    for b0 in range(0, len(items), workers):
        if len(best) >= coverage_target * n:
            break
        pending = np.array([k for k in range(n) if k not in best], dtype=int)
        batch = items[b0:b0 + workers]
        with ThreadPoolExecutor(max_workers=len(batch)) as ex:
            res = list(ex.map(lambda it: _safe(_scene, log, it, ids, bbox, indices, kind, labels, pending, min_pix, min_clear), batch))
        for it, r in zip(batch, res):                      # newest first wins
            if r is None:
                continue
            df, status, frac = r
            date = pd.Timestamp(it.datetime).tz_localize(None).normalize().date()
            added = 0
            for rec in df[df[first].notna()].to_dict("records"):
                k = pos[rec["cache_id"]]
                if k in best:
                    continue
                rec.update(field_id=fid[k], obs_date=date, scene_id=it.id)
                best[k] = rec; added += 1
            if added:
                used.append(it.id)
            tag = "from cache" if status == "cached" else (f"{frac:.0%} clear → skipped" if status == "cloudy" else f"{frac:.0%} clear")
            progress(0.05 + 0.9 * len(best) / max(n, 1), f"{date} · {tag} · +{added} fields → {len(best)}/{n} have a clear look")
    cols = ["field_id", "n_pix", "obs_date", "scene_id"] + [c for k in indices for c in (k, f"{k}_p10", f"{k}_p90")]
    df = pd.DataFrame(list(best.values())) if best else pd.DataFrame(columns=cols)
    return _numeric(df.reindex(columns=cols), [c for c in cols if c not in ("field_id", "obs_date", "scene_id")]), used


def _numeric(df: pd.DataFrame, cols) -> pd.DataFrame:
    """Values can arrive as 'object' when cached and fresh rows are concatenated — force floats."""
    for c in cols:
        if c in df:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df


def _safe(fn, log, it, *a):
    try:
        return fn(it, *a)
    except Exception as e:
        log(f"  skip {it.id}: {e}")
        return None


# ============================================================================ Landsat 8/9 (green-up back-up)
# Same design as Sentinel-2: cached monthly searches (unsigned), per-field cloud screen on the 120 m overview,
# one 30 m read (QA + red + NIR) cut to the clear fields' window, several scenes in parallel, per-scene field cache,
# ■ Stop drops queued scenes at once.
LS_COLLECTION = "landsat-c2-l2"
PC_STAC = "https://planetarycomputer.microsoft.com/api/stac/v1"


from functools import lru_cache  # noqa: E402


@lru_cache(maxsize=1)
def _pc_client():
    import pystac_client
    return pystac_client.Client.open(PC_STAC)              # no signing here — links are signed when read


def _ls_searcher(b, s_, e_):
    items = list(_pc_client().search(collections=[LS_COLLECTION], bbox=b, datetime=f"{s_}/{e_}",
                                     query={"platform": {"in": ["landsat-8", "landsat-9"]}}).items())
    return sorted(items, key=lambda it: it.datetime)


def _ls_load(item, bbox, bands, res):
    import odc.stac
    import planetary_computer as pc
    ds = odc.stac.load([item], bands=bands, bbox=bbox, resolution=res, chunks=None, groupby="solar_day",
                       resampling={"qa_pixel": "nearest"}, fail_on_error=True, patch_url=pc.sign, pool=min(3, len(bands)))
    return ds.isel(time=0)


def _ls_clear(qa: np.ndarray):
    qa = qa.astype("uint16")
    return (((qa >> 6) & 1) == 1) & (((qa >> 1) & 1) == 0) & (((qa >> 3) & 1) == 0) & (((qa >> 4) & 1) == 0)


LS_KIND = "l89vi2h"         # Landsat green-up cache: NDVI + EVI2 per field per scene, bandpass-adjusted to Sentinel-2
# NASA HLS bandpass adjustment (MSI → OLI, S2A): OLI ≈ slope · MSI + offset. We go the other way (OLI → MSI-like):
# MSI ≈ (OLI − offset) / slope. https://hls.gsfc.nasa.gov/bandpass-adjustment/
HLS_RED = (0.9765, 0.0009)
HLS_NIR = (0.9983, -0.0001)


def _ls_scene(item, ids, bbox, labels: _LabelStore, min_pix: int):
    from .imagery import _zonal
    cols = ["cache_id", "n_pix", "ndvi", "evi2"]
    cached = _read_cache(LS_KIND, item.id)
    have = cached[cached["cache_id"].isin(ids)] if cached is not None else pd.DataFrame(columns=cols)
    todo = np.flatnonzero(~np.isin(ids, have["cache_id"].values))
    if len(todo) == 0:
        return have, "cached", np.nan
    n = len(ids)
    dm = _ls_load(item, bbox, ["qa_pixel"], 120)                             # 1/16 of the pixels of a 30 m read
    lab = labels.get(dm, all_touched=True)
    qa = dm["qa_pixel"].values
    foot = (qa.astype("uint16") & 1) == 0                                   # not fill
    lp = np.where(np.isin(lab, todo + 1) & foot, lab, 0).ravel()
    tot = np.bincount(lp, minlength=n + 1)
    clr = np.bincount(lp, weights=_ls_clear(qa).ravel().astype("float64"), minlength=n + 1)
    frac = float(clr[1:].sum() / max(tot[1:].sum(), 1))
    worth = (tot[todo + 1] > 0) & (clr[todo + 1] / np.maximum(tot[todo + 1], 1) >= 0.3)
    new = pd.DataFrame({"cache_id": ids[todo], "n_pix": 0, "ndvi": np.nan, "evi2": np.nan})
    if not worth.any():
        _write_cache(LS_KIND, item.id, new)
        return pd.concat([have, new], ignore_index=True), "cloudy", frac
    med = np.full(n, np.nan); med2 = np.full(n, np.nan); cnt = np.zeros(n, int)
    for grp in labels.windows(todo[worth], 30, C.MAX_WINDOW_PX):
        ds = _ls_load(item, labels.bounds(grp), ["qa_pixel", "red", "nir08"], 30)
        lab = labels.get(ds, subset=grp)
        keep = (lab > 0) & _ls_clear(ds["qa_pixel"].values)
        lab1 = lab[keep]
        R = ((ds["red"].values[keep].astype("float32") * 0.0000275 - 0.2) - HLS_RED[1]) / HLS_RED[0]
        N = ((ds["nir08"].values[keep].astype("float32") * 0.0000275 - 0.2) - HLS_NIR[1]) / HLS_NIR[0]
        del ds, keep, lab
        ok_ = (R > 0) & (N > 0)
        md, _, _, c = _zonal((N - R) / (N + R + 1e-6), lab1, n, ok_)
        md2, _, _, _ = _zonal(2.5 * (N - R) / (N + 2.4 * R + 1.0), lab1, n, ok_)
        g_ = c > 0
        med[g_], med2[g_], cnt[g_] = md[g_], md2[g_], c[g_]
    ok = cnt[todo] >= max(C.MIN_PIXELS_10M // 9, 6)
    new["n_pix"] = cnt[todo].astype(int)
    new["ndvi"] = np.where(ok, med[todo], np.nan)
    new["evi2"] = np.where(ok, med2[todo], np.nan)
    _write_cache(LS_KIND, item.id, new)
    return pd.concat([have, new], ignore_index=True), "read", frac


def landsat_series(fields, start: dt.date, end: dt.date, workers: int | None = None, progress=None, log=print):
    """NDVI of every clear Landsat 8/9 look of `fields` between start and end (green-up back-up for Sentinel-2).
    Returns (DataFrame[field_id, date, ndvi, n_pix], scenes_found, scenes_with_data)."""
    progress = progress or (lambda f, m: None)
    workers = workers or C.WORKERS
    bbox = tuple(fields.total_bounds)
    ids = cache_ids(fields)
    m = dict(zip(ids, fields["field_id"].astype(str).values))
    progress(0.0, "Landsat 8/9 · searching…")
    items = _search(bbox, start.isoformat(), end.isoformat(), collection=LS_COLLECTION, searcher=_ls_searcher,
                    tile_of=lambda it: it.id.split("_")[2] if it.id.count("_") >= 3 else it.id,
                    prefer=lambda it: (it.id.endswith("T1"), it.id))
    progress(0.02, f"Landsat · {len(items)} scenes · screening clouds at 120 m, {workers} at a time")
    labels = _LabelStore(fields["geometry_inner"])
    out, used = [], 0
    ex = ThreadPoolExecutor(max_workers=workers)
    try:
        futs = {ex.submit(_ls_scene, it, ids, bbox, labels, C.MIN_PIXELS_10M): it for it in items}
        for i, fu in enumerate(as_completed(futs)):
            it = futs[fu]; date = pd.Timestamp(it.datetime).tz_localize(None).normalize()
            try:
                df, status, frac = fu.result()
            except Exception as e:
                log(f"  skip {it.id}: {e}"); continue
            good = df[pd.to_numeric(df["ndvi"], errors="coerce").notna()]
            used += bool(len(good))
            out.append(good.assign(field_id=good["cache_id"].map(m), date=date))
            tag = "from cache" if status == "cached" else (f"{frac:.0%} clear → skipped" if status == "cloudy" else f"{frac:.0%} clear")
            progress(0.05 + 0.93 * (i + 1) / max(len(items), 1), f"Landsat · {i + 1}/{len(items)} · {date.date()} · {tag} · {len(good)} fields")
    finally:
        ex.shutdown(wait=False, cancel_futures=True)
    cols = ["field_id", "date", "ndvi", "evi2", "n_pix"]
    df = pd.concat(out, ignore_index=True).reindex(columns=cols) if out else pd.DataFrame(columns=cols)
    df = _numeric(df, ["ndvi", "evi2", "n_pix"])
    return df, len(items), used
