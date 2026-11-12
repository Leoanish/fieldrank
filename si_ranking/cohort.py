"""
Planting-window cohort with tiered imagery and a per-field confidence score.

Tier 1  Sentinel-2 L2A (10 m), NDVI, window planting −30 / +60 d.
        Cloud-mask band is read FIRST; scenes with < MIN_SCENE_CLEAR clear
        pixels over the fields are skipped before touching red/NIR.
Tier 2  Landsat 8/9 C2 L2 (30 m) via Planetary Computer — fetched only when
        needed: target not detected, buffer coverage < COVERAGE_TRIGGER, or
        target's bracketing gap > GAP_TRIGGER days.
Tier 3  (not here) Sentinel-1 — see README.

Green-up = first crossing of baseline + GREENUP_DELTA on the rising limb,
linearly interpolated between the bracketing observations.
Confidence 0-1 = 0.5·gap + 0.25·rise + 0.25·monotonic (+0.1 if two sensors
bracket the crossing). Labels: high ≥ 0.7, medium ≥ 0.4, low.

Neighbours whose green-up is within ±window of the TARGET's green-up form the
planting-matched cohort. Low-confidence fields stay in the table but are kept
out of the virtual-reference calculation (ref_ok flag).
"""
from __future__ import annotations

import datetime as dt
import hashlib

import geopandas as gpd
import numpy as np
import pandas as pd

from . import config as C
from .imagery import _rasterize_fields, _zonal  # noqa: F401  (Landsat tier)

CACHE_VERSION = "v9-landsat-harmonized"   # bump when detector/search changes -> old caches ignored
GREENUP_DELTA = 0.10           # rise above bare-soil baseline that marks green-up
GREENUP_CONFIRM_ABS = 0.28     # canopy must reach this NDVI ...
GREENUP_CONFIRM_REL = 0.15     # ... and baseline + this, to count as a real crop
DAYS_BEFORE, DAYS_AFTER = 30, 60   # window around planting (cotton/peanut are slow)
NOMINAL_EMERGENCE_DAYS = 10
MIN_SCENE_CLEAR = 0.10         # skip a scene if < 10 % of field pixels are clear
COVERAGE_TRIGGER = 0.90        # tier 2 if fewer than 90 % of fields detected
GAP_TRIGGER = 12               # tier 2 if target's bracketing gap > 12 d
CONF_MED, CONF_HIGH = 0.4, 0.7
PC_STAC = "https://planetarycomputer.microsoft.com/api/stac/v1"
TS_COLS = ["field_id", "date", "ndvi", "evi2", "n_pix", "sensor"]


def _cache_key(fields: gpd.GeoDataFrame, planting: dt.date) -> str:
    return hashlib.md5((",".join(sorted(fields["field_id"].astype(str))) + str(planting) + CACHE_VERSION).encode()).hexdigest()[:12]


def _load(items, bbox, bands, res, patch=None):
    import odc.stac
    ds = odc.stac.load(items, bands=bands, bbox=bbox, resolution=res, chunks=None, groupby="solar_day",
                       resampling={b: "nearest" for b in bands if b in ("scl", "qa_pixel")}, fail_on_error=False, patch_url=patch)
    return ds.isel(time=0)


# ----------------------------------------------------------------------------- tier 1: Sentinel-2
SHARED_FROM_DAYS = 10   # from planting + 10 d on, read every index at once (shared with ranking / scout check / season)
SHARE_WITH_RANKING = False   # off: "Find comparable fields" reads NDVI only (fastest first step); ranking reads its own dates


def _s2_series(fields, planting, days_before, days_after, log, progress):
    """NDVI for every clear look in the window (fast engine: 60 m cloud screen, parallel scenes, per-scene cache).
    Bare-soil dates read NDVI only; once the crop can be up, the same single read stores all indices in the
    cache the ranking, scout check and season views use — so those never re-open these scenes."""
    from . import scenes
    start = planting - dt.timedelta(days=days_before)
    end = planting + dt.timedelta(days=days_after)
    cut = (min(max(planting + dt.timedelta(days=SHARED_FROM_DAYS), start), end) if SHARE_WITH_RANKING
           else end + dt.timedelta(days=1))
    f1 = 0.35 if cut <= end else 0.95
    progress(0.0, f"Sentinel-2 · searching {start} → {end}…")
    parts, used = [], 0
    if cut > start:
        d1, _, u1 = scenes.series(fields, start, min(cut - dt.timedelta(days=1), end),
                                  {"ndvi": ("nir", "red"), "evi2": ("evi2", "nir", "red")}, kind="ndvi_evi2",
                                  min_pix=C.MIN_PIXELS_10M, min_clear=MIN_SCENE_CLEAR,
                                  progress=lambda f, m: progress(f1 * f, m), log=log)
        parts.append(d1[["field_id", "date", "n_pix", "ndvi", "evi2"]]); used += u1
    if cut <= end:
        d2, _, u2 = scenes.series(fields, cut, end, dict(C.VI_ENGINE), kind=C.VI_KIND,
                                  min_pix=C.MIN_PIXELS_10M, min_clear=MIN_SCENE_CLEAR,
                                  progress=lambda f, m: progress(0.35 + 0.6 * f, m), log=log)
        parts.append(d2[["field_id", "date", "n_pix", "NDVI", "EVI2"]].rename(columns={"NDVI": "ndvi", "EVI2": "evi2"})); used += u2
    df = pd.concat([p_ for p_ in parts if len(p_)], ignore_index=True) if any(len(p_) for p_ in parts) else pd.DataFrame(columns=["field_id", "date", "n_pix", "ndvi", "evi2"])
    ts = pd.DataFrame({"field_id": df["field_id"], "date": pd.to_datetime(df["date"]), "ndvi": pd.to_numeric(df["ndvi"], errors="coerce"),
                       "evi2": pd.to_numeric(df.get("evi2"), errors="coerce"),
                       "n_pix": pd.to_numeric(df["n_pix"], errors="coerce").fillna(0).astype(int), "sensor": "S2"}, columns=TS_COLS)
    return ts, used


# ----------------------------------------------------------------------------- tier 2: Landsat 8/9
def harmonize_landsat(ts: pd.DataFrame, max_days: int = 2):
    """Line Landsat up with Sentinel-2 so mixed curves don't jump between sensors.
    Pairs = the same field seen by both within ±max_days. ≥ 15 pairs: robust line (Theil–Sen) S2 ≈ a + b·Landsat;
    5–14: median difference only; fewer: left as is (the HLS bandpass adjustment is already applied at read time).
    Returns (ts, info) — info per index: pairs, slope, offset, median gap before/after."""
    info = {}
    if ts.empty or not (ts["sensor"] == "L89").any() or not (ts["sensor"] == "S2").any():
        return ts, info
    s2 = ts[ts["sensor"] == "S2"]; ls = ts[ts["sensor"] == "L89"]
    pr = pd.merge_asof(ls.sort_values("date"), s2.sort_values("date")[["date", "field_id", "ndvi", "evi2"]].rename(
        columns={"ndvi": "ndvi_s2", "evi2": "evi2_s2", "date": "date_s2"}), left_on="date", right_on="date_s2", by="field_id",
        direction="nearest", tolerance=pd.Timedelta(days=max_days))
    ts = ts.copy()
    for col in ("ndvi", "evi2"):
        if col not in ts:
            continue
        d = pr[[col, f"{col}_s2"]].dropna()
        n = len(d); a_, b_ = 0.0, 1.0
        if n >= 15 and d[col].std() > 0.02:
            from scipy.stats import theilslopes
            b_, a_, _, _ = theilslopes(d[f"{col}_s2"], d[col])
            if not 0.7 <= b_ <= 1.3:                                  # implausible → fall back to an offset
                b_, a_ = 1.0, float((d[f"{col}_s2"] - d[col]).median())
        elif n >= 5:
            a_ = float((d[f"{col}_s2"] - d[col]).median())
        m = ts["sensor"] == "L89"
        ts.loc[m, col] = a_ + b_ * ts.loc[m, col]
        info[col] = {"pairs": n, "slope": round(float(b_), 3), "offset": round(float(a_), 3),
                     "gap_before": round(float((d[f"{col}_s2"] - d[col]).median()), 3) if n else None,
                     "gap_after": round(float((d[f"{col}_s2"] - (a_ + b_ * d[col])).median()), 3) if n else None}
    return ts, info


def _landsat_series(fields, planting, days_before, days_after, log, progress):
    """Landsat 8/9 NDVI for the window (fast engine: cached monthly search, 120 m per-field cloud screen,
    one 30 m read per useful scene, scenes in parallel, per-scene field cache)."""
    from . import scenes
    try:
        import planetary_computer  # noqa: F401
    except ImportError:
        log("Landsat tier needs: pip install planetary-computer"); return pd.DataFrame(columns=TS_COLS), 0
    start = planting - dt.timedelta(days=days_before)
    end = planting + dt.timedelta(days=days_after)
    try:
        df, n_found, used = scenes.landsat_series(fields, start, end, progress=progress, log=log)
    except Exception as e:
        log(f"Landsat failed: {type(e).__name__}: {e}"); return pd.DataFrame(columns=TS_COLS), 0
    ts = pd.DataFrame({"field_id": df["field_id"], "date": pd.to_datetime(df["date"]), "ndvi": df["ndvi"], "evi2": df.get("evi2"),
                       "n_pix": pd.to_numeric(df["n_pix"], errors="coerce").fillna(0).astype(int), "sensor": "L89"}, columns=TS_COLS)
    log(f"Landsat: {n_found} scenes, {used} used")
    return ts, used


# ----------------------------------------------------------------------------- detection + confidence
ONSET_FRAC = 0.20          # green-up = the day the fitted curve has made 20 % of the field's own rise (≈ base + 0.10 NDVI)
FIT_MAX_NRMSE = 0.20       # a fit is usable when its error is ≤ 20 % of the rise
AGREE_DAYS = 6             # NDVI and EVI2 green-ups this close → the date is confirmed


def _crossing(s: pd.DataFrame, planting: dt.date | None = None) -> dict:
    """Threshold crossing on NDVI (base + 0.10): decides the status (crossing / not emerged / …) and is the fallback date."""
    out = {"greenup_date": None, "greenup_status": "too_few_obs", "gap_days": np.nan, "rise": np.nan, "mono": np.nan,
           "n_obs": 0, "sensors": "", "confidence": 0.0, "conf_label": "none"}
    if len(s) == 0:
        return out
    s = s.dropna(subset=["ndvi"]).sort_values("date")
    if s["date"].duplicated().any():          # same day from two sensors: keep the higher value (fast path skips this)
        s = s.groupby("date", as_index=False).agg(ndvi=("ndvi", "max"), sensor=("sensor", lambda x: "+".join(sorted(set(x)))))
    out["n_obs"] = len(s)
    out["sensors"] = "+".join(sorted(set("+".join(s["sensor"]).split("+")))) if len(s) else ""
    if len(s) < 3:
        return out
    dates, vals = s["date"].values, s["ndvi"].values.astype(float)
    # bare-soil baseline: low quantile of observations up to ~5 d after planting (cover crops in March
    # don't matter because the 20th percentile picks the bare April values)
    pre = vals[dates <= np.datetime64(pd.Timestamp(planting) + pd.Timedelta(days=5))] if planting is not None else vals[:2]
    base = float(np.quantile(pre, 0.2)) if len(pre) >= 2 else float(np.min(vals[:2]))
    thr = base + GREENUP_DELTA
    # the crop's canopy peak can only be AFTER planting; a March cover crop / wheat peak is ignored
    after = np.flatnonzero(dates >= np.datetime64(pd.Timestamp(planting))) if planting is not None else np.arange(len(vals))
    if len(after) == 0:
        out["greenup_status"] = "too_few_obs"; return out
    peak_i = int(after[np.argmax(vals[after])]); peak = float(vals[peak_i])
    rise = peak - base
    out["rise"] = rise
    confirm = max(GREENUP_CONFIRM_ABS, base + GREENUP_CONFIRM_REL)
    if peak < confirm:
        # no crop canopy yet by the end of the window -> planted later than the window can see
        tail = vals[-3:]
        if len(vals) >= 3 and vals[-1] >= thr and np.all(np.diff(tail) > -0.02) and peak_i >= len(vals) - 2:
            out["greenup_status"] = "still_emerging"      # rising at the very end
        else:
            out["greenup_status"] = "not_emerged"         # bare through the window: planted later (or fallow)
        return out
    # walk back from the peak while still above threshold -> start of the crop's rising limb
    i = peak_i
    while i - 1 >= 0 and vals[i - 1] >= thr:
        i -= 1
    if i == 0:
        out.update({"greenup_date": pd.Timestamp(dates[0]).date(), "greenup_status": "already_green", "confidence": 0.2, "conf_label": "low"})
        return out
    frac = (thr - vals[i - 1]) / (vals[i] - vals[i - 1])
    d0, d1 = pd.Timestamp(dates[i - 1]), pd.Timestamp(dates[i])
    gap = (d1 - d0).days
    post = vals[i:i + 5]
    mono = float(np.mean(np.diff(post) >= -0.02)) if len(post) >= 2 else 0.5
    gap_score = float(np.clip((20 - gap) / 15, 0, 1))
    rise_score = float(np.clip((rise - GREENUP_DELTA) / 0.25, 0, 1))
    two_sensors = ("+" in s["sensor"].iloc[i - 1]) or ("+" in s["sensor"].iloc[i]) or (s["sensor"].iloc[i - 1] != s["sensor"].iloc[i])
    conf = float(np.clip(0.5 * gap_score + 0.25 * rise_score + 0.25 * mono + (0.1 if two_sensors else 0.0), 0, 1))
    out.update({"greenup_date": (d0 + (d1 - d0) * frac).date(), "greenup_status": "crossing", "gap_days": gap,
                "mono": mono, "confidence": conf,
                "conf_label": "high" if conf >= CONF_HIGH else "medium" if conf >= CONF_MED else "low"})
    return out




_S_GRID = np.array([2.0, 3.0, 4.0, 5.0, 6.5, 8.0, 10.0, 13.0, 17.0, 22.0])      # curve steepness (days)


def _fit_onset(t: np.ndarray, v: np.ndarray, base: float, peak_i: int):
    """Fit an S-curve  v = b + a / (1 + exp(-(t - t0)/s))  through the looks up to the peak (bare soil → canopy).
    Fast and stable: every (t0, s) on a grid, with b and a solved exactly by least squares, best fit kept.
    Returns (onset_day, nrmse, amplitude) or None. Onset = the day the curve reaches ONSET_FRAC of its rise."""
    tt, vv = t[:peak_i + 1], v[:peak_i + 1]
    ok = np.isfinite(vv)
    tt, vv = tt[ok], vv[ok]
    # start at the bare-soil low point before the crop's rise: anything green before it (a winter cover crop, weeds,
    # last year's stubble greening) is not this crop and would break the S-curve
    if len(vv) >= 3:
        lo = int(np.argmin(vv[:-1]))
        keep_from = max(lo - 1, 0) if lo > 0 and vv[lo - 1] - vv[lo] < 0.05 else lo   # keep one more bare look if it is bare too
        tt, vv = tt[keep_from:], vv[keep_from:]
    if len(tt) < 4:
        return None
    t0s = np.arange(tt[0] - 10, tt[-1] + 3, 0.5)
    T0, S = np.meshgrid(t0s, _S_GRID, indexing="ij")
    X = 1.0 / (1.0 + np.exp(-(tt[None, None, :] - T0[..., None]) / S[..., None]))     # (t0, s, n)
    xm, vm = X.mean(-1), vv.mean()
    cov = ((X - xm[..., None]) * (vv - vm)).mean(-1); var = X.var(-1)
    A = np.where(var > 1e-9, cov / np.maximum(var, 1e-9), 0.0)
    B = vm - A * xm
    sse = ((B[..., None] + A[..., None] * X - vv) ** 2).sum(-1)
    sse = np.where((A >= 0.05) & (np.abs(B - base) <= 0.15), sse, np.inf)          # a real rise, starting near the field's bare soil
    if not np.isfinite(sse).any():
        return None
    i, j = np.unravel_index(np.argmin(sse), sse.shape)
    a_, t0, s_ = float(A[i, j]), float(T0[i, j]), float(S[i, j])
    onset = t0 - s_ * np.log(1 / ONSET_FRAC - 1)
    nrmse = float(np.sqrt(sse[i, j] / len(tt)) / max(a_, 1e-6))
    if onset < tt[0] - 5 or onset > tt[-1]:          # extrapolated before the first look, or never reached
        return None
    return float(onset), nrmse, a_


def detect_greenup(s: pd.DataFrame, planting: dt.date | None = None) -> dict:
    """s: DataFrame[date, ndvi, (evi2), sensor] for one field (green-up window).
    1. NDVI threshold crossing decides whether the crop came up in the window (status) — and is the fallback date.
    2. A smooth S-curve is fitted to NDVI and to EVI2 (less sensitive to soil colour and wetness); green-up = the day the
       curve reaches 20 % of the field's own rise. Both agree within AGREE_DAYS → the date is confirmed."""
    out = _crossing(s[["date", "ndvi", "sensor"]], planting)
    out.update({"method": "crossing" if out["greenup_status"] == "crossing" else "", "onset_ndvi": None, "onset_evi2": None,
                "agree_days": np.nan, "fit_nrmse": np.nan})
    if out["greenup_status"] != "crossing":
        return out
    s = s.dropna(subset=["ndvi"]).sort_values("date")
    if s["date"].duplicated().any():
        agg = {"ndvi": ("ndvi", "max")}
        if "evi2" in s:
            agg["evi2"] = ("evi2", "max")
        s = s.groupby("date", as_index=False).agg(**agg)
    d0 = s["date"].min()
    t = ((s["date"] - d0).dt.total_seconds() / 86400.0).to_numpy()
    pl_t = (pd.Timestamp(planting) - d0).total_seconds() / 86400.0 if planting is not None else -np.inf
    fits = {}
    for col in ("ndvi", "evi2"):
        if col not in s or s[col].notna().sum() < 4:
            continue
        v = s[col].to_numpy(dtype=float)
        pre = v[(t <= pl_t + 5) & np.isfinite(v)]
        base = float(np.quantile(pre, 0.2)) if len(pre) >= 2 else float(np.nanmin(v[:2]))
        after = np.flatnonzero((t >= pl_t) & np.isfinite(v))
        if not len(after):
            continue
        peak_i = int(after[np.nanargmax(v[after])])
        r = _fit_onset(t, v, base, peak_i)
        if r is not None and r[1] <= FIT_MAX_NRMSE:
            fits[col] = r
    day = lambda x: (d0 + pd.Timedelta(days=float(x))).date()
    if "ndvi" in fits:
        out["onset_ndvi"] = day(fits["ndvi"][0])
    if "evi2" in fits:
        out["onset_evi2"] = day(fits["evi2"][0])
    if fits:
        if len(fits) == 2:
            diff = abs(fits["ndvi"][0] - fits["evi2"][0])
            out["agree_days"] = round(diff, 1)
            onset = (fits["ndvi"][0] + fits["evi2"][0]) / 2 if diff <= 2 * AGREE_DAYS else min(fits.values(), key=lambda r: r[1])[0]
        else:
            onset = next(iter(fits.values()))[0]
        out["fit_nrmse"] = round(min(r[1] for r in fits.values()), 3)
        out["greenup_date"] = day(onset); out["method"] = "curve"
        # clear looks around the curve's green-up
        before, after_ = t[t <= onset], t[t > onset]
        if len(before) and len(after_):
            out["gap_days"] = int(round(after_.min() - before.max()))
    return out


def _detect_all(ts: pd.DataFrame, fields: gpd.GeoDataFrame, planting) -> pd.DataFrame:
    recs = []
    groups = {k: g for k, g in ts.groupby("field_id")} if len(ts) else {}
    for fid_ in fields["field_id"]:
        g = groups.get(fid_, pd.DataFrame(columns=["date", "ndvi", "evi2", "sensor"]))
        d = detect_greenup(g.reindex(columns=["date", "ndvi", "evi2", "sensor"]), planting)
        d["field_id"] = fid_; recs.append(d)
    return pd.DataFrame(recs)


# ----------------------------------------------------------------------------- orchestration
def planting_cohort(fields: gpd.GeoDataFrame, target_id: str, planting: dt.date, window_days: int = 10,
                    log=print, use_cache: bool = True, progress=None, allow_landsat: bool = True,
                    days_before: int = DAYS_BEFORE, days_after: int = DAYS_AFTER, match_irrigation: bool = False):
    """Returns (detections DataFrame, info dict, raw NDVI series DataFrame)."""
    progress = progress or (lambda f, m: None)
    key = _cache_key(fields, planting)
    p = C.TS_DIR / f"greenup_{key}.parquet"
    if use_cache and p.exists():
        ts = pd.read_parquet(p)
        if "sensor" not in ts:
            ts["sensor"] = "S2"
        n_s2 = int(ts.loc[ts["sensor"] == "S2", "date"].nunique()); n_l = int(ts.loc[ts["sensor"] == "L89", "date"].nunique())
        progress(1.0, "Loaded green-up series from cache"); log(f"cache: {p.name}")
    else:
        ts, n_s2 = _s2_series(fields, planting, days_before, days_after, log, progress); n_l = 0
    progress(0.96, "Detecting green-up (Sentinel-2)…")
    det = _detect_all(ts, fields, planting)
    tgt = det[det["field_id"] == target_id].iloc[0]
    coverage = float(det["greenup_date"].notna().mean())
    tgt_missing = tgt["greenup_date"] is None or pd.isna(tgt["greenup_date"])
    need_l = allow_landsat and n_l == 0 and (tgt_missing or coverage < COVERAGE_TRIGGER or
                                             (pd.notna(tgt["gap_days"]) and tgt["gap_days"] > GAP_TRIGGER))
    tier2_reason = ""; ls_cal = {}
    if need_l:
        tier2_reason = ("target not detected" if tgt_missing else
                        f"coverage {coverage:.0%} < {COVERAGE_TRIGGER:.0%}" if coverage < COVERAGE_TRIGGER else
                        f"target gap {tgt['gap_days']:.0f} d > {GAP_TRIGGER} d")
        log(f"Tier 2 (Landsat) triggered: {tier2_reason}")
        # only the fields Sentinel-2 could not resolve (and mine) — usually a small part of the area
        need_ids = set(det.loc[det["greenup_date"].isna() | det["gap_days"].gt(GAP_TRIGGER), "field_id"]) | {target_id}
        ls_fields = fields[fields["field_id"].isin(need_ids)].reset_index(drop=True)
        log(f"Landsat for {len(ls_fields)}/{len(fields)} fields")
        ls, n_l = _landsat_series(ls_fields, planting, days_before, days_after, log, progress)
        if len(ls):
            ts = pd.concat([ts, ls], ignore_index=True)
            ts, ls_cal = harmonize_landsat(ts)
            progress(0.96, "Detecting green-up (Sentinel-2 + Landsat)…")
            det = _detect_all(ts, fields, planting)
            tgt = det[det["field_id"] == target_id].iloc[0]
    if len(ts):
        ts.to_parquet(p, index=False)

    anchor_obs = not (tgt["greenup_date"] is None or pd.isna(tgt["greenup_date"]))
    anchor = tgt["greenup_date"] if anchor_obs else planting + dt.timedelta(days=NOMINAL_EMERGENCE_DAYS)
    anchor_src = "target green-up observed" if anchor_obs else f"target green-up not visible; using planting + {NOMINAL_EMERGENCE_DAYS} d"
    gd = pd.to_datetime(det["greenup_date"])
    det["delta_days"] = (gd - pd.Timestamp(anchor)).dt.days
    det["in_window"] = (det["delta_days"].abs().le(window_days) & det["greenup_status"].eq("crossing")) | det["field_id"].eq(target_id)
    if match_irrigation and "pivot_like" in fields:
        irr = fields.set_index("field_id")["pivot_like"].astype(bool)
        det["irrigated"] = det["field_id"].map(irr).fillna(False).astype(bool)
        mine = bool(irr.get(target_id, False))
        det["in_window"] = det["in_window"] & (det["irrigated"].eq(mine) | det["field_id"].eq(target_id))
    det["ref_ok"] = det["in_window"] & (det["confidence"] >= CONF_MED)
    info = {"anchor_greenup": anchor, "anchor_source": anchor_src, "anchor_observed": bool(anchor_obs), "n_scenes": int(n_s2), "n_landsat": int(n_l),
            "tier2": bool(need_l), "tier2_reason": tier2_reason, "landsat_cal": ls_cal,
            "n_detected": int(det["greenup_date"].notna().sum()), "n_in_window": int(det["in_window"].sum()),
            "n_high": int((det["conf_label"] == "high").sum()), "n_medium": int((det["conf_label"] == "medium").sum()),
            "n_low": int((det["conf_label"] == "low").sum()),
            "status_counts": det.loc[det["field_id"] != target_id, "greenup_status"].value_counts().to_dict(),
            "target_conf": float(tgt["confidence"]), "target_conf_label": str(tgt["conf_label"]), "target_gap": tgt["gap_days"]}
    log(f"anchor {anchor} ({anchor_src}); detected {info['n_detected']}/{len(det)}, in window {info['n_in_window']}, "
        f"confidence high/med/low {info['n_high']}/{info['n_medium']}/{info['n_low']}")
    return det, info, ts
