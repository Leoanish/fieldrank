"""
Step 3 — Tentative planting-date estimation per field.

Method
------
1. Smooth the field's NDVI series onto a daily grid (Whittaker smoother),
   working only with clear observations.
2. Bare-soil baseline = low quantile of the pre-season observations.
3. Emergence = first day the smoothed NDVI exceeds baseline + EMERG_DELTA and
   the series later reaches EMERG_CONFIRM_PEAK (so weeds / cover crop don't
   count). Uncertainty = half the gap between the bracketing real observations.
4. Planting = walk backwards from emergence accumulating crop-specific GDD
   (from daily weather) until the thermal time to emergence is met.
5. Combine with the crop's Georgia planting prior (Gaussian on DOY) in a
   conjugate update -> posterior mean and sd. Report both the raw image-based
   estimate and the posterior; let the user override with a known date.

All thresholds live in config.py and should be calibrated against known
planting dates (UGA variety trials, cooperators, your own plots).
"""
from __future__ import annotations

import datetime as dt
from dataclasses import dataclass, asdict

import numpy as np
import pandas as pd
from scipy import sparse
from scipy.sparse.linalg import spsolve

from . import config as C
from .weather import daily_gdd


@dataclass
class PlantingEstimate:
    field_id: str
    crop: str
    emergence_date: dt.date | None
    emergence_sd_days: float
    planting_image: dt.date | None       # back-calculated from imagery only
    planting_posterior: dt.date | None   # combined with prior
    planting_sd_days: float
    method: str
    n_obs: int
    baseline: float
    peak: float


# ----------------------------------------------------------------------------
# Smoothing
# ----------------------------------------------------------------------------
def whittaker(y: np.ndarray, w: np.ndarray, lam: float = C.SMOOTH_LAMBDA, d: int = 2) -> np.ndarray:
    """Weighted Whittaker smoother (Eilers 2003). w=0 marks missing days."""
    m = len(y)
    E = sparse.identity(m, format="csc")
    D = E[1:] - E[:-1]
    for _ in range(d - 1):
        D = D[1:] - D[:-1]
    W = sparse.diags(w)
    A = (W + lam * D.T @ D).tocsc()
    z = spsolve(A, w * np.nan_to_num(y))
    return np.asarray(z)


def to_daily(series: pd.Series, start: pd.Timestamp, end: pd.Timestamp) -> tuple[pd.DatetimeIndex, np.ndarray, np.ndarray]:
    idx = pd.date_range(start, end, freq="D")
    y = np.full(len(idx), np.nan)
    w = np.zeros(len(idx))
    s = series.groupby(series.index).max()  # if two scenes same day keep max (cloud-robust)
    pos = idx.get_indexer(s.index)
    ok = pos >= 0
    y[pos[ok]] = s.values[ok]
    w[pos[ok]] = 1.0
    return idx, y, w


# ----------------------------------------------------------------------------
# Emergence detection
# ----------------------------------------------------------------------------
def detect_emergence(obs: pd.Series, year: int, crop: str):
    """obs: Series indexed by Timestamp of clear-sky NDVI medians for one field.
    Returns (emergence_date, sd_days, baseline, peak, method)."""
    p = C.CROP_PARAMS[crop]
    obs = obs.dropna().sort_index()
    obs = obs[(obs.index.year == year)]
    if len(obs) < 4:
        return None, np.nan, np.nan, np.nan, "too_few_obs"
    start = pd.Timestamp(year, 1, 1); end = pd.Timestamp(year, 12, 31)
    idx, y, w = to_daily(obs, start, end)
    z = whittaker(y, w)
    doy = np.arange(1, len(idx) + 1)
    win = (doy >= p.emerg_doy_min) & (doy <= p.emerg_doy_max)

    # Baseline from observations before the window opens (plus the first bit of it)
    pre = obs[obs.index.dayofyear <= p.emerg_doy_min + 20]
    baseline = float(np.nanquantile(pre.values, C.BASELINE_QUANTILE)) if len(pre) >= 2 \
        else float(np.nanquantile(obs.values, C.BASELINE_QUANTILE))
    peak = float(np.nanmax(z[win])) if win.any() else float(np.nanmax(z))
    if peak < C.EMERG_CONFIRM_PEAK:
        return None, np.nan, baseline, peak, "no_canopy"

    thr = baseline + C.EMERG_DELTA
    # First day inside the window where the smoothed curve crosses thr and keeps rising to the peak
    peak_i = int(np.nanargmax(np.where(win, z, -np.inf)))
    cand = np.flatnonzero(win & (z >= thr) & (doy <= doy[peak_i]))
    if cand.size == 0:
        return None, np.nan, baseline, peak, "no_crossing"
    # walk back from the peak to the last contiguous run start -> robust to early blips
    i = peak_i
    while i - 1 >= 0 and z[i - 1] >= thr and win[i - 1]:
        i -= 1
    cross_i = i
    # Onset refinement: fit the steep part of the rising limb (thr .. 60 % of rise)
    # and extrapolate back to the baseline. Removes part of the detection lag.
    hi = baseline + 0.6 * (peak - baseline)
    seg = np.flatnonzero(win & (z >= thr) & (z <= hi) & (doy >= doy[cross_i]) & (doy <= doy[peak_i]))
    emerg_i = cross_i
    if seg.size >= 3:
        slope, icpt = np.polyfit(doy[seg], z[seg], 1)
        if slope > 0:
            onset_doy = (baseline - icpt) / slope
            emerg_i = int(np.clip(round(onset_doy) - 1, max(cross_i - 25, 0), cross_i))
    # Uncertainty: half gap between the bracketing real observations
    obs_pos = np.flatnonzero(w > 0)
    before = obs_pos[obs_pos <= cross_i]
    after = obs_pos[obs_pos >= cross_i]
    gap = (after[0] - before[-1]) if (before.size and after.size) else 10
    sd = max(gap / 2.0, 1.5)
    return idx[emerg_i].date(), float(sd), baseline, peak, "threshold_crossing"


# ----------------------------------------------------------------------------
# GDD back-calculation and prior combination
# ----------------------------------------------------------------------------
def backcalc_planting(emergence: dt.date, wx: pd.DataFrame, crop: str) -> tuple[dt.date, float]:
    """Walk backward from emergence accumulating GDD until thermal time is met.
    Returns (planting_date, lag_days)."""
    p = C.CROP_PARAMS[crop]
    target = p.gdd_to_emergence + C.detect_lag_gdd(crop)
    g = daily_gdd(wx, p.gdd_base_c, p.gdd_upper_c)
    g = g[g.index < pd.Timestamp(emergence)].sort_index(ascending=False)
    acc = 0.0
    lag = 0
    for day, val in g.items():
        acc += val
        lag += 1
        if acc >= target:
            return day.date(), lag
        if lag > 60:  # cold spring guard
            break
    return (pd.Timestamp(emergence) - pd.Timedelta(days=lag)).date(), lag


def combine_with_prior(planting_image: dt.date, sd_image: float, crop: str, year: int) -> tuple[dt.date, float]:
    """Gaussian conjugate update on day-of-year."""
    p = C.CROP_PARAMS[crop]
    x = pd.Timestamp(planting_image).dayofyear
    prec = 1 / sd_image ** 2 + 1 / p.prior_doy_sd ** 2
    mean = (x / sd_image ** 2 + p.prior_doy_mean / p.prior_doy_sd ** 2) / prec
    sd = float(np.sqrt(1 / prec))
    date = (pd.Timestamp(year, 1, 1) + pd.Timedelta(days=int(round(mean)) - 1)).date()
    return date, sd


def estimate_planting(field_id: str, crop: str, obs: pd.Series, wx: pd.DataFrame, year: int,
                      sd_lag_days: float = 2.0) -> PlantingEstimate:
    emerg, sd_e, base, peak, method = detect_emergence(obs, year, crop)
    if emerg is None:
        return PlantingEstimate(field_id, crop, None, np.nan, None, None, np.nan, method,
                                int(obs.dropna().size), base, peak)
    plant_img, lag = backcalc_planting(emerg, wx, crop)
    sd_img = float(np.sqrt(sd_e ** 2 + sd_lag_days ** 2))
    post, sd_post = combine_with_prior(plant_img, sd_img, crop, year)
    return PlantingEstimate(field_id, crop, emerg, sd_e, plant_img, post, sd_post, method,
                            int(obs.dropna().size), base, peak)


def estimate_all(ts: pd.DataFrame, fields: pd.DataFrame, wx_lookup, year: int,
                 vi: str = C.EMERG_VI) -> pd.DataFrame:
    """ts: long time-series table; fields: needs field_id, crop_latest, lon, lat.
    wx_lookup(lon, lat) -> daily weather DataFrame."""
    t = ts[ts["vi"] == vi]
    out = []
    for _, f in fields.iterrows():
        s = t[t["field_id"] == f["field_id"]].set_index("date")["median"]
        crop = f["crop_latest"]
        wx = wx_lookup(f["lon"], f["lat"])
        out.append(asdict(estimate_planting(f["field_id"], crop, s, wx, year)))
    return pd.DataFrame(out)
