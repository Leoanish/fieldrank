"""
Planting dates for every field in the search area — from the satellite green-up, local weather and the farmer's own field.

The idea
  green-up   what the satellite sees: the day a field's NDVI rises a clear step (+0.10) above its own bare soil
             and keeps rising into a real canopy (cohort.detect_greenup).
  emergence  seedlings out of the ground — too small to see at 10 m.
  planting   before both. How long planting → visible green-up takes is set by temperature (germination and early
             growth run on heat units), so a March corn field needs more days than a May cotton field.

Estimate
  1. Thermal time (growing degree-days, crop base temperature) from planting to visible green-up:
       • calibrated on the farmer's own field: the planting date he entered → his field's observed green-up, when that is
         plausible for the crop — same season, same place, same detector, so systematic errors cancel;
       • otherwise the crop default (GREENUP_GDD).
  2. Each field's planting = walk back from its green-up, adding up daily GDD until that thermal time is reached.
     Without weather data the same is done in days (own-field lag, or the crop default).
  3. Uncertainty (± days) = half the gap between the two clear images around the green-up + 2 days for crop/soil spread.
  4. Date quality, two classes (shown as "well dated" / "roughly dated"):
       well dated     a smooth growth curve fits the field, NDVI and EVI2 agree on the green-up within 6 days, clear
                      images ≤ 15 days apart around it and a real rise → planting date good to about ± 1 week;
       roughly dated  green-up seen, but blurred: cloudy weeks, too few looks for a curve, the two indices disagree,
                      a weak rise or Landsat only → could be off by 1–2 weeks.
     Fields with no visible green-up in the window get no planting date (planted later, too cloudy, or green earlier).
"""
from __future__ import annotations

import datetime as dt

import numpy as np
import pandas as pd

from . import config as C

# Planting → visible green-up (NDVI +0.10 over bare soil at 10 m), °C-days above the crop base temperature.
# Starting values for Georgia (≈ 3 weeks for March corn, ≈ 2–2.5 weeks for May cotton/peanut, ≈ 2 weeks for soybean);
# the farmer's own field re-calibrates them on every run.
GREENUP_GDD = {"corn": 150.0, "cotton": 150.0, "peanut": 190.0, "soybean": 230.0}
GREENUP_DAYS = {"corn": 21, "cotton": 17, "peanut": 17, "soybean": 14}          # fallback without weather
LAG_DAYS_OK = (5, 45)               # plausible planting → visible green-up, days
GDD_OK = (0.4, 2.5)                 # own-field thermal time accepted within this × the crop default
GDD_TYPICAL = (0.6, 1.7)            # … and called a clear confirmation within this × the crop default (≈ 2–3 weeks)
# How much my field's own planting → green-up timing replaces the crop default (adaptive calibration):
# the farmer's date is known; a clearly seen green-up at a typical delay confirms it and teaches the timing for this season,
# place and weather. A blurred green-up or an unusual delay is trusted only partly.
CAL_WEIGHT = {("high", True): 1.0, ("high", False): 0.7, ("low", True): 0.7, ("low", False): 0.4}
HIGH_MAX_GAP = 15                   # clear images ≤ 15 d apart around the green-up → about ± 1 week
HIGH_MIN_RISE = 0.12
HIGH_MIN_MONO = 0.6
AGREE = 6                           # NDVI vs EVI2 green-up, days
LAG_SPREAD_DAYS = 2.0


def plant_confidence(row) -> str | None:
    """'high' = well dated · 'low' = roughly dated · None = no green-up seen in the window.
    Well dated needs all of: a smooth growth curve fitted to the field (not just two points), NDVI and EVI2 giving the
    same green-up within AGREE days (when both fit), clear images ≤ HIGH_MAX_GAP days apart around it, a real rise,
    and Sentinel-2 in the bracket (not Landsat only)."""
    if row.get("greenup_status") != "crossing" or pd.isna(row.get("greenup_date")):
        return None
    return "high" if not _low_reasons(row) else "low"


def _low_reasons(row) -> list[str]:
    r = []
    if row.get("method", "curve") != "curve":
        r.append("no smooth growth curve (too few clear looks on the rise)")
    ag = row.get("agree_days")
    if pd.notna(ag) and ag > AGREE:
        r.append(f"NDVI and EVI2 disagree by {ag:.0f} days")
    gap = row.get("gap_days")
    if pd.isna(gap) or gap > HIGH_MAX_GAP:
        r.append(f"clear images {gap:.0f} days apart around the green-up (clouds)" if pd.notna(gap) else "no clear images around the green-up")
    rise = row.get("rise")
    if pd.isna(rise) or rise < HIGH_MIN_RISE:
        r.append("weak rise")
    if str(row.get("sensors", "")) == "L89":
        r.append("Landsat only")
    return r


def why_low(row) -> str:
    return ", ".join(_low_reasons(row)) if row.get("greenup_status") == "crossing" else ""


def _cum_gdd(wx: pd.DataFrame, crop: str):
    """(dates[np.datetime64 daily], cumulative GDD array) from daily weather."""
    from .weather import daily_gdd
    p = C.CROP_PARAMS[crop]
    w = wx.dropna(subset=["tmax_c", "tmin_c"]).sort_values("date")
    g = daily_gdd(w, p.gdd_base_c, p.gdd_upper_c)
    g = g.groupby(level=0).mean().asfreq("D").interpolate()          # interior gaps only; the record ends at the last real day
    return g.index.values.astype("datetime64[D]"), np.cumsum(g.values)


def _gdd_between(days, cum, d0, d1) -> float | None:
    """GDD accumulated on days d0 … d1-1."""
    i0 = int((np.datetime64(d0, "D") - days[0]).astype(int)); i1 = int((np.datetime64(d1, "D") - days[0]).astype(int))
    if i0 < 0 or i1 - 1 >= len(cum) or i1 <= i0:
        return None
    return float(cum[i1 - 1] - (cum[i0 - 1] if i0 > 0 else 0.0))


def _back(days, cum, gu, target):
    """Latest planting day d with GDD(d … gu-1) ≥ target; None if the weather record is too short."""
    i1 = int((np.datetime64(gu, "D") - days[0]).astype(int))
    if i1 - 1 >= len(cum) or i1 < 1:
        return None
    val = cum[i1 - 1] - target
    if val < 0:
        return None
    j = int(np.searchsorted(cum, val, side="right")) - 1          # last day with cum ≤ val
    d = days[0] + np.timedelta64(j + 1, "D")
    return pd.Timestamp(d).date()


def estimate(det: pd.DataFrame, crop: str, my_planting: dt.date, my_id: str = "MY_FIELD", wx: pd.DataFrame | None = None):
    """Adds est_planting, plant_lo, plant_hi, plant_unc, plant_conf, plant_why, lag_days to `det` (one row per field).
    Returns (det, info)."""
    det = det.copy()
    det["plant_conf"] = [plant_confidence(r) for r in det.to_dict("records")]
    det["plant_why"] = [why_low(r) if c == "low" else "" for r, c in zip(det.to_dict("records"), det["plant_conf"])]
    gu = pd.to_datetime(det["greenup_date"], errors="coerce")
    me = det[det["field_id"] == my_id].iloc[0]
    my_gu = pd.to_datetime(me["greenup_date"]) if me["greenup_status"] == "crossing" else pd.NaT
    my_lag = (my_gu.date() - my_planting).days if pd.notna(my_gu) else None

    days = cum = None
    if wx is not None and len(wx):
        try:
            days, cum = _cum_gdd(wx, crop)
        except Exception:
            days = cum = None
    info = {"method": "degree-days" if cum is not None else "days", "default_gdd": GREENUP_GDD[crop], "default_days": GREENUP_DAYS[crop],
            "my_lag_days": my_lag, "calibrated": False, "cal_weight": 0.0, "cal_state": "default", "warn": ""}
    my_conf = me["plant_conf"]

    # ---- thermal time (or days) planting → visible green-up
    target_gdd, lag_days = GREENUP_GDD[crop], GREENUP_DAYS[crop]
    if my_lag is not None:
        ok_days = LAG_DAYS_OK[0] <= my_lag <= LAG_DAYS_OK[1]
        if cum is not None:
            g_me = _gdd_between(days, cum, my_planting, my_gu.date())
            info["my_gdd"] = g_me
            if ok_days and g_me is not None and GDD_OK[0] * GREENUP_GDD[crop] <= g_me <= GDD_OK[1] * GREENUP_GDD[crop]:
                typical = GDD_TYPICAL[0] * GREENUP_GDD[crop] <= g_me <= GDD_TYPICAL[1] * GREENUP_GDD[crop]
                w = CAL_WEIGHT.get((my_conf, typical), 0.4)
                target_gdd = w * g_me + (1 - w) * GREENUP_GDD[crop]
                info.update(calibrated=True, cal_weight=w)
        elif ok_days:
            typical = GDD_TYPICAL[0] * GREENUP_DAYS[crop] <= my_lag <= GDD_TYPICAL[1] * GREENUP_DAYS[crop]
            w = CAL_WEIGHT.get((my_conf, typical), 0.4)
            lag_days = w * my_lag + (1 - w) * GREENUP_DAYS[crop]
            info.update(calibrated=True, cal_weight=w)
        info["cal_state"] = "confirmed" if info["cal_weight"] >= 0.95 else ("partly" if info["calibrated"] else "default")
        if not info["calibrated"]:
            info["warn"] = (f"My field greened up {my_lag} days {'after' if my_lag >= 0 else 'BEFORE'} the planting date entered — for {crop} "
                            f"this usually takes {LAG_DAYS_OK[0]}–{LAG_DAYS_OK[1]} days. Neighbours' planting dates use the typical "
                            f"{'heat units' if cum is not None else 'timing'} instead; please check the date.")
    info["target_gdd"] = target_gdd if cum is not None else None
    info["lag_days_used"] = lag_days if cum is None else None

    # ---- every field
    est, lags = [], []
    for g_ in gu:
        if pd.isna(g_):
            est.append(None); lags.append(np.nan); continue
        d_ = _back(days, cum, g_.date(), target_gdd) if cum is not None else None
        if d_ is None:                                    # no weather for this date → days
            d_ = (g_ - pd.Timedelta(days=round(lag_days if cum is None else GREENUP_DAYS[crop] * target_gdd / GREENUP_GDD[crop]))).date()
        est.append(d_); lags.append((g_.date() - d_).days)
    det["est_planting"] = est
    det["lag_days"] = lags
    unc = np.maximum(det["gap_days"].astype(float).fillna(20) / 2.0, 1.0) + LAG_SPREAD_DAYS
    det["plant_unc"] = np.where(det["est_planting"].notna(), np.round(unc), np.nan)
    ep = pd.to_datetime(det["est_planting"])
    det["plant_lo"] = (ep - pd.to_timedelta(det["plant_unc"], unit="D")).dt.date
    det["plant_hi"] = (ep + pd.to_timedelta(det["plant_unc"], unit="D")).dt.date

    # ---- my field: satellite-only estimate (crop default timing) next to the date the farmer knows
    if pd.notna(my_gu):
        sat = _back(days, cum, my_gu.date(), GREENUP_GDD[crop]) if cum is not None else None
        sat = sat or (my_gu - pd.Timedelta(days=GREENUP_DAYS[crop])).date()
        mine_ = det.loc[det["field_id"] == my_id].iloc[0]
        info.update(my_sat_planting=sat, my_sat_unc=float(mine_["plant_unc"]), my_conf=mine_["plant_conf"])
    else:
        info.update(my_sat_planting=None, my_sat_unc=None, my_conf=None)
    # ---- planting difference to mine in HEAT-UNIT days: a 10-day gap in a cool March is less crop development than in a hot
    # June, so neighbours are matched on the °C-days between the two planting dates, expressed in this season's typical days
    det["delta_tdays"] = np.nan
    if cum is not None:
        g0 = _gdd_between(days, cum, my_planting - dt.timedelta(days=15), my_planting + dt.timedelta(days=30))
        per_day = (g0 / 45.0) if g0 else None
        if per_day and per_day > 0.5:
            info["gdd_per_day"] = per_day
            td = []
            for d_ in det["est_planting"]:
                if d_ is None or pd.isna(d_):
                    td.append(np.nan); continue
                a_, b_ = (my_planting, d_) if d_ >= my_planting else (d_, my_planting)
                g_ = _gdd_between(days, cum, a_, b_) if a_ != b_ else 0.0
                td.append(np.nan if g_ is None else np.sign((d_ - my_planting).days) * g_ / per_day)
            det["delta_tdays"] = td
    # the farmer's date is what he knows: my field's planting is the entered date
    det.loc[det["field_id"] == my_id, ["est_planting", "plant_lo", "plant_hi"]] = [my_planting, my_planting, my_planting]
    det.loc[det["field_id"] == my_id, "plant_unc"] = 0
    det.loc[det["field_id"] == my_id, "delta_tdays"] = 0.0
    nb = det[det["field_id"] != my_id]
    info["n_high"] = int((nb["plant_conf"] == "high").sum()); info["n_low"] = int((nb["plant_conf"] == "low").sum())
    info["n_none"] = int(nb["plant_conf"].isna().sum())
    return det, info


def weather_for(lon: float, lat: float, planting: dt.date, log=print) -> pd.DataFrame | None:
    """Daily weather around the season (Open-Meteo archive, cached per ~10 km cell). None if unavailable."""
    try:
        from .weather import get_daily
        end = min(dt.date.today() - dt.timedelta(days=1), planting + dt.timedelta(days=100))
        return get_daily(lon, lat, str(planting - dt.timedelta(days=75)), str(end))
    except Exception as e:
        log(f"weather unavailable: {type(e).__name__}: {e}")
        return None
