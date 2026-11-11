"""
Daily weather (tmax/tmin/precip) per ~10 km grid cell, used for GDD
accumulation in the planting-date back-calculation and as a cohort variable.

Default source: Open-Meteo archive API (ERA5-Land based, ~5-day latency, free,
no key). Swap `fetch_daily` for PRISM / gridMET / Daymet if you prefer; the
rest of the pipeline only needs a DataFrame with date, tmax_c, tmin_c, prcp_mm.
"""
from __future__ import annotations

import pandas as pd
import numpy as np

from . import config as C

OPEN_METEO = "https://archive-api.open-meteo.com/v1/archive"


def grid_key(lon: float, lat: float) -> tuple[float, float]:
    g = C.WX_GRID_DEG
    return (round(np.floor(lon / g) * g + g / 2, 3), round(np.floor(lat / g) * g + g / 2, 3))


def fetch_daily(lon: float, lat: float, start: str, end: str) -> pd.DataFrame:
    import requests
    r = requests.get(OPEN_METEO, params={
        "latitude": lat, "longitude": lon, "start_date": start, "end_date": end,
        "daily": "temperature_2m_max,temperature_2m_min,precipitation_sum",
        "timezone": "America/New_York"}, timeout=60)
    r.raise_for_status()
    d = r.json()["daily"]
    df = pd.DataFrame({"date": pd.to_datetime(d["time"]), "tmax_c": d["temperature_2m_max"],
                       "tmin_c": d["temperature_2m_min"], "prcp_mm": d["precipitation_sum"]})
    return df


def get_daily(lon: float, lat: float, start: str, end: str) -> pd.DataFrame:
    """Cached daily weather for the grid cell containing (lon, lat)."""
    glon, glat = grid_key(lon, lat)
    p = C.WX_DIR / f"wx_{glon}_{glat}.parquet".replace("-", "m")
    if p.exists():
        df = pd.read_parquet(p)
        if df["date"].min() <= pd.Timestamp(start) and df["date"].max() >= pd.Timestamp(end) - pd.Timedelta(days=7):
            return df
    df = fetch_daily(glon, glat, start, end)
    df.to_parquet(p, index=False)
    return df


def daily_gdd(df: pd.DataFrame, base_c: float, upper_c: float = 30.0) -> pd.Series:
    """Standard capped-average GDD (°C-days)."""
    tmax = np.minimum(df["tmax_c"].values, upper_c)
    tmin = np.maximum(df["tmin_c"].values, base_c)
    tmax = np.maximum(tmax, base_c)
    g = (tmax + tmin) / 2.0 - base_c
    return pd.Series(np.clip(g, 0, None), index=df["date"].values)
