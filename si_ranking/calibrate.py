"""
Calibrate the planting-date model against known planting dates.

Input CSV: field_id, crop, planting_date (YYYY-MM-DD)  [field_id must exist in
the field table; or give lon,lat and we snap to the nearest field].

What is fitted: per crop, the thermal time between the true planting date
and the satellite-detected green-up (so it folds together planting->emergence
and emergence->detectable canopy). Written to data/calibration.json, which
planting.py picks up automatically.

    python -m si_ranking.calibrate truth.csv --lon -83.52 --lat 31.45 --radius-km 15 --year 2025
"""
from __future__ import annotations

import argparse
import json

import numpy as np
import pandas as pd

from . import config as C
from .planting import detect_emergence
from .weather import daily_gdd


def calibrate(truth: pd.DataFrame, ts: pd.DataFrame, fields: pd.DataFrame, wx_lookup, year: int) -> dict:
    t = ts[ts["vi"] == C.EMERG_VI]
    recs = []
    for _, r in truth.iterrows():
        crop = r["crop"]
        s = t[t["field_id"] == r["field_id"]].set_index("date")["median"]
        emerg, sd, base, peak, method = detect_emergence(s, year, crop)
        if emerg is None:
            continue
        f = fields[fields["field_id"] == r["field_id"]].iloc[0]
        wx = wx_lookup(f["lon"], f["lat"])
        p = C.CROP_PARAMS[crop]
        g = daily_gdd(wx, p.gdd_base_c, p.gdd_upper_c)
        total = float(g[(g.index >= pd.Timestamp(r["planting_date"])) & (g.index < pd.Timestamp(emerg))].sum())
        recs.append({"field_id": r["field_id"], "crop": crop, "gdd_plant_to_detect": total,
                     "lag_days": (pd.Timestamp(emerg) - pd.Timestamp(r["planting_date"])).days})
    df = pd.DataFrame(recs)
    out = {"detect_lag_gdd": {}, "n": {}, "lag_days_median": {}}
    for crop, grp in df.groupby("crop"):
        p = C.CROP_PARAMS[crop]
        out["detect_lag_gdd"][crop] = float(max(np.median(grp["gdd_plant_to_detect"]) - p.gdd_to_emergence, 0))
        out["n"][crop] = int(len(grp))
        out["lag_days_median"][crop] = float(grp["lag_days"].median())
    return out, df


def main():
    from .fields import load_fields, fields_in_buffer
    from .imagery import aoi_tag_for, load_timeseries
    from .weather import get_daily, grid_key
    ap = argparse.ArgumentParser()
    ap.add_argument("truth_csv"); ap.add_argument("--lon", type=float, required=True)
    ap.add_argument("--lat", type=float, required=True); ap.add_argument("--radius-km", type=float, default=C.DEFAULT_BUFFER_KM)
    ap.add_argument("--year", type=int, required=True)
    a = ap.parse_args()
    truth = pd.read_csv(a.truth_csv, parse_dates=["planting_date"])
    fields = fields_in_buffer(load_fields(), a.lon, a.lat, a.radius_km)
    ts = load_timeseries(aoi_tag_for(a.lon, a.lat, a.radius_km))
    cache = {}

    def wx_lookup(lon, lat):
        k = grid_key(lon, lat)
        if k not in cache:
            cache[k] = get_daily(lon, lat, f"{a.year}-01-01", f"{a.year}-12-31")
        return cache[k]

    cal, df = calibrate(truth, ts, fields, wx_lookup, a.year)
    print(df.to_string(index=False)); print(json.dumps(cal, indent=2))
    C.CALIBRATION_FILE.write_text(json.dumps(cal, indent=2))
    print(f"-> {C.CALIBRATION_FILE}")


if __name__ == "__main__":
    main()
