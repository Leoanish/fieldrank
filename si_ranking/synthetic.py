"""
Synthetic end-to-end demo (no network): fake fields around Tifton GA, fake
weather, fake NDVI curves with known planting dates -> run the planting-date
estimator and the ranking engine, and report planting-date error.

    python -m si_ranking.cli demo
"""
from __future__ import annotations

import datetime as dt

import numpy as np
import pandas as pd
import geopandas as gpd
from shapely.geometry import Point

from . import config as C
from .planting import estimate_all
from .ranking import build_table, rank_field, CohortFilters
from .imagery import latest_observation
from .weather import daily_gdd


def synth_weather(year: int, rng) -> pd.DataFrame:
    idx = pd.date_range(f"{year}-01-01", f"{year}-12-31", freq="D")
    doy = idx.dayofyear.values
    tmean = 18 + 10 * np.sin((doy - 110) / 365 * 2 * np.pi)  # south GA-ish annual cycle
    tmean = tmean + rng.normal(0, 2.5, len(idx))
    return pd.DataFrame({"date": idx, "tmax_c": tmean + 6, "tmin_c": tmean - 6, "prcp_mm": 0.0})


def synth_fields(n: int, rng, center=(-83.52, 31.45)) -> gpd.GeoDataFrame:
    lon = center[0] + rng.uniform(-0.12, 0.12, n)
    lat = center[1] + rng.uniform(-0.1, 0.1, n)
    crops = rng.choice(["corn", "cotton", "peanut", "soybean"], n, p=[0.3, 0.35, 0.25, 0.1])
    geoms = [Point(x, y).buffer(0.003) for x, y in zip(lon, lat)]
    g = gpd.GeoDataFrame({"field_id": [f"F{i:03d}" for i in range(n)], "crop_latest": crops,
                          "acres": rng.uniform(15, 120, n), "lon": lon, "lat": lat,
                          "pivot_like": rng.random(n) < 0.5}, geometry=geoms, crs="EPSG:4326")
    g["geometry_inner"] = g.geometry.to_crs(C.EQUAL_AREA_CRS).buffer(-20).to_crs("EPSG:4326")
    return g


def synth_series(planting: dt.date, crop: str, wx: pd.DataFrame, rng, quality: float):
    """Logistic NDVI curve starting at emergence (from true GDD), sampled every 5 d
    with ~45 % cloud loss, noise, and a canopy peak scaled by `quality`."""
    p = C.CROP_PARAMS[crop]
    g = daily_gdd(wx, p.gdd_base_c, p.gdd_upper_c)
    acc = g[g.index >= pd.Timestamp(planting)].cumsum()
    emerg = acc[acc >= p.gdd_to_emergence].index[0]
    days = pd.date_range(f"{planting.year}-01-01", f"{planting.year}-12-31", freq="5D")
    t = (days - emerg).days.values.astype(float)
    base = 0.17
    peak = 0.55 + 0.35 * quality
    rise = base + (peak - base) / (1 + np.exp(-(t - 30) / 8))
    senes = 1 / (1 + np.exp((t - 120) / 10))
    ndvi = base + (rise - base) * senes + rng.normal(0, 0.02, len(days))
    keep = rng.random(len(days)) > 0.45
    return pd.Series(ndvi[keep], index=days[keep]), emerg.date()


def run_demo(n_fields: int = 60, year: int = 2026, seed: int = 7):
    rng = np.random.default_rng(seed)
    wx = synth_weather(year, rng)
    fields = synth_fields(n_fields, rng)
    rows, truth = [], []
    for _, f in fields.iterrows():
        p = C.CROP_PARAMS[f["crop_latest"]]
        doy = int(np.clip(rng.normal(p.prior_doy_mean, 8), 50, 200))
        plant = (pd.Timestamp(year, 1, 1) + pd.Timedelta(days=doy - 1)).date()
        q = rng.random()
        s, emerg = synth_series(plant, f["crop_latest"], wx, rng, q)
        truth.append({"field_id": f["field_id"], "true_planting": plant, "true_emergence": emerg, "quality": q})
        for d, v in s.items():
            for vi, scale in (("NDVI", 1.0), ("GNDVI", 0.9), ("NDRE", 0.6)):
                rows.append({"scene_id": f"S{d:%Y%m%d}", "date": d, "field_id": f["field_id"], "vi": vi,
                             "median": v * scale, "p10": v * scale - 0.05, "p90": v * scale + 0.05, "n_pix": 500})
    ts = pd.DataFrame(rows)
    truth = pd.DataFrame(truth)

    est = estimate_all(ts, fields, lambda lon, lat: wx, year)
    m = est.merge(truth, on="field_id")
    ok = m["planting_posterior"].notna()
    err_post = (pd.to_datetime(m.loc[ok, "planting_posterior"]) - pd.to_datetime(m.loc[ok, "true_planting"])).dt.days
    err_img = (pd.to_datetime(m.loc[ok, "planting_image"]) - pd.to_datetime(m.loc[ok, "true_planting"])).dt.days
    err_em = (pd.to_datetime(m.loc[ok, "emergence_date"]) - pd.to_datetime(m.loc[ok, "true_emergence"])).dt.days
    print(f"\nPlanting-date demo on {n_fields} synthetic fields ({ok.sum()} detected):")
    print(f"  emergence error   : bias {err_em.mean():+.1f} d, MAE {err_em.abs().mean():.1f} d")
    print(f"  planting (image)  : bias {err_img.mean():+.1f} d, MAE {err_img.abs().mean():.1f} d")
    print(f"  planting (posterior): bias {err_post.mean():+.1f} d, MAE {err_post.abs().mean():.1f} d")
    print(m["method"].value_counts().to_string())

    as_of = dt.date(year, 7, 15)
    latest = latest_observation(ts, "NDVI", as_of=as_of)
    anc = fields[["field_id", "pivot_like"]].rename(columns={"pivot_like": "irrigated"})
    table = build_table(fields, latest, est, anc)
    target = fields.loc[fields["crop_latest"] == "cotton", "field_id"].iloc[0]
    res = rank_field(table, target, CohortFilters(planting_window_days=10, same_irrigation=True))
    print(f"\nRanking demo as of {as_of}: target {target} ({res['crop']})  NDVI={res['target_vi']:.3f} "
          f"ref={res['vi_ref']:.3f} SI={res['target_SI']:.2f}  rank {res['target_rank']}/{res['cohort_size']}")
    r = res["ranked"].merge(truth[["field_id", "quality"]], on="field_id")
    print(r[["rank", "field_id", "vi_median", "SI", "percentile", "planting_posterior", "irrigated", "quality"]].head(10).to_string(index=False))
    rho = r[["SI", "quality"]].corr(method="spearman").iloc[0, 1]
    print(f"Spearman(SI, true canopy quality) = {rho:.2f}")
    return m, res


if __name__ == "__main__":
    run_demo()
