"""
Command-line pipeline.

  python -m si_ranking.cli fields  --csb data/raw/NationalCSB_2018-2025_rev23.zip
  python -m si_ranking.cli ingest  --lon -83.52 --lat 31.45 --radius-km 15 --start 2026-02-01 --end 2026-11-05
  python -m si_ranking.cli planting --lon -83.52 --lat 31.45 --radius-km 15 --year 2026
  python -m si_ranking.cli ancillary --lon -83.52 --lat 31.45 --radius-km 15
  python -m si_ranking.cli rank --lon -83.52 --lat 31.45 --radius-km 15 --vi NDRE
  python -m si_ranking.cli demo        # synthetic end-to-end run, no network
"""
from __future__ import annotations

import argparse
import datetime as dt

import pandas as pd

from . import config as C


def cmd_fields(a):
    from .fields import load_csb_georgia, prepare_fields, save_fields
    g = load_csb_georgia(a.csb, a.layer)
    g = prepare_fields(g, min_acres=a.min_acres)
    print(g["crop_latest"].value_counts())
    save_fields(g)
    from .fields import build_tiles
    build_tiles(g)


def _aoi(a):
    from .fields import load_fields, fields_in_buffer
    from .imagery import aoi_tag_for
    fields = load_fields()
    sel = fields_in_buffer(fields, a.lon, a.lat, a.radius_km)
    tag = aoi_tag_for(a.lon, a.lat, a.radius_km)
    print(f"{len(sel)} fields within {a.radius_km} km  (tag {tag})")
    return sel, tag


def cmd_ingest(a):
    from .imagery import ingest
    sel, tag = _aoi(a)
    ingest(sel, a.start, a.end, tag, resolution=a.resolution, collection=a.collection, force=a.force)


def cmd_planting(a):
    from .imagery import load_timeseries
    from .planting import estimate_all
    from .weather import get_daily
    sel, tag = _aoi(a)
    ts = load_timeseries(tag)
    wx_cache = {}

    def wx_lookup(lon, lat):
        from .weather import grid_key
        k = grid_key(lon, lat)
        if k not in wx_cache:
            wx_cache[k] = get_daily(lon, lat, f"{a.year}-01-01", f"{a.year}-12-31")
        return wx_cache[k]

    est = estimate_all(ts, sel, wx_lookup, a.year)
    out = C.TS_DIR / f"planting_{tag}_{a.year}.parquet"
    est.to_parquet(out, index=False)
    print(est[["field_id", "crop", "emergence_date", "planting_image", "planting_posterior", "planting_sd_days", "method"]].to_string())
    print(f"-> {out}")


def cmd_ancillary(a):
    from .ancillary import all_ancillary
    sel, tag = _aoi(a)
    anc = all_ancillary(sel, do_elev=not a.no_elev, do_soil=not a.no_soil, lanid=a.lanid)
    print(anc.head())


def cmd_rank(a):
    from .imagery import load_timeseries, latest_observation
    from .ranking import build_table, rank_field, CohortFilters
    from .fields import nearest_field
    sel, tag = _aoi(a)
    ts = load_timeseries(tag)
    latest = latest_observation(ts, a.vi, as_of=a.as_of)
    plant_p = sorted(C.TS_DIR.glob(f"planting_{tag}_*.parquet"))
    planting = pd.read_parquet(plant_p[-1]) if plant_p else None
    anc_p = C.ANC_DIR / "ancillary_merged.parquet"
    anc = pd.read_parquet(anc_p) if anc_p.exists() else None
    table = build_table(sel, latest, planting, anc)
    target = a.field_id or nearest_field(sel, a.lon, a.lat)["field_id"]
    res = rank_field(table, target, CohortFilters(planting_window_days=a.window))
    cols = ["rank", "field_id", "crop_latest", "acres", "vi_median", "SI", "percentile", "planting_posterior", "irrigated"]
    cols = [c for c in cols if c in res["ranked"]]
    print(f"Target {target}: {a.vi}={res['target_vi']:.3f}  ref={res['vi_ref']:.3f}  SI={res['target_SI']:.3f}  "
          f"rank {res['target_rank']}/{res['cohort_size']} (pct {res['target_percentile']:.0f}) obs {res['obs_date']}")
    print(res["ranked"][cols].head(25).to_string())


def cmd_rank_lite(a):
    """On-demand: latest clear Sentinel-2 over the buffer -> VI -> reference -> SI -> rank."""
    import geopandas as gpd
    from .fields import load_fields, fields_in_buffer, nearest_field
    from .lite import rank_buffer, make_target
    fields = load_fields()
    if a.boundary:
        g = gpd.read_file(a.boundary).to_crs("EPSG:4326")
        geom = g.geometry.union_all() if hasattr(g.geometry, "union_all") else g.geometry.unary_union
        crop = a.crop or "cotton"
    else:
        f = nearest_field(fields, a.lon, a.lat)
        geom, crop = f.geometry, (a.crop or f["crop_latest"])
        print(f"Target = CSB field {f['field_id']} ({f['crop_latest']} 2025, {f['acres']:.0f} ac)")
    c = geom.centroid
    tgt = make_target(geom, crop)
    buf = fields_in_buffer(fields, c.x, c.y, a.radius_km)
    buf = buf[~buf.geometry.intersects(geom)]
    allf = gpd.GeoDataFrame(pd.concat([tgt, buf], ignore_index=True), geometry="geometry", crs="EPSG:4326")
    allf["geometry_inner"] = gpd.GeoSeries(allf["geometry_inner"], crs="EPSG:4326")
    res = rank_buffer(allf, "MY_FIELD", crop, a.vi, as_of=a.as_of, max_days=a.max_days)
    t = res.table
    me = t[t["field_id"] == "MY_FIELD"].iloc[0]
    print(f"\nMy field {a.vi}={me['vi_median']:.3f} (obs {me['obs_date']})  reference={res.reference:.3f}  "
          f"SI={me['SI']:.2f}  rank {me['rank']}/{me['n_ranked']}  pct {me['percentile']:.0f}")
    print(t[["rank", "field_id", "acres", "vi_median", "SI", "percentile", "obs_date"]].head(20).to_string(index=False))
    out = C.DATA_DIR / "last_ranking.csv"; t.to_csv(out, index=False); print(f"-> {out}")


def cmd_demo(a):
    from .synthetic import run_demo
    run_demo()


def main():
    p = argparse.ArgumentParser(prog="si_ranking")
    sp = p.add_subparsers(dest="cmd", required=True)

    s = sp.add_parser("fields"); s.add_argument("--csb", required=True); s.add_argument("--layer")
    s.add_argument("--min-acres", type=float, default=None); s.set_defaults(fn=cmd_fields)

    def aoi_args(s):
        s.add_argument("--lon", type=float, required=True); s.add_argument("--lat", type=float, required=True)
        s.add_argument("--radius-km", type=float, default=C.DEFAULT_BUFFER_KM)

    s = sp.add_parser("ingest"); aoi_args(s)
    s.add_argument("--start", required=True); s.add_argument("--end", required=True)
    s.add_argument("--resolution", type=int, default=10); s.add_argument("--collection", default=C.STAC_COLLECTION)
    s.add_argument("--force", action="store_true"); s.set_defaults(fn=cmd_ingest)

    s = sp.add_parser("planting"); aoi_args(s); s.add_argument("--year", type=int, default=dt.date.today().year)
    s.set_defaults(fn=cmd_planting)

    s = sp.add_parser("ancillary"); aoi_args(s); s.add_argument("--no-elev", action="store_true")
    s.add_argument("--no-soil", action="store_true"); s.add_argument("--lanid"); s.set_defaults(fn=cmd_ancillary)

    s = sp.add_parser("rank"); aoi_args(s); s.add_argument("--vi", default="NDVI", choices=list(C.VI_DEFS))
    s.add_argument("--field-id"); s.add_argument("--as-of", type=dt.date.fromisoformat, default=None)
    s.add_argument("--window", type=int, default=C.PLANTING_WINDOW_DAYS); s.set_defaults(fn=cmd_rank)

    s = sp.add_parser("rank-lite", help="on-demand latest-image ranking (no ingest needed)")
    s.add_argument("--lon", type=float); s.add_argument("--lat", type=float)
    s.add_argument("--boundary", help="GeoJSON/KML/GPKG/shapefile of your field (instead of lon/lat)")
    s.add_argument("--crop", choices=C.CROPS); s.add_argument("--vi", default="NDVI", choices=list(C.VI_DEFS))
    s.add_argument("--radius-km", type=float, default=10); s.add_argument("--as-of", type=dt.date.fromisoformat, default=None)
    s.add_argument("--max-days", type=int, default=45); s.set_defaults(fn=cmd_rank_lite)

    s = sp.add_parser("demo"); s.set_defaults(fn=cmd_demo)
    s = sp.add_parser("tiles", help="split data/fields/ga_fields.parquet into small map tiles for fast loading / hosting")
    s.set_defaults(fn=lambda a: __import__("si_ranking.fields", fromlist=["build_tiles"]).build_tiles())

    a = p.parse_args()
    a.fn(a)


if __name__ == "__main__":
    main()
