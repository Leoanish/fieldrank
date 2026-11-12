"""
Step 1 — Field boundaries for Georgia from USDA NASS Crop Sequence Boundaries (CSB).

CSB is a national GDB (several GB). We read only Georgia rows using an
attribute filter, keep fields whose *latest* CDL year is one of our four
crops, apply size thresholds and an inward buffer, and save GeoParquet.

Usage (after downloading the CSB zip into data/raw):
    python -m si_ranking.cli fields --csb data/raw/NationalCSB_2018-2025_rev23.zip
"""
from __future__ import annotations

import re
import zipfile
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import pyogrio

from . import config as C


def _find_gdb(path: Path) -> str:
    """Return an OGR-openable path for a .gdb folder, or a .gdb inside a .zip."""
    path = Path(path)
    if path.suffix.lower() == ".gdb" or path.is_dir():
        return str(path)
    if path.suffix.lower() == ".zip":
        with zipfile.ZipFile(path) as z:
            gdbs = sorted({n.split("/")[0] for n in z.namelist() if ".gdb" in n})
        if not gdbs:
            raise FileNotFoundError(f"No .gdb inside {path}")
        return f"/vsizip/{path}/{gdbs[0]}"
    return str(path)


def _cdl_columns(cols) -> list[str]:
    return sorted([c for c in cols if re.fullmatch(r"CDL\d{4}", c)], key=lambda c: int(c[3:]))


def load_csb_georgia(csb_path: Path | str, layer: str | None = None) -> gpd.GeoDataFrame:
    src = _find_gdb(Path(csb_path))
    layers = pyogrio.list_layers(src)
    if layer is None:
        layer = layers[0][0]
    info = pyogrio.read_info(src, layer=layer)
    cols = list(info["fields"])
    cdl_cols = _cdl_columns(cols)
    if not cdl_cols:
        raise ValueError(f"No CDL#### columns found in layer {layer}: {cols}")
    state_col = next((c for c in cols if c.upper() in ("STATEFIPS", "STATE_FIPS", "STATE")), None)
    where = f"{state_col} = '{C.STATE_FIPS}'" if state_col else None
    print(f"Reading layer '{layer}' from {src}\n  CDL years: {cdl_cols[0]}..{cdl_cols[-1]}  filter: {where}")
    gdf = pyogrio.read_dataframe(src, layer=layer, where=where,
                                 bbox=None if where else C.GA_BBOX)
    gdf = gpd.GeoDataFrame(gdf)
    return gdf


def prepare_fields(gdf: gpd.GeoDataFrame, crops: list[str] | None = None,
                   min_acres: float | None = None) -> gpd.GeoDataFrame:
    """Every field that grew a target crop in ANY year 2018–2025 (incl. double crops), with its crop for each year
    (crop_2018 … crop_2025), so past seasons compare against the fields that really grew the crop that year."""
    crops = crops or C.CROPS
    cdl_cols = _cdl_columns(gdf.columns)
    latest = cdl_cols[-1]
    codes = [k for k, v in C.SUMMER_CROP.items() if v in crops]

    g = gdf[gdf[cdl_cols].isin(codes).any(axis=1)].copy()
    for col in cdl_cols:
        g[f"crop_{col[3:]}"] = g[col].map(lambda v: C.SUMMER_CROP.get(int(v), "other") if pd.notna(v) else "other")
        g[f"dbl_{col[3:]}"] = g[col].isin(list(C.DOUBLE_CROP))
    g["crop_latest"] = g[f"crop_{latest[3:]}"]
    g["cdl_year_latest"] = int(latest[3:])
    # Rotation history string, e.g. "corn>cotton>peanut>cotton" (double crops shown as "wheat/soybean")
    def _lab(v):
        v = int(v)
        if v in C.DOUBLE_CROP:
            return "grain/" + C.DOUBLE_CROP[v]
        return C.CODE_TO_CROP.get(v, str(v))
    g["rotation"] = g[cdl_cols].apply(lambda r: ">".join(_lab(v) for v in r), axis=1)
    g["n_years_crop"] = (g[cdl_cols].isin(codes)).sum(axis=1)

    # Geometry in equal-area CRS for area / buffer, keep WGS84 copy for the app
    ga = g.to_crs(C.EQUAL_AREA_CRS)
    ga["area_ha"] = ga.geometry.area / 1e4
    ga["acres"] = ga["area_ha"] * 2.47105
    ga["geometry_inner"] = inner_buffer(ga.geometry)
    ga = ga[~ga["geometry_inner"].is_empty]
    ga["inner_ha"] = ga["geometry_inner"].area / 1e4

    # Center-pivot heuristic: Polsby-Popper circularity 4πA/P²
    pp = 4 * np.pi * ga.geometry.area / (ga.geometry.length ** 2)
    ga["circularity"] = pp
    ga["pivot_like"] = (pp > 0.80) & (ga["acres"] > 20)

    thr = min_acres if min_acres is not None else min(p.min_acres_10m for p in C.CROP_PARAMS.values())
    ga = ga[ga["acres"] >= thr]

    cent = ga.geometry.centroid.to_crs("EPSG:4326")
    ga["lon"] = cent.x.values
    ga["lat"] = cent.y.values
    id_col = next((c for c in ga.columns if c.upper() == "CSBID"), None)
    ga["field_id"] = ga[id_col].astype(str) if id_col else ("GA" + ga.index.astype(str))

    keep = ["field_id", "crop_latest", "cdl_year_latest", "rotation", "n_years_crop",
            "acres", "area_ha", "inner_ha", "circularity", "pivot_like", "lon", "lat"] + \
        [c for c in ga.columns if re.fullmatch(r"(crop|dbl)_\d{4}", c)]
    cnty = next((c for c in ga.columns if c.upper() in ("CNTY", "COUNTY", "COUNTYFIPS")), None)
    if cnty:
        ga["county"] = ga[cnty].astype(str)
        keep.append("county")
    out = gpd.GeoDataFrame(ga[keep + ["geometry_inner"]].copy(), geometry=ga.geometry.to_crs("EPSG:4326"), crs="EPSG:4326")
    out["geometry_inner"] = gpd.GeoSeries(out["geometry_inner"], crs=C.EQUAL_AREA_CRS).to_crs("EPSG:4326")
    return out.reset_index(drop=True)


def save_fields(gdf: gpd.GeoDataFrame, name: str = "ga_fields.parquet") -> Path:
    path = C.FIELDS_DIR / name
    # GeoParquet supports one geometry column natively; store the inner one as WKB
    g = gdf.copy()
    g["geometry_inner_wkb"] = g["geometry_inner"].apply(lambda s: s.wkb)
    g = g.drop(columns=["geometry_inner"])
    g.to_parquet(path)
    print(f"Saved {len(g):,} fields -> {path}")
    return path


def load_fields(name: str = "ga_fields.parquet") -> gpd.GeoDataFrame:
    from shapely import wkb
    g = gpd.read_parquet(C.FIELDS_DIR / name)
    g["geometry_inner"] = gpd.GeoSeries(g["geometry_inner_wkb"].apply(wkb.loads), crs="EPSG:4326")
    return g.drop(columns=["geometry_inner_wkb"])


def fields_in_buffer(fields: gpd.GeoDataFrame, lon: float, lat: float, radius_km: float,
                     crop: str | None = None, crop_col: str = "crop_latest") -> gpd.GeoDataFrame:
    """Fields whose centroid lies within radius_km of (lon, lat)."""
    from shapely.geometry import Point
    pt = gpd.GeoSeries([Point(lon, lat)], crs="EPSG:4326").to_crs(C.EQUAL_AREA_CRS).iloc[0]
    cents = fields.geometry.to_crs(C.EQUAL_AREA_CRS).centroid
    d = cents.distance(pt) / 1000.0
    sel = fields[d <= radius_km].copy()
    sel["dist_km"] = d[d <= radius_km].values
    if crop:
        sel = sel[sel[crop_col if crop_col in sel else "crop_latest"] == crop]
    return sel.sort_values("dist_km")


def nearest_field(fields: gpd.GeoDataFrame, lon: float, lat: float) -> pd.Series:
    from shapely.geometry import Point
    pt = Point(lon, lat)
    hit = fields[fields.contains(pt)]
    if len(hit):
        return hit.iloc[0]
    pt_ea = gpd.GeoSeries([pt], crs="EPSG:4326").to_crs(C.EQUAL_AREA_CRS).iloc[0]
    d = fields.geometry.to_crs(C.EQUAL_AREA_CRS).distance(pt_ea)
    return fields.loc[d.idxmin()]


# ============================================================================ tiled store (hosting)
# The statewide table is split into 0.25° tiles by field centroid, without the stored inward buffer
# (it was ~80 % of the bytes and is rebuilt in milliseconds for the few hundred fields actually used).
# A small index (id, crop, acres, centroid, tile) answers "which fields are near here" without touching
# any geometry, so the app only ever reads the 1–6 tiles around the chosen field.
import math
from functools import lru_cache

TILE_DEG = 0.25
TILE_X0, TILE_Y0 = -86.0, 30.0
TILES_DIR = C.FIELDS_DIR / "tiles"


def tile_name(lon: float, lat: float) -> str:
    return f"t{int(math.floor((lon - TILE_X0) / TILE_DEG)):02d}_{int(math.floor((lat - TILE_Y0) / TILE_DEG)):02d}"


def build_tiles(src: Path | str | gpd.GeoDataFrame | None = None, out_dir: Path | None = None, grid: float = 1e-6) -> Path:
    """ga_fields.parquet → data/fields/tiles/{index.parquet, tXX_YY.parquet} (zstd, ~1 m-free precision trim)."""
    import shapely
    out_dir = Path(out_dir or TILES_DIR); out_dir.mkdir(parents=True, exist_ok=True)
    g = src if isinstance(src, gpd.GeoDataFrame) else gpd.read_parquet(src or C.FIELDS_DIR / "ga_fields.parquet")
    g = g.drop(columns=[c for c in ("geometry_inner", "geometry_inner_wkb") if c in g])
    g["geometry"] = shapely.set_precision(g.geometry.values, grid)
    g = g[~g.geometry.is_empty]
    g["tile"] = [tile_name(x, y) for x, y in zip(g["lon"], g["lat"])]
    for c in ("acres", "area_ha", "inner_ha", "circularity"):
        if c in g:
            g[c] = g[c].astype("float32")
    for t, part in g.groupby("tile"):
        part.drop(columns="tile").reset_index(drop=True).to_parquet(out_dir / f"{t}.parquet", compression="zstd")
    idx = pd.DataFrame({"field_id": g["field_id"].astype(str), "crop_latest": g["crop_latest"].astype("category"),
                        "acres": g["acres"].astype("float32"), "lon": g["lon"], "lat": g["lat"], "tile": g["tile"].astype("category")})
    for c in sorted(c for c in g.columns if re.fullmatch(r"crop_\d{4}", c)):
        idx[c] = g[c].astype("category").values
    idx.to_parquet(out_dir / "index.parquet", compression="zstd")
    print(f"{len(g):,} fields → {g['tile'].nunique()} tiles in {out_dir}")
    return out_dir


def _shrink(ga: gpd.GeoSeries, d: float) -> gpd.GeoSeries:
    """Inward buffer by d metres (works with every geopandas/shapely 2 version)."""
    import shapely
    return gpd.GeoSeries(shapely.buffer(np.asarray(ga.values), -d, quad_segs=2),
                         index=ga.index, crs=ga.crs)


def inner_buffer(ga: gpd.GeoSeries) -> gpd.GeoSeries:
    """Shrink fields (equal-area CRS) so edge pixels — tree lines, roads, ditches, mixed border pixels — never enter the median.
    30 m inward by default; small or narrow fields that would lose most of their area fall back to 20 m, then 10 m."""
    out = _shrink(ga, C.INWARD_BUFFER_M)
    bad = out.is_empty | (out.area < 0.30 * ga.area)
    for d_, keep in ((20.0, 0.20), (10.0, 0.0)):
        if not bad.any():
            break
        alt_ = _shrink(ga[bad], d_)
        out[bad] = alt_
        bad = out.is_empty | (out.area <= keep * ga.area)
    return out


def _add_inner(g: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    ga = g.geometry.to_crs(C.EQUAL_AREA_CRS)
    g["geometry_inner"] = gpd.GeoSeries(inner_buffer(ga), crs=C.EQUAL_AREA_CRS).to_crs("EPSG:4326").values
    return g


class FieldStore:
    """Georgia CSB fields, read tile by tile on demand. Same results as the old whole-state table."""

    def __init__(self, tiles_dir: Path | None = None):
        self.dir = Path(tiles_dir or TILES_DIR)
        if not (self.dir / "index.parquet").exists():
            full = C.FIELDS_DIR / "ga_fields.parquet"
            if not full.exists():
                raise FileNotFoundError(f"No field tiles in {self.dir} and no {full}")
            build_tiles(full, self.dir)           # one-time, local installs only
        self.index = pd.read_parquet(self.dir / "index.parquet")
        self._x = np.radians(self.index["lon"].values) * np.cos(np.radians(32.5)) * 6371.0
        self._y = np.radians(self.index["lat"].values) * 6371.0
        self._load = lru_cache(maxsize=16 if C.HOSTED else 48)(self._read_tile)     # tiles kept in memory

    def __len__(self):
        return len(self.index)

    def _read_tile(self, t: str) -> gpd.GeoDataFrame:
        return _add_inner(gpd.read_parquet(self.dir / f"{t}.parquet"))

    def _dist_km(self, lon, lat):
        return np.hypot(self._x - math.radians(lon) * math.cos(math.radians(32.5)) * 6371.0, self._y - math.radians(lat) * 6371.0)

    def _fields(self, rows: pd.DataFrame) -> gpd.GeoDataFrame:
        if rows.empty:
            return gpd.GeoDataFrame({"field_id": []}, geometry=gpd.GeoSeries([], crs="EPSG:4326"), crs="EPSG:4326").assign(geometry_inner=None)
        parts = [self._load(str(t)) for t in rows["tile"].astype(str).unique()]
        g = pd.concat(parts, ignore_index=True)
        g = g[g["field_id"].astype(str).isin(set(rows["field_id"]))]
        return gpd.GeoDataFrame(g, geometry="geometry", crs="EPSG:4326")

    @property
    def years(self) -> list[int]:
        """Years with a crop label in this field store (e.g. 2018…2025); [] for old tiles with the latest label only."""
        return sorted(int(c[5:]) for c in self.index.columns if re.fullmatch(r"crop_\d{4}", c))

    def label_year(self, season_year: int) -> tuple[int | None, str]:
        """Which year's crop labels to use for a season: the same year when we have it, else the newest (or oldest).
        Returns (year, column)."""
        ys = self.years
        if not ys:
            return None, "crop_latest"
        y = min(max(season_year, ys[0]), ys[-1])
        return y, f"crop_{y}"

    def near(self, lon: float, lat: float, radius_km: float, crop: str | None = None, season_year: int | None = None) -> gpd.GeoDataFrame:
        """Fields whose centroid is within radius_km, growing `crop` in `season_year` (labels of that year when available)."""
        d = self._dist_km(lon, lat)
        m = d <= radius_km * 1.02 + 0.2                       # fast pre-filter, exact distance below
        col = self.label_year(season_year)[1] if season_year else "crop_latest"
        if crop:
            m &= (self.index[col] == crop).values
        g = self._fields(self.index[m])
        return fields_in_buffer(g, lon, lat, radius_km, crop, col) if len(g) else g.assign(dist_km=[])

    def at(self, lon: float, lat: float) -> pd.Series:
        """The field containing the point, else the nearest one."""
        d = self._dist_km(lon, lat)
        cand = self.index.iloc[np.argsort(d)[:40]]
        return nearest_field(self._fields(cand), lon, lat)

    def by_id(self, fid: str) -> gpd.GeoDataFrame:
        return self._fields(self.index[self.index["field_id"] == str(fid)])
