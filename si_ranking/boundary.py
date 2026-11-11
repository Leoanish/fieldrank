"""
Read a farmer's field boundary from almost anything they have.

    read_boundary([(filename, bytes), ...]) -> (GeoDataFrame[name, acres, kind, geometry] in EPSG:4326, notes)

Accepted
  • Shapefile parts uploaded together: .shp + .shx + .dbf + .prj (+ .cpg …) — matched by file stem;
    a missing .shx is rebuilt, a missing .prj is guessed from the coordinates
  • .zip — anything inside (nested folders, several shapefiles, KML/KMZ, GeoJSON, GeoPackage …)
  • .kml / .kmz — GDAL first, built-in XML parser as fallback (folders, multi-geometry, 3-D coordinates, holes)
  • .geojson / .json, .gpkg (all layers), .gml, .fgb, .parquet (GeoParquet), .gpx
  • .csv / .txt with a WKT column, or lat / lon columns (points)
Points and lines are kept with kind="point" / "line" so the app can snap them to the field that contains them.
"""
from __future__ import annotations

import os
import re
import tempfile
import zipfile
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
from shapely.geometry import LineString, MultiPolygon, Point, Polygon
from shapely.ops import polygonize, unary_union

VECTOR_EXT = {".shp", ".kml", ".kmz", ".geojson", ".json", ".gpkg", ".gml", ".fgb", ".parquet", ".gpx", ".csv", ".txt", ".zip"}
SIDECAR_EXT = {".shx", ".dbf", ".prj", ".cpg", ".qix", ".sbn", ".sbx", ".xml", ".qmd", ".fix", ".aih", ".ain", ".atx", ".ixs", ".mxs"}
UPLOAD_TYPES = sorted(e.lstrip(".") for e in VECTOR_EXT | SIDECAR_EXT)
NAME_COLS = ("name", "Name", "NAME", "field", "Field", "FIELD", "field_name", "FieldName", "FIELD_NAME", "fieldname", "label", "Label", "farm", "Farm", "FARM")


# ----------------------------------------------------------------------------- geometry helpers
def _2d(g):
    from shapely import force_2d
    return force_2d(g) if g is not None else None


def _clean(g):
    from shapely import make_valid
    if g is None or g.is_empty:
        return None
    g = make_valid(_2d(g))
    if g.geom_type == "GeometryCollection":
        polys = [p for p in g.geoms if isinstance(p, (Polygon, MultiPolygon))]
        if polys:
            return unary_union(polys)
        lines = [p for p in g.geoms if isinstance(p, LineString)]
        if lines:
            return unary_union(lines)
        pts = [p for p in g.geoms if isinstance(p, Point)]
        return pts[0] if pts else None
    return g


def _kind(g):
    t = g.geom_type
    if "Polygon" in t:
        return "polygon"
    if "Line" in t:
        return "line"
    return "point"


def _close_lines(g):
    """A field drawn as a closed line (common in KML / GPS exports) → polygon."""
    if g is not None and "Line" in g.geom_type:
        polys = list(polygonize(g))
        if polys:
            return unary_union(polys)
    return g


def _guess_crs(gdf: gpd.GeoDataFrame, notes: list) -> gpd.GeoDataFrame:
    if gdf.crs is not None:
        return gdf
    x0, y0, x1, y1 = gdf.total_bounds
    if -180 <= x0 <= 180 and -90 <= y0 <= 90 and -180 <= x1 <= 180 and -90 <= y1 <= 90:
        notes.append("No projection file (.prj) — coordinates look like longitude/latitude, so WGS84 was assumed.")
        return gdf.set_crs("EPSG:4326")
    # projected but unknown: try the usual Georgia suspects and keep the one that lands in Georgia
    for epsg in (32617, 32616, 26917, 26916, 2239, 2240, 5070, 3857):
        try:
            t = gdf.set_crs(epsg).to_crs("EPSG:4326")
            bx = t.total_bounds
            if -86.0 < bx[0] < -80.5 and 30.2 < bx[1] < 35.2:
                notes.append(f"No projection file (.prj) — the coordinates fit Georgia as EPSG:{epsg}, so that was assumed. Please check the outline on the map.")
                return t
        except Exception:
            continue
    raise ValueError("The file has no projection (.prj) and the coordinates don't look like longitude/latitude. Please include the .prj file.")


# ----------------------------------------------------------------------------- KML fallback parser
def _kml_coords(text: str):
    pts = []
    for tok in re.split(r"\s+", text.strip()):
        parts = tok.split(",")
        if len(parts) >= 2:
            try:
                pts.append((float(parts[0]), float(parts[1])))
            except ValueError:
                pass
    return pts


def parse_kml(data: bytes) -> gpd.GeoDataFrame:
    """Minimal, dependency-free KML reader: every Placemark's Polygon / LineString / Point (incl. MultiGeometry)."""
    import xml.etree.ElementTree as ET
    root = ET.fromstring(data)

    def local(tag):
        return tag.rsplit("}", 1)[-1]

    rows = []
    for pm in root.iter():
        if local(pm.tag) != "Placemark":
            continue
        name = next((el.text for el in pm if local(el.tag) == "name" and el.text), None)
        geoms = []
        for el in pm.iter():
            t = local(el.tag)
            if t == "Polygon":
                outer, holes = None, []
                for b in el.iter():
                    bt = local(b.tag)
                    if bt in ("outerBoundaryIs", "innerBoundaryIs"):
                        c = next((x for x in b.iter() if local(x.tag) == "coordinates"), None)
                        ring = _kml_coords(c.text or "") if c is not None else []
                        if len(ring) >= 3:
                            if bt == "outerBoundaryIs":
                                outer = ring
                            else:
                                holes.append(ring)
                if outer:
                    geoms.append(Polygon(outer, holes))
            elif t == "LineString":
                c = next((x for x in el if local(x.tag) == "coordinates"), None)
                pts = _kml_coords(c.text or "") if c is not None else []
                if len(pts) >= 2:
                    geoms.append(LineString(pts))
            elif t == "Point":
                c = next((x for x in el if local(x.tag) == "coordinates"), None)
                pts = _kml_coords(c.text or "") if c is not None else []
                if pts:
                    geoms.append(Point(pts[0]))
        polys = [g for g in geoms if isinstance(g, Polygon)]
        g = unary_union(polys) if polys else (unary_union(geoms) if geoms else None)
        if g is not None:
            rows.append({"name": name, "geometry": g})
    return gpd.GeoDataFrame(rows, geometry="geometry", crs="EPSG:4326") if rows else gpd.GeoDataFrame({"name": []}, geometry=[], crs="EPSG:4326")


# ----------------------------------------------------------------------------- readers
def _read_csv(path: Path) -> gpd.GeoDataFrame:
    df = pd.read_csv(path, sep=None, engine="python")
    low = {c.lower(): c for c in df.columns}
    wkt_col = next((low[c] for c in ("wkt", "geometry", "geom", "the_geom", "shape") if c in low), None)
    if wkt_col:
        from shapely import wkt
        return gpd.GeoDataFrame(df.drop(columns=[wkt_col]), geometry=[wkt.loads(str(v)) for v in df[wkt_col]], crs="EPSG:4326")
    lat = next((low[c] for c in ("lat", "latitude", "y") if c in low), None)
    lon = next((low[c] for c in ("lon", "lng", "long", "longitude", "x") if c in low), None)
    if lat and lon:
        return gpd.GeoDataFrame(df, geometry=gpd.points_from_xy(df[lon], df[lat]), crs="EPSG:4326")
    raise ValueError("CSV needs a WKT column (wkt / geometry) or lat + lon columns.")


def _read_layers(path: Path) -> list[gpd.GeoDataFrame]:
    """All layers of one vector file (GeoPackage / KML can hold several)."""
    try:
        import pyogrio
        layers = [l[0] for l in pyogrio.list_layers(path)]
    except Exception:
        layers = [None]
    out = []
    for lyr in layers or [None]:
        try:
            g = gpd.read_file(path, layer=lyr) if lyr is not None else gpd.read_file(path)
            if len(g):
                g["_layer"] = lyr
                out.append(g)
        except Exception:
            continue
    return out


def _read_one(path: Path, notes: list) -> list[gpd.GeoDataFrame]:
    ext = path.suffix.lower()
    if ext in (".csv", ".txt"):
        return [_read_csv(path)]
    if ext == ".parquet":
        return [gpd.read_parquet(path)]
    if ext == ".kmz":
        with zipfile.ZipFile(path) as z:
            kml_name = next((n for n in z.namelist() if n.lower().endswith(".kml")), None)
            if not kml_name:
                raise ValueError(f"{path.name}: no .kml inside the .kmz")
            kp = path.with_suffix(".kml")
            kp.write_bytes(z.read(kml_name))
        return _read_one(kp, notes)
    if ext == ".shp":
        os.environ["SHAPE_RESTORE_SHX"] = "YES"             # rebuild a missing / broken .shx
        if not path.with_suffix(".dbf").exists() and not path.with_suffix(".DBF").exists():
            notes.append(f"{path.name}: no .dbf — field names/attributes are not available (outline is fine).")
    got = _read_layers(path)
    if ext == ".kml":
        got = [g for g in got if len(g)]
        if not got:                                         # GDAL without KML support, or an odd KML
            k = parse_kml(path.read_bytes())
            got = [k] if len(k) else []
    return got


def _gather(paths: list[Path], notes: list) -> list[gpd.GeoDataFrame]:
    frames = []
    for p in paths:
        ext = p.suffix.lower()
        if ext == ".zip":
            d = Path(tempfile.mkdtemp(prefix="fr_zip_"))
            with zipfile.ZipFile(p) as z:
                for m in z.infolist():
                    if m.is_dir() or "__MACOSX" in m.filename or Path(m.filename).name.startswith("._"):
                        continue
                    target = d / Path(m.filename).name if Path(m.filename).suffix.lower() not in SIDECAR_EXT | {".shp"} else d / m.filename
                    target.parent.mkdir(parents=True, exist_ok=True)
                    target.write_bytes(z.read(m))
            inner = [q for q in d.rglob("*") if q.is_file() and q.suffix.lower() in VECTOR_EXT]
            if not inner:
                raise ValueError(f"{p.name}: no boundary file inside the zip (looked for .shp, .kml, .kmz, .geojson, .gpkg …).")
            frames += _gather(inner, notes)
        elif ext in VECTOR_EXT:
            try:
                for g in _read_one(p, notes):
                    g["_file"] = p.name
                    frames.append(g)
            except Exception as e:
                notes.append(f"Could not read {p.name}: {e}")
    return frames


def read_boundary(files: list[tuple[str, bytes]]) -> tuple[gpd.GeoDataFrame, list[str]]:
    """files: [(filename, bytes)] as uploaded. Returns (features in EPSG:4326, notes for the user)."""
    notes: list[str] = []
    d = Path(tempfile.mkdtemp(prefix="fr_up_"))
    paths = []
    for name, data in files:
        p = d / Path(name).name
        p.write_bytes(data)
        paths.append(p)
    exts = {p.suffix.lower() for p in paths}
    if exts & SIDECAR_EXT and ".shp" not in exts and ".zip" not in exts:
        raise ValueError("Shapefile parts were uploaded without the .shp — please add the .shp file (or upload everything as one .zip).")
    if ".shp" in exts:
        for shp in [p for p in paths if p.suffix.lower() == ".shp"]:
            missing = [e for e in (".shx", ".dbf", ".prj") if not (shp.with_suffix(e).exists() or shp.with_suffix(e.upper()).exists())]
            if missing:
                notes.append(f"{shp.name}: missing {', '.join(missing)} — handled automatically where possible.")
    main = [p for p in paths if p.suffix.lower() in VECTOR_EXT]
    frames = _gather(main, notes)
    if not frames:
        raise ValueError("No field boundary found in the upload. " + (" ".join(notes) if notes else ""))

    feats = []
    for g in frames:
        g = _guess_crs(g, notes) if g.crs is None else g
        g = g.to_crs("EPSG:4326")
        name_col = next((c for c in NAME_COLS if c in g.columns), None)
        for i, (_, r) in enumerate(g.iterrows()):
            geom = _close_lines(_clean(r.geometry))
            if geom is None or geom.is_empty:
                continue
            nm = r.get(name_col) if name_col else None
            nm = str(nm) if nm is not None and str(nm).strip() not in ("", "nan", "None") else f"{r.get('_file', 'feature')} #{i + 1}"
            feats.append({"name": nm, "geometry": geom, "source": r.get("_file", "")})
    if not feats:
        raise ValueError("The file was read, but it contains no usable geometry.")
    out = gpd.GeoDataFrame(feats, geometry="geometry", crs="EPSG:4326")
    from shapely import set_precision
    key = gpd.GeoSeries(set_precision(out.geometry.values, 1e-5), crs=out.crs).normalize().to_wkb()   # ~1 m: same field in two files
    out = out[~key.duplicated()]
    out["kind"] = [_kind(g) for g in out.geometry]
    out["acres"] = np.where(out["kind"] == "polygon", out.to_crs(5070).area / 4046.8564224, np.nan)
    # polygons first, largest first; tiny slivers (< 0.5 ac) dropped when real fields exist
    out["_k"] = out["kind"].map({"polygon": 0, "line": 1, "point": 2})
    out = out.sort_values(["_k", "acres"], ascending=[True, False]).drop(columns="_k").reset_index(drop=True)
    if (out["kind"] == "polygon").any():
        big = out[(out["kind"] == "polygon") & (out["acres"] >= 0.5)]
        out = big if len(big) else out[out["kind"] == "polygon"]
    b = out.total_bounds
    if not (-86.0 < b[0] < -80.5 and 30.2 < b[1] < 35.2 and -86.0 < b[2] < -80.5 and 30.2 < b[3] < 35.2):
        notes.append("The boundary is not inside Georgia — FieldRank currently covers Georgia fields only.")
    return out.reset_index(drop=True), notes
