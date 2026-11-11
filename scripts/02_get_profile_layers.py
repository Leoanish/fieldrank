"""
Download / prepare the static layers used for "compare fields with a similar profile".

    1. LANID-US irrigation (Xie & Lark 2021, Zenodo 10.5281/zenodo.5548555), 30 m, 1997-2017
         lanid2011-2017.zip   (~360 MB)  -> lanid2015/2016/2017
         irrFreqAndChange.zip (~590 MB)  -> irrFreq (years irrigated of 21), maxIrrExt (ever irrigated)
       Only the Georgia window is kept (data/ancillary/lanid/*_GA.tif, a few MB each).
    2. Field profiles (terrain from 3DEP 10 m, soil from gNATSGO + SSURGO, irrigation)
       pre-computed for an area so the app never waits for them. Without this step the app
       computes them on the fly for each new buffer (≈ 10-40 s the first time, then cached).

Usage (from the project folder):
    python scripts/02_get_profile_layers.py lanid                       # download + clip LANID
    python scripts/02_get_profile_layers.py lanid --raw-dir ~/Downloads/lanid --delete-zips
    python scripts/02_get_profile_layers.py profiles --around -83.51 31.45 --km 40
    python scripts/02_get_profile_layers.py profiles --all              # every GA field (long: hours)
    python scripts/02_get_profile_layers.py all --around -83.51 31.45 --km 40
"""
from __future__ import annotations

import argparse
import shutil
import sys
import time
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from si_ranking import config as C  # noqa: E402

ZENODO = "https://zenodo.org/records/5548555/files/{name}?download=1"
ZIPS = {"lanid2011-2017.zip": ["lanid2015", "lanid2016", "lanid2017"],
        "irrFreqAndChange.zip": ["irrfreq", "maxirrext"]}
GA_BBOX_LL = (-85.65, 30.30, -80.70, 35.05)
RAW = Path.home() / "FieldRank_downloads" / "lanid"   # big downloads stay OUTSIDE the OneDrive project folder
OUT = C.ANC_DIR / "lanid"


def _download(name: str) -> Path:
    import requests
    RAW.mkdir(parents=True, exist_ok=True)
    dst = RAW / name
    part = dst.with_suffix(dst.suffix + ".part")
    if dst.exists() and zipfile.is_zipfile(dst):
        print(f"✓ {name} already downloaded ({dst.stat().st_size / 1e6:.0f} MB)")
        return dst
    have = part.stat().st_size if part.exists() else 0
    headers = {"User-Agent": "Mozilla/5.0 FieldRank", **({"Range": f"bytes={have}-"} if have else {})}
    url = ZENODO.format(name=name)
    print(f"↓ {name} from Zenodo" + (f" (resuming at {have / 1e6:.0f} MB)" if have else ""))
    with requests.get(url, headers=headers, stream=True, timeout=120) as r:
        if r.status_code == 416:            # already complete
            part.rename(dst); return dst
        r.raise_for_status()
        total = int(r.headers.get("Content-Length", 0)) + have
        mode = "ab" if (have and r.status_code == 206) else "wb"
        got = have if mode == "ab" else 0
        t0 = time.time()
        with open(part, mode) as f:
            for chunk in r.iter_content(chunk_size=1 << 20):
                f.write(chunk); got += len(chunk)
                if total:
                    rate = (got - have) / max(time.time() - t0, 1e-3) / 1e6
                    print(f"\r  {got / 1e6:7.0f} / {total / 1e6:.0f} MB  {rate:5.1f} MB/s", end="", flush=True)
    print()
    part.rename(dst)
    if not zipfile.is_zipfile(dst):
        raise RuntimeError(f"{dst} is not a zip — the download failed (check the URL in a browser: {url})")
    return dst


def _extract_wanted(zpath: Path, wanted: list[str]) -> dict[str, Path]:
    """Extract members whose file name starts with a wanted key (case-insensitive), descending into nested zips."""
    found = {}
    tmp = RAW / "_x"
    tmp.mkdir(exist_ok=True)
    with zipfile.ZipFile(zpath) as z:
        for info in z.infolist():
            base = Path(info.filename).name
            low = base.lower()
            if low.endswith(".zip"):
                inner = tmp / base
                with z.open(info) as s, open(inner, "wb") as d:
                    shutil.copyfileobj(s, d)
                found.update(_extract_wanted(inner, [w for w in wanted if w not in found]))
                continue
            if not low.endswith((".tif", ".tiff")):
                continue
            key = next((w for w in wanted if low.startswith(w) and low[len(w):len(w) + 1] in ("", ".", "_")), None)
            if key and key not in found:
                dst = tmp / base
                if not dst.exists():
                    print(f"  extracting {info.filename} ({info.file_size / 1e6:.0f} MB)")
                    with z.open(info) as s, open(dst, "wb") as d:
                        shutil.copyfileobj(s, d)
                found[key] = dst
    return found


def _clip_ga(src_path: Path, key: str) -> Path:
    import rasterio
    from rasterio.warp import transform_bounds
    from rasterio.windows import from_bounds
    OUT.mkdir(parents=True, exist_ok=True)
    names = {"lanid2015": "lanid2015", "lanid2016": "lanid2016", "lanid2017": "lanid2017", "irrfreq": "irrFreq", "maxirrext": "maxIrrExt"}
    dst = OUT / f"{names[key]}_GA.tif"
    with rasterio.open(src_path) as src:
        b = transform_bounds("EPSG:4326", src.crs, *GA_BBOX_LL, densify_pts=21)
        win = from_bounds(*b, transform=src.transform).round_offsets().round_lengths()
        arr = src.read(1, window=win, boundless=True, fill_value=src.nodata if src.nodata is not None else 0)
        prof = src.profile.copy()
        prof.update(height=arr.shape[0], width=arr.shape[1], transform=src.window_transform(win),
                    compress="deflate", tiled=True, blockxsize=512, blockysize=512, driver="GTiff", BIGTIFF="IF_SAFER")
        with rasterio.open(dst, "w", **prof) as d:
            d.write(arr, 1)
    print(f"✓ {dst.name}  {arr.shape[1]}×{arr.shape[0]} px  {dst.stat().st_size / 1e6:.1f} MB")
    return dst


def get_lanid(keep_raw: bool = False, delete_zips: bool = False):
    done = {p.stem.lower().replace("_ga", "") for p in OUT.glob("*_GA.tif")} if OUT.exists() else set()
    for zname, keys in ZIPS.items():
        need = [k for k in keys if k not in done]
        if not need:
            print(f"✓ {', '.join(keys)} already clipped"); continue
        z = _download(zname)
        got = _extract_wanted(z, need)
        missing = [k for k in need if k not in got]
        if missing:
            print(f"! could not find {missing} inside {zname}; members are listed below:")
            with zipfile.ZipFile(z) as zz:
                for n in zz.namelist()[:60]:
                    print("   ", n)
        for k, p in got.items():
            _clip_ga(p, k)
        clipped = {q.stem.lower().replace("_ga", "") for q in OUT.glob("*_GA.tif")}
        if delete_zips and all(k in clipped for k in keys):
            z.unlink(missing_ok=True); print(f"  deleted {z.name}")
    if not keep_raw:
        shutil.rmtree(RAW / "_x", ignore_errors=True)   # big CONUS tifs; the zips stay so this is re-runnable
    print(f"LANID ready in {OUT}  (raw downloads: {RAW})")


def get_profiles(around=None, km=40.0, all_fields=False):
    """Profiles (elevation, soil, irrigation) for CSB fields, cached in data/ancillary/field_profile.parquet.
    --all walks Georgia tile by tile; re-running skips everything already cached, so it can be stopped any time."""
    from si_ranking.fields import FieldStore
    from si_ranking.profile import field_profiles, PROFILE_PATH
    S = FieldStore()
    if all_fields:
        groups = [(t, None) for t in sorted(S.index["tile"].astype(str).unique())]
    else:
        lon, lat = around or (-83.51, 31.45)
        groups = [("around", S.near(lon, lat, km))]
    t0 = time.time(); n_done = 0
    total = len(S.index) if all_fields else len(groups[0][1])
    print(f"Computing profiles for {total:,} fields (corn / cotton / peanut / soybean) — cached fields are skipped…")
    for k, (name, sel) in enumerate(groups):
        sel = S._load(name) if sel is None else sel
        sel = sel[sel["crop_latest"].isin(C.CROPS)]
        if sel.empty:
            continue
        pr = field_profiles(sel, log=lambda m: None, progress=lambda f, m: print(f"\r  {name}: {m}            ", end="", flush=True))
        n_done += len(sel)
        el = pr["elev_median"].notna().mean() if len(pr) else 0; so = pr["sand"].notna().mean() if len(pr) else 0
        print(f"\r  [{k + 1}/{len(groups)}] {name}: {len(sel):,} fields · terrain {el:.0%} · soil {so:.0%} · "
              f"{n_done:,}/{total:,} done · {(time.time() - t0) / 60:.1f} min            ")
    print(f"✓ profiles cached in {PROFILE_PATH}")


def main():
    global RAW
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("what", choices=["lanid", "soil", "profiles", "all"])
    ap.add_argument("--around", nargs=2, type=float, metavar=("LON", "LAT"))
    ap.add_argument("--km", type=float, default=40.0)
    ap.add_argument("--all", action="store_true", help="profiles for every GA field")
    ap.add_argument("--keep-raw", action="store_true", help="keep the unzipped CONUS rasters")
    ap.add_argument("--raw-dir", type=Path, default=None,
                    help=f"where the ~950 MB LANID zips are downloaded (default {RAW}, outside the project folder)")
    ap.add_argument("--delete-zips", action="store_true", help="delete each zip once its Georgia layers are clipped")
    a = ap.parse_args()
    if a.raw_dir:
        RAW = a.raw_dir.expanduser().resolve()
    if a.what in ("lanid", "all"):
        print(f"Raw downloads → {RAW}\nGeorgia layers → {OUT}")
        get_lanid(a.keep_raw, a.delete_zips)
    if a.what in ("soil", "all"):
        from si_ranking.profile import build_gnatsgo_table
        t0 = time.time()
        tbl = build_gnatsgo_table(print)
        print(f"✓ soil table ready: {len(tbl):,} map units · {tbl['sand'].notna().mean():.0%} with texture · {time.time() - t0:.0f} s")
    if a.what in ("profiles", "all"):
        get_profiles(tuple(a.around) if a.around else None, a.km, a.all)


if __name__ == "__main__":
    main()
