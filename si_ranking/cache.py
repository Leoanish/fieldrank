"""
Disk cache housekeeping for the imagery results (data/timeseries/).

What is cached
  scenes/<engine>_<kind>/<scene>.parquet  one small table per satellite scene: every field's value from that scene
  stac/                                   catalogue searches per month (which scenes exist)
  greenup_*.parquet                       green-up result per search area

Every file is small (tens of KB) and is only ever added to, so testing many fields grows the cache slowly.
`prune()` keeps it under a size budget by deleting the least recently used files first, and removes
cache folders from older engine versions straight away.
"""
from __future__ import annotations

import os
import shutil
import time
from pathlib import Path

from . import config as C

ROOT = C.TS_DIR
_LAST = {"t": 0.0}


def _files():
    if not ROOT.exists():
        return []
    out = []
    for p in ROOT.rglob("*"):
        if p.is_file() and p.suffix in (".parquet", ".json", ".pkl", ".tmp"):
            try:
                st = p.stat()
                out.append((p, st.st_size, max(st.st_atime, st.st_mtime)))
            except OSError:
                pass
    return out


def usage() -> dict:
    """{'mb': total MB, 'files': n, 'scenes': n scene files}."""
    fs = _files()
    return {"mb": sum(s for _, s, _ in fs) / 2 ** 20, "files": len(fs),
            "scenes": sum(1 for p, _, _ in fs if "scenes" in p.parts)}


def _obsolete_dirs():
    from .scenes import ENGINE_VERSION
    live = {f"{ENGINE_VERSION}_{k}" for k in ("ndvi_evi2", C.VI_KIND, "l89vi2h")}
    d = ROOT / "scenes"
    return [p for p in d.iterdir() if p.is_dir() and p.name not in live] if d.exists() else []


def prune(max_mb: float | None = None, min_interval_s: float = 300) -> float:
    """Trim the cache to `max_mb` (oldest first). Cheap to call often: runs at most every `min_interval_s`.
    Returns MB freed."""
    now = time.time()
    if now - _LAST["t"] < min_interval_s:
        return 0.0
    _LAST["t"] = now
    max_mb = C.CACHE_MAX_MB if max_mb is None else max_mb
    freed = 0
    for d in _obsolete_dirs():
        freed += sum(f.stat().st_size for f in d.rglob("*") if f.is_file())
        shutil.rmtree(d, ignore_errors=True)
    fs = sorted(_files(), key=lambda x: x[2])
    total = sum(s for _, s, _ in fs)
    budget = max_mb * 2 ** 20
    for p, s, _ in fs:
        if total <= budget * 0.85:            # trim to 85 % so it does not run on every call
            break
        try:
            os.remove(p); total -= s; freed += s
        except OSError:
            pass
    return freed / 2 ** 20


def clear() -> float:
    """Delete every cached imagery result. Returns MB freed."""
    mb = usage()["mb"]
    for name in ("scenes", "stac"):
        shutil.rmtree(ROOT / name, ignore_errors=True)
    for p in ROOT.glob("greenup_*.parquet"):
        try:
            p.unlink()
        except OSError:
            pass
    try:
        from . import scenes
        scenes._SEARCH_MEM.clear()
    except Exception:
        pass
    return mb
