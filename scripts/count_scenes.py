"""
How many satellite looks does a field get in a month?  Sentinel-2 · Landsat 8/9 · Sentinel-1.

    python scripts/count_scenes.py                       # Dooly County cotton example, August 2026
    python scripts/count_scenes.py 31.45 -83.51 2026-08  # any lat lon and month

Counts acquisition DATES that cover the point (several tiles/frames on one day count once), by satellite,
and for the optical ones how many were mostly clear (scene cloud ≤ 40 % — a field-level check is stricter).
"""
from __future__ import annotations

import datetime as dt
import sys
from collections import Counter, defaultdict

import pystac_client

LAT, LON = (float(sys.argv[1]), float(sys.argv[2])) if len(sys.argv) >= 3 else (32.1688, -83.7074)
MONTH = sys.argv[3] if len(sys.argv) >= 4 else "2026-08"
y, m = map(int, MONTH.split("-"))
start = dt.date(y, m, 1)
end = (dt.date(y + (m == 12), m % 12 + 1, 1) - dt.timedelta(days=1))
when = f"{start}/{end}"
pt = {"type": "Point", "coordinates": [LON, LAT]}

ES = pystac_client.Client.open("https://earth-search.aws.element84.com/v1")
PC = pystac_client.Client.open("https://planetarycomputer.microsoft.com/api/stac/v1")


def by_date(items, plat_key="platform"):
    days = defaultdict(lambda: {"plat": set(), "cloud": []})
    for it in items:
        p = it.properties
        d = p["datetime"][:10]
        days[d]["plat"].add(str(p.get(plat_key, p.get("platform", "?"))).lower())
        if "eo:cloud_cover" in p:
            days[d]["cloud"].append(p["eo:cloud_cover"])
        if "sat:orbit_state" in p:
            days[d]["plat"].add(p["sat:orbit_state"][:4])
    return dict(sorted(days.items()))


def report(name, days, optical=True):
    n = len(days)
    plats = Counter(pl for v in days.values() for pl in v["plat"] if not pl.startswith(("asce", "desc")))
    line = f"{name:<14} {n:>2} dates  ({', '.join(f'{k} {c}' for k, c in sorted(plats.items()))})"
    if optical:
        clear = sum(1 for v in days.values() if v["cloud"] and min(v["cloud"]) <= 40)
        line += f"  · {clear} mostly clear (≤ 40 % cloud)"
    print(line)
    print("               " + "  ".join(f"{d[5:]}{'' if not optical else '·' + str(round(min(v['cloud']))) + '%' if v['cloud'] else ''}" for d, v in days.items()))
    return n


print(f"Point {LAT:.4f}, {LON:.4f} · {start:%B %Y}\n")
s2 = by_date(ES.search(collections=["sentinel-2-l2a"], intersects=pt, datetime=when).items())
n2 = report("Sentinel-2", s2)
ls = by_date(PC.search(collections=["landsat-c2-l2"], intersects=pt, datetime=when,
                       query={"platform": {"in": ["landsat-8", "landsat-9"]}}).items())
nl = report("Landsat 8/9", ls)
s1 = by_date(PC.search(collections=["sentinel-1-rtc"], intersects=pt, datetime=when).items())
if not s1:
    s1 = by_date(PC.search(collections=["sentinel-1-grd"], intersects=pt, datetime=when).items())
n1 = report("Sentinel-1", s1, optical=False)

opt = sorted(set(s2) | set(ls)); allx = sorted(set(s2) | set(ls) | set(s1))
gap = lambda ds: max(((dt.date.fromisoformat(b) - dt.date.fromisoformat(a)).days for a, b in zip(ds, ds[1:])), default=None)
print(f"\nOptical (S2 + Landsat): {len(opt)} dates, longest gap {gap(opt)} days")
print(f"All three:              {len(allx)} dates, longest gap {gap(allx)} days  → about one look every {31 / max(len(allx), 1):.1f} days")
print("Note: Landsat Level-2 and Sentinel-1 RTC can appear 1–3 weeks after acquisition; re-run later for the full month.")
