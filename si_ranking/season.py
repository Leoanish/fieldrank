"""
Season views built from ONE time-series read.

    season_series(fields, start, end)        every clear Sentinel-2 look of every field (NDVI, GNDVI, NDRE)
    ts_vs_reference(S, ...)                  per date: my value, comparable-field band, virtual reference, SI
    season_composite(S, vi, stat)            per field: season median (or peak) of its per-date field medians
    si_trend(ts)                             direction of my SI over the season

Speed: only my field + similar-planting fields are read; the scene engine screens clouds at 60 m,
reads 4 scenes in parallel and caches every (scene, field) value — shared with the single-date ranking
(kind "vi3"), so a range that overlaps earlier work costs only the new scenes.
The season composite is a *field-level* median: the median of each field's per-date medians over clear
looks. For ranking fields this is practically identical to a pixel-level median composite and ~50× cheaper.
"""
from __future__ import annotations

import datetime as dt

import numpy as np
import pandas as pd

from . import config as C


def season_series(fields, start: dt.date, end: dt.date, progress=None, log=print) -> tuple[pd.DataFrame, int, int]:
    """Long table [field_id, date, NDVI, GNDVI, NDRE, …] — one row per field per clear look."""
    from . import scenes
    df, n_found, n_used = scenes.series(fields, start, end, dict(C.VI_ENGINE), kind=C.VI_KIND, min_pix=C.MIN_PIXELS_10M,
                                        progress=progress, log=log)
    if len(df):
        df["date"] = pd.to_datetime(df["date"]).dt.normalize()
        for c in ["n_pix"] + [c for c in df.columns if c.split("_")[0] in C.VI_ENGINE]:
            df[c] = pd.to_numeric(df[c], errors="coerce")
        # a field seen by two overlapping tiles on the same day → keep the look with more clear pixels
        df = df.sort_values("n_pix", ascending=False).drop_duplicates(["field_id", "date"]).sort_values(["field_id", "date"])
    return df.reset_index(drop=True), n_found, n_used


def reference_by_date(S: pd.DataFrame, comp_ids, ref_ids, vi: str, q: float = C.VIRTUAL_REF_QUANTILE) -> pd.Series:
    """THE reference used everywhere (rank, scout answer, trend): per image date, quantile q of the confident
    comparable fields clear that day (never my own field); if fewer than 3 of those, quantile q of all comparable
    fields clear that day; if fewer than 3 of those either, no reference (NaN) — too few to judge."""
    S = S.assign(**{vi: pd.to_numeric(S[vi], errors="coerce")}).dropna(subset=[vi])
    comp = S[S["field_id"].isin(set(comp_ids))]
    refs = S[S["field_id"].isin(set(ref_ids))]
    q3 = lambda v: np.nanquantile(v, q) if len(v) >= 3 else np.nan   # noqa: E731
    r = refs.groupby("date")[vi].agg(q3)
    c = comp.groupby("date")[vi].agg(q3)
    out = r.reindex(c.index.union(r.index)).fillna(c)
    return out.where(out > 0)                                         # a ratio needs a positive reference


def ts_vs_reference(S: pd.DataFrame, comp_ids, ref_ids, my_id: str, vi: str, q: float = C.VIRTUAL_REF_QUANTILE,
                    min_n: int = 5) -> pd.DataFrame:
    """Per image date: comparable-field quartiles, the shared reference (reference_by_date), my value and
    SI = mine / reference. Dates where fewer than `min_n` comparable fields are clear are dropped."""
    S = S.assign(**{vi: pd.to_numeric(S[vi], errors="coerce")}).dropna(subset=[vi])
    comp = S[S["field_id"].isin(set(comp_ids))]
    mine = S[S["field_id"] == my_id].set_index("date")[vi]
    g = comp.groupby("date")[vi]
    out = pd.DataFrame({"n": g.size(), "p25": g.quantile(0.25), "p50": g.median(), "p75": g.quantile(0.75)})
    out["ref"] = reference_by_date(S, comp_ids, ref_ids, vi, q).reindex(out.index)
    out["me"] = mine.reindex(out.index)
    out = out[out["n"] >= min_n]
    out["SI"] = out["me"] / out["ref"]
    out["rank"] = np.nan
    for d in out.index[out["me"].notna()]:
        v = comp.loc[comp["date"] == d, vi].values
        out.loc[d, "rank"] = 1 + int((v > out.loc[d, "me"]).sum())
    return out.reset_index().rename(columns={"index": "date"})


def rank_on_date(S: pd.DataFrame, comp_ids, ref_ids, my_id: str, vi: str, img_date, as_of, q: float = C.VIRTUAL_REF_QUANTILE,
                 window: int = 8) -> pd.DataFrame:
    """Rank my field against EVERY comparable field around one image date.

    My field uses `img_date`. Every other field uses its clear image nearest to it (img_date - window … as_of). Images of
    different days differ (haze, sun angle, a few days of growth), so each other day is first CALIBRATED to the check
    image with the fields that were clear on both days: k = median(value on img_date / value on that day). Values are put
    on the check image's scale (value × k), then ONE reference (quantile q of the other fields, never mine) and one SI
    are computed for everybody. A day that shares fewer than 3 fields with the check image is calibrated with the ratio
    of the two days' references; a day where neither works is not used. Fields with no usable image get rank NaN."""
    ids = list(dict.fromkeys(list(comp_ids) + [my_id]))
    comp_set, ref_set = set(comp_ids) - {my_id}, set(ref_ids) - {my_id}
    S = S[S["field_id"].isin(ids)]
    S = S.assign(**{vi: pd.to_numeric(S[vi], errors="coerce")}).dropna(subset=[vi])
    S = S[S[vi] > 0]
    d0 = pd.Timestamp(img_date)
    S = S[(S["date"] >= d0 - pd.Timedelta(days=window)) & (S["date"] <= pd.Timestamp(as_of))]
    base = S[S["date"] == d0].set_index("field_id")[vi]
    day_ref = reference_by_date(S, comp_ids, ref_ids, vi, q)
    k = {d0: 1.0}
    for d in sorted(set(S["date"]) - {d0}):
        vd = S[S["date"] == d].set_index("field_id")[vi]
        common = sorted((set(base.index) & set(vd.index)) - {my_id})
        if len(common) >= 3:
            k[d] = float(np.median(base[common].values / vd[common].values))
        elif pd.notna(day_ref.get(d0)) and pd.notna(day_ref.get(d)):
            k[d] = float(day_ref[d0] / day_ref[d])
    S = S[S["date"].isin(list(k))].assign(k=lambda x: x["date"].map(k))
    S = S.assign(adj=S[vi] * S["k"], gap=(S["date"] - d0).dt.days.abs(), later=(S["date"] > d0).astype(int))
    mine = S[(S["field_id"] == my_id) & (S["date"] == d0)]
    others = S[S["field_id"] != my_id].sort_values(["field_id", "gap", "later"]).drop_duplicates("field_id")
    pick = pd.concat([mine, others], ignore_index=True)
    rv = pick.loc[pick["field_id"].isin(ref_set), "adj"]
    cv = pick.loc[pick["field_id"].isin(comp_set), "adj"]
    ref = float(np.nanquantile(rv, q)) if len(rv) >= 3 else (float(np.nanquantile(cv, q)) if len(cv) >= 3 else np.nan)
    out = pd.DataFrame({"field_id": ids}).merge(
        pick[["field_id", "date", vi, "adj", "k", "gap"] + [c for c in ("n_pix", "scene_id") if c in pick]], on="field_id", how="left")
    out = out.rename(columns={vi: "vi_raw", "adj": "vi_median", "date": "obs_date", "gap": "days_from_image", "k": "calibration"})
    out["ref_day"] = ref
    out["SI"] = out["vi_median"] / ref if ref > 0 else np.nan
    ok = out["SI"].notna()
    out["rank"] = pd.array([pd.NA] * len(out), dtype="Int64")
    r_ = out.loc[ok, "SI"].rank(ascending=False, method="first")
    out.loc[ok, "rank"] = r_.astype("Int64")
    out["percentile"] = np.nan
    out.loc[ok, "percentile"] = (1 - (r_ - 1) / max(int(ok.sum()) - 1, 1)) * 100
    out["n_ranked"] = int(ok.sum())
    out["in_ref"] = out["field_id"].isin(ref_set) & ok
    out["reason"] = np.where(ok, "", "no clear image")
    return out


def rank_series(S: pd.DataFrame, comp_ids, ref_ids, my_id: str, vi: str, as_of, q: float = C.VIRTUAL_REF_QUANTILE,
                window: int = 8, last: int = 8, all_fields: bool = False):
    """The rank of my field on each of its recent clear images, every point computed exactly like rank_on_date —
    so the trend and the headline can never disagree."""
    m_ = S[(S["field_id"] == my_id) & (S["date"] <= pd.Timestamp(as_of))]
    dates = sorted(m_.loc[pd.to_numeric(m_[vi], errors="coerce").notna(), "date"].unique())[-last:]
    rows, every = [], []
    for d in dates:
        d = pd.Timestamp(d)
        rt = rank_on_date(S, comp_ids, ref_ids, my_id, vi, d, min(pd.Timestamp(as_of), d + pd.Timedelta(days=window)), q, window)
        every.append(rt.loc[rt["SI"].notna(), ["field_id", "SI"]].assign(date=d))
        m = rt[rt["field_id"] == my_id].iloc[0]
        if pd.notna(m["SI"]):
            others = rt[(rt["field_id"] != my_id) & rt["SI"].notna()]["vi_median"]
            rows.append({"date": d, "SI": float(m["SI"]), "rank": int(m["rank"]), "n": int(m["n_ranked"]) - 1,
                         "me": float(m["vi_median"]), "ref": float(m["ref_day"]),
                         "p25": float(others.quantile(.25)) if len(others) else np.nan, "p50": float(others.median()) if len(others) else np.nan,
                         "p75": float(others.quantile(.75)) if len(others) else np.nan})
    ts = pd.DataFrame(rows, columns=["date", "SI", "rank", "n", "me", "ref", "p25", "p50", "p75"])
    if all_fields:
        return ts, (pd.concat(every, ignore_index=True) if every else pd.DataFrame(columns=["field_id", "SI", "date"]))
    return ts


def season_composite(S: pd.DataFrame, vi: str, stat: str = "median", min_obs: int = 3) -> pd.DataFrame:
    """Per field: season median (stat="median") or season peak (stat="peak" = 90th percentile) of the
    per-date field medians; fields with fewer than `min_obs` clear looks get NaN."""
    S = S.assign(**{vi: pd.to_numeric(S[vi], errors="coerce")}).dropna(subset=[vi])
    g = S.groupby("field_id")
    val = g[vi].median() if stat == "median" else g[vi].quantile(0.9)
    out = pd.DataFrame({"vi_median": val, "n_obs": g.size(), "first_obs": g["date"].min().dt.date, "last_obs": g["date"].max().dt.date})
    out.loc[out["n_obs"] < min_obs, "vi_median"] = np.nan
    return out.reset_index()


def si_trend(ts: pd.DataFrame, last_n: int = 6) -> dict:
    """Mean SI over the season, latest SI, and the slope of SI over the last `last_n` clear dates (per 10 days)."""
    t = ts.dropna(subset=["SI"]).sort_values("date")
    if t.empty:
        return {"n": 0, "mean": np.nan, "latest": np.nan, "slope10": np.nan, "word": "—"}
    tail = t.tail(last_n)
    slope = np.nan
    if len(tail) >= 3:
        x = (tail["date"] - tail["date"].min()).dt.days.values.astype(float)
        if np.ptp(x) > 0:
            slope = float(np.polyfit(x, tail["SI"].values, 1)[0] * 10)
    word = "—" if np.isnan(slope) else ("improving" if slope > 0.01 else "declining" if slope < -0.01 else "steady")
    return {"n": len(t), "mean": float(t["SI"].mean()), "latest": float(t["SI"].iloc[-1]), "slope10": slope, "word": word,
            "share_good": float((t["SI"] >= 0.85).mean())}
