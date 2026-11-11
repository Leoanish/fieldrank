"""
Crop stage awareness + the "right now" read of my field.

    stage_at(crop, dap)                   which growth stage a field is in, days after planting
    add_derived(df)                       WDRVI / GCVI / CIre from the cached NDVI / GNDVI / NDRE (no new download)
    now_status(S, ...)                    newest clear look vs comparable fields, change since the look before,
                                          stage, index for that stage, plain-language verdict
    window_composite(S, planting, ...)    each field's value inside the crop's yield-critical window, counted
                                          from that field's own estimated planting date

Why the derived indices are exact and free: the engine stores per-field medians (and p10 / p90) of
normalized differences x = (a - b) / (a + b). GCVI = a/b - 1 = 2x / (1 - x), CIre the same on the red-edge
pair, and WDRVI = (α·a/b - 1) / (α·a/b + 1) are strictly increasing functions of x, and quantiles commute with
increasing functions — so the median GCVI of a field is exactly f(median GNDVI). Rankings on one date are
therefore identical to the base index; what changes is the SI scale: the derived indices do not flatten out
once the canopy closes, so the gap between a good and a very good field stays visible.

Stage calendars are typical Georgia timings in days after planting — approximate, used to pick the index and
the wording, never to filter fields.
"""
from __future__ import annotations

import datetime as dt
from dataclasses import dataclass

import numpy as np
import pandas as pd

from . import config as C

WDRVI_ALPHA = 0.2
DERIVED = {  # name: (base index, kind)
    "WDRVI": ("NDVI", "wdrvi"),
    "GCVI": ("GNDVI", "ci"),
    "CIre": ("NDRE", "ci"),
}
ALL_VI = list(C.VI_ENGINE) + list(DERIVED)
VI_HELP = {
    "NDVI": "overall greenness · best early, saturates once the canopy closes",
    "GNDVI": "chlorophyll-sensitive greenness",
    "NDRE": "red-edge · chlorophyll / nitrogen in closed canopies",
    "NDMI": "canopy water (NIR vs short-wave infrared) · drops with drought stress",
    "EVI2": "enhanced vegetation index · less saturation and soil influence than NDVI",
    "WDRVI": "wide-dynamic-range NDVI (≈ simple ratio, scaled 0–1) · keeps separating dense canopies",
    "GCVI": "green chlorophyll index · the index behind the Stanford SCYM corn & soybean yield maps",
    "CIre": "red-edge chlorophyll index (705 nm) · closed-canopy chlorophyll, less saturation than NDRE",
}


def _tf(x, kind):
    x = np.clip(np.asarray(x, dtype="float64"), -0.999, 0.999)
    if kind == "ci":
        return 2 * x / (1 - x)
    r = (1 + x) / (1 - x)
    w = (WDRVI_ALPHA * r - 1) / (WDRVI_ALPHA * r + 1)
    w0 = (WDRVI_ALPHA - 1) / (WDRVI_ALPHA + 1)            # bare soil (NIR = Red) → 0, dense canopy → 1
    return (w - w0) / (1 - w0)                            # scaled WDRVI: positive, so "% of the best" is meaningful


def add_derived(df: pd.DataFrame | None) -> pd.DataFrame | None:
    """Add WDRVI, GCVI, CIre (+ _p10 / _p90) next to their base index columns."""
    if df is None or df.empty:
        return df
    df = df.copy()
    for name, (base, kind) in DERIVED.items():
        for suf in ("", "_p10", "_p90"):
            if base + suf in df:
                df[name + suf] = _tf(pd.to_numeric(df[base + suf], errors="coerce"), kind)
    return df


# ----------------------------------------------------------------------------- stage calendars
@dataclass(frozen=True)
class Stage:
    key: str          # early | vegetative | critical | late | post
    label: str
    start: int        # days after planting (inclusive)
    end: int          # exclusive
    vi: str           # index used for "right now" in this stage
    note: str         # how to read SI in this stage
    check: str        # what is worth looking at when the field is behind


STAGES: dict[str, list[Stage]] = {
    "corn": [
        Stage("early", "Emergence & early leaves (VE–V6)", 0, 30, "NDVI",
              "Differences now are mostly stand and emergence.", "stand count, skips, crusting, early weed pressure"),
        Stage("vegetative", "Rapid growth (V7–VT)", 30, 58, "WDRVI",
              "Canopy is closing — fields separate on growth rate.", "nitrogen side-dress, weeds, water"),
        Stage("critical", "Tassel → kernel set (VT–R2)", 58, 88, "CIre",
              "Yield-critical: kernel number is set now — gaps here matter most for yield.", "water stress at silking, nitrogen, disease"),
        Stage("late", "Grain fill → maturity (R3–R6)", 88, 125, "GCVI",
              "Greenness falls as the crop dries down — a later hybrid or later planting looks greener without yielding more.",
              "early leaf loss from disease or drought"),
        Stage("post", "After the season", 125, 10_000, "NDVI",
              "Season is over — the yield-window result is the final read.", ""),
    ],
    "soybean": [
        Stage("early", "Emergence & early growth (VE–V3)", 0, 28, "NDVI",
              "Differences now are mostly stand and emergence.", "stand, replant gaps, weeds"),
        Stage("vegetative", "Vegetative → first flowers (V4–R1)", 28, 45, "WDRVI",
              "Canopy is filling the rows.", "weeds, water, herbicide injury"),
        Stage("critical", "Flowering → seed fill (R2–R5)", 45, 95, "EVI2",
              "Yield-critical: pods and seed number are set now — gaps here matter most for yield.", "water stress, insects, disease"),
        Stage("late", "Seed fill → maturity (R6–R8)", 95, 130, "NDVI",
              "Leaves yellow and drop as the crop matures — earlier varieties go first.", "early defoliation from disease"),
        Stage("post", "After the season", 130, 10_000, "NDVI",
              "Season is over — the yield-window result is the final read.", ""),
    ],
    "cotton": [
        Stage("early", "Emergence → squaring", 0, 35, "NDVI",
              "Differences now are mostly stand and early vigour.", "stand, thrips, seedling disease"),
        Stage("vegetative", "Squaring → first bloom", 35, 60, "WDRVI",
              "Canopy is closing — fields separate on growth rate.", "water, nitrogen, plant-growth-regulator timing"),
        Stage("critical", "Flowering & boll fill", 60, 120, "CIre",
              "Yield-critical: bolls are set and filled now (best signal ~75–120 days) — gaps here matter most for yield.",
              "water stress during bloom, nitrogen, stink bugs"),
        Stage("late", "Boll opening → defoliation", 120, 160, "NDVI",
              "Greenness drops with opening bolls and defoliant — not a problem by itself.", ""),
        Stage("post", "After the season", 160, 10_000, "NDVI",
              "Season is over — the yield-window result is the final read.", ""),
    ],
    "peanut": [
        Stage("early", "Emergence & early growth", 0, 30, "NDVI",
              "Differences now are mostly stand and emergence.", "stand, thrips, TSWV"),
        Stage("vegetative", "Flowering & pegging", 30, 60, "WDRVI",
              "Canopy is lapping the rows.", "water, weeds, calcium for pegging"),
        Stage("critical", "Pod set & pod fill", 60, 105, "WDRVI",
              "Yield-critical: pods set and fill now — healthy green leaf area is the best yield signal before digging.",
              "water stress, leaf spot, white mold"),
        Stage("late", "Maturity & digging", 105, 150, "NDVI",
              "A field that is greener now may simply be less mature — check pod maturity before reading low SI as a problem; "
              "sharp drops can also be leaf-spot defoliation.", "leaf-spot defoliation, pod maturity (hull scrape)"),
        Stage("post", "After the season", 150, 10_000, "NDVI",
              "Season is over — the yield-window result is the final read.", ""),
    ],
}


def stages(crop: str) -> list[Stage]:
    return STAGES.get(crop, STAGES["cotton"])


def stage_at(crop: str, dap: int | float | None) -> Stage | None:
    if dap is None or pd.isna(dap):
        return None
    if dap < 0:
        return Stage("pre", "Not planted yet", -10_000, 0, "NDVI", "Nothing to compare before planting.", "")
    for s in stages(crop):
        if s.start <= dap < s.end:
            return s
    return stages(crop)[-1]


def critical(crop: str) -> Stage:
    return next(s for s in stages(crop) if s.key == "critical")


# ----------------------------------------------------------------------------- yield window composite
def window_composite(S: pd.DataFrame, planting: dict, crop: str, vi: str, stat: str = "mean", min_obs: int = 2,
                     as_of: dt.date | None = None) -> pd.DataFrame:
    """Per field: mean (or peak = 90th pct) of its clear looks inside the crop's yield-critical window, the
    window counted from that field's own estimated planting date. Fields with < min_obs looks get NaN."""
    cw = critical(crop)
    S = S.assign(**{vi: pd.to_numeric(S[vi], errors="coerce")}).dropna(subset=[vi])
    if as_of is not None:
        S = S[S["date"] <= pd.Timestamp(as_of)]
    p = pd.to_datetime(S["field_id"].map(planting))
    dap = (S["date"] - p).dt.days
    W = S[(dap >= cw.start) & (dap < cw.end)]
    g = W.groupby("field_id")[vi]
    val = g.mean() if stat == "mean" else g.quantile(0.9)
    out = pd.DataFrame({"vi_median": val, "n_obs": g.size()})
    if len(W):
        out["first_obs"] = W.groupby("field_id")["date"].min().dt.date
        out["last_obs"] = W.groupby("field_id")["date"].max().dt.date
    out.loc[out["n_obs"] < min_obs, "vi_median"] = np.nan
    return out.reset_index()


# ----------------------------------------------------------------------------- right now
SI_WORDS = [(0.95, "On par with the best comparable fields", "g"),
            (0.85, "Slightly behind the best comparable fields", "g"),
            (0.70, "Clearly behind the best comparable fields — worth scouting", "a"),
            (-1.0, "Well behind the best comparable fields — scout soon", "r")]


def verdict(si: float) -> tuple[str, str]:
    for thr, w, tone in SI_WORDS:
        if si >= thr:
            return w, tone
    return SI_WORDS[-1][1], SI_WORDS[-1][2]


def pick_image_date(S: pd.DataFrame, comp_ids, my_id: str, as_of: dt.date, lookback: int = 30, min_n: int = 5):
    """The image the whole check uses: the newest date on or before `as_of` (within `lookback` days) where my field
    is clear together with at least `min_n` comparable fields. If a slightly older image (≤ 10 days earlier) has
    twice as many comparable fields clear, that one is used — a fairer, fuller comparison.
    Returns (date or None, n comparable fields clear on it)."""
    if S is None or S.empty:
        return None, 0
    S = S[(S["date"] <= pd.Timestamp(as_of)) & (S["date"] >= pd.Timestamp(as_of - dt.timedelta(days=lookback)))]
    v = pd.to_numeric(S["NDVI"], errors="coerce")
    S = S[v.notna()]
    mine = set(S.loc[S["field_id"] == my_id, "date"])
    if not mine:
        return None, 0
    n = S[S["field_id"].isin(set(comp_ids)) & S["date"].isin(mine)].groupby("date").size().reindex(sorted(mine)).fillna(0).astype(int)
    ok = n[n >= min_n] if (n >= min_n).any() else n[n == n.max()]
    d = ok.index.max()
    near = n[(n.index >= d - pd.Timedelta(days=10)) & (n.index <= d)]
    if len(near) and near.max() >= 2 * max(int(n[d]), 1):
        d = near.idxmax()
    return d.date(), int(n[d])


def now_status(S: pd.DataFrame, comp_ids, ref_ids, my_id: str, crop: str, my_planting: dt.date, as_of: dt.date,
               q: float = C.VIRTUAL_REF_QUANTILE, min_n: int = 5, fresh_days: int = 10, vi: str | None = None,
               planting: dict | None = None, at_date: dt.date | None = None, ts_override: pd.DataFrame | None = None) -> dict:
    """Newest same-day comparison of my field with the comparable fields, on or before `as_of`.
    vi=None → the index recommended for my stage on that date."""
    from .season import ts_vs_reference
    out = {"ok": False, "as_of": as_of, "stage_today": stage_at(crop, (as_of - my_planting).days)}
    if S is None or S.empty:
        out["why"] = "no clear images in the last weeks"
        return out
    S = add_derived(S)
    S = S[S["date"] <= pd.Timestamp(as_of)]
    mine = S[(S["field_id"] == my_id)].dropna(subset=["NDVI"])
    if mine.empty:
        out["why"] = "no clear look of my field in the last weeks"
        return out
    # stage → index (decided on my newest clear look, so it matches the image being compared)
    d_last = mine["date"].max()
    st_ = stage_at(crop, (d_last.date() - my_planting).days)
    vi = vi or st_.vi
    ts = ts_override if ts_override is not None else ts_vs_reference(S, comp_ids, ref_ids, my_id, vi, q, min_n=min_n)
    ts = ts.dropna(subset=["SI"]).sort_values("date")
    if at_date is not None:                                   # the check date's image (same one the ranking uses)
        ts = ts[ts["date"] <= pd.Timestamp(at_date)]
    if ts.empty:
        out.update(why=f"no date where my field and at least {min_n} comparable fields were clear together", stage=st_, vi=vi)
        return out
    L = ts.iloc[-1]
    prev = ts[ts["date"] <= L["date"] - pd.Timedelta(days=5)]
    P = prev.iloc[-1] if len(prev) and (L["date"] - prev.iloc[-1]["date"]).days <= 24 else None
    st_ = stage_at(crop, (L["date"].date() - my_planting).days)
    word, tone = verdict(float(L["SI"]))
    dSI = float(L["SI"] - P["SI"]) if P is not None else np.nan
    trend = ("—" if np.isnan(dSI) else "closing the gap" if dSI > 0.03 else "falling behind" if dSI < -0.03 else "holding steady")
    if not np.isnan(dSI) and L["SI"] >= 0.95 and dSI > 0.03:
        trend = "pulling ahead"
    my_ndvi = float(mine.set_index("date")["NDVI"].get(L["date"], np.nan))
    # every comparable field on that same image (for the "where you stand" strip) + within-field spread (NDVI p90 - p10)
    day = S[(S["date"] == L["date"]) & S["field_id"].isin(set(comp_ids) | {my_id})].copy()
    day["value"] = pd.to_numeric(day[vi], errors="coerce")
    day["pct"] = day["value"] / float(L["ref"]) * 100
    day["is_me"] = day["field_id"].eq(my_id)
    spread_me = spread_comp = np.nan
    if {"NDVI_p10", "NDVI_p90"} <= set(day.columns):
        sp = pd.to_numeric(day["NDVI_p90"], errors="coerce") - pd.to_numeric(day["NDVI_p10"], errors="coerce")
        spread_me = float(sp[day["is_me"]].iloc[0]) if day["is_me"].any() else np.nan
        spread_comp = float(np.nanmedian(sp[~day["is_me"]])) if (~day["is_me"]).any() else np.nan
    last3 = ts.tail(3)
    n_behind = int((last3["SI"] < 0.85).sum())
    harvested = st_.key in ("late", "post") and my_ndvi < 0.30
    age = (as_of - L["date"].date()).days
    out.update(ok=True, stage=st_, vi=vi, date=L["date"].date(), age=age, fresh=age <= fresh_days,
               SI=float(L["SI"]), rank=int(L["rank"]), n=int(L["n"]) + 1, me=float(L["me"]), ref=float(L["ref"]),
               prev_date=None if P is None else P["date"].date(), prev_SI=None if P is None else float(P["SI"]),
               prev_rank=None if P is None else int(P["rank"]), prev_n=None if P is None else int(P["n"]) + 1,
               dSI=dSI, trend=trend, word=word, tone=tone, harvested=harvested, my_ndvi=my_ndvi, dap=(L["date"].date() - my_planting).days,
               ts=ts[["date", "SI", "rank", "n", "me", "ref", "p25", "p50", "p75"]].tail(10).reset_index(drop=True),
               day=day[["field_id", "value", "pct", "is_me"]].dropna(subset=["value"]).reset_index(drop=True),
               spread_me=spread_me, spread_comp=spread_comp, n_behind=n_behind, n_last=len(last3),
               n_dates=len(ts))
    # yield window so far (mean of my clear looks inside the critical window, vs comparable fields' own windows)
    if planting:
        cw = critical(crop)
        wc = window_composite(S, planting, crop, cw.vi, "mean", min_obs=1, as_of=as_of)
        wc = wc[wc["field_id"].isin(set(comp_ids) | {my_id})].dropna(subset=["vi_median"])
        refv = wc[wc["field_id"].isin(set(ref_ids))]["vi_median"]
        refv = refv if len(refv) >= 3 else wc[wc["field_id"] != my_id]["vi_median"]
        m = wc[wc["field_id"] == my_id]
        if len(m) and len(refv) >= 3:
            r = float(np.nanquantile(refv, q)); v = float(m["vi_median"].iloc[0])
            others = wc[wc["field_id"] != my_id]["vi_median"].values
            out["window"] = {"SI": v / r, "rank": 1 + int((others > v).sum()), "n": len(others) + 1,
                             "looks": int(m["n_obs"].iloc[0]), "vi": cw.vi, "label": cw.label,
                             "start": my_planting + dt.timedelta(days=cw.start), "end": my_planting + dt.timedelta(days=cw.end - 1)}
    return out


def fetch_ranges(crop: str, my_planting: dt.date, as_of: dt.date, recent_days: int = 42, pad: int = 12) -> list[tuple[dt.date, dt.date]]:
    """Date ranges to read for the 'right now' view: the last `recent_days`, plus the yield-critical window
    (padded for planting differences) when it has started and is not already inside the recent range."""
    rec = (as_of - dt.timedelta(days=recent_days), as_of)
    cw = critical(crop)
    w0, w1 = my_planting + dt.timedelta(days=cw.start - pad), min(as_of, my_planting + dt.timedelta(days=cw.end + pad))
    if w0 >= as_of or w1 <= w0:
        return [rec]
    if w0 >= rec[0]:
        return [rec]
    if w1 >= rec[0]:
        return [(w0, as_of)]
    return [(w0, w1), rec]


# ----------------------------------------------------------------------------- plain-language advice
def advise(ns: dict, today_stage: Stage | None = None) -> dict:
    """Turn the numbers into one decision + reasons a farmer can act on.

    level: scout | watch | ok | info      headline / reason: one line each
    why: short bullet strings             look_for: what to check in the field
    conf: high | medium | low (+ conf_why)

    Rules (deliberately simple, all visible in "How we decide"):
      scout  — under 70% of the best fields, or under 85% AND (behind on 2+ of the last 3 images or slipping
               by 5+ points), or under 80% during the yield-critical stage
      watch  — under 85% on this image only, or under 95% and slipping by 5+ points, or under 95% and in the
               bottom quarter of 8+ comparable fields
      ok     — otherwise
      info   — no usable image, before planting, late season / harvested, or season over
    """
    if not ns.get("ok"):
        st_ = today_stage
        if st_ is not None and st_.key == "pre":
            return dict(level="info", icon="🌱", headline="Too early to compare", reason="The crop is not planted yet on this date.", why=[], look_for="", conf=None)
        return dict(level="info", icon="☁️", headline="Waiting for a clear image",
                    reason=f"No usable comparison yet — {ns.get('why', 'clouds')}. Press Update after the next clear day.", why=[], look_for="", conf=None)
    st_ = ns["stage"]; si = ns["SI"]; pct = round(si * 100); d = ns["dSI"]
    slipping = not np.isnan(d) and d <= -0.05
    gaining = not np.isnan(d) and d >= 0.05
    persistent = ns["n_behind"] >= 2
    crit = st_.key == "critical"
    n_comp = ns["n"] - 1
    better_than = round((n_comp - (ns["rank"] - 1)) / max(n_comp, 1) * 100) if n_comp else 0

    # ---- confidence
    cw = []
    conf = "high"
    win = ns.get("rank_window")
    near = f" (clear within ±{win} days)" if win else " on the same image"
    if n_comp < 5:
        conf = "low"; cw.append(f"only {n_comp} comparable fields to compare with{near}")
    elif n_comp < 10:
        conf = "medium"; cw.append(f"{n_comp} comparable fields{near}")
    else:
        cw.append(f"{n_comp} comparable fields{near}")
    if ns.get("n_noimg"):
        cw.append(f"{ns['n_noimg']} more had no clear image")
    if ns["age"] > 10:
        conf = "low" if conf != "high" else "medium"; cw.append(f"the image is {ns['age']} days old")
    else:
        cw.append(f"image {ns['age']} day{'s' if ns['age'] != 1 else ''} old")
    if ns["n_dates"] < 2:
        conf = "medium" if conf == "high" else conf; cw.append("only one clear date so far")

    # ---- reasons (always the same order: standing, change, consistency, stage, pattern)
    why = [f"<b>{pct}%</b> as green as the best comparable fields — {ns['rank']}{_ord(ns['rank'])} of {ns['n']}"
           + ((f", greener than {better_than}% of them" if better_than > 0 else ", the least green of them") if n_comp else "")]
    if ns["prev_date"] is not None:
        a, b = round(ns["prev_SI"] * 100), pct
        word = "slipping" if slipping else "catching up" if gaining else "about the same"
        why.append(f"Since {ns['prev_date']:%b %d}: {a}% → {b}% · <b>{word}</b>")
    if ns["n_last"] >= 2:
        why.append(f"Behind (under 85%) on <b>{ns['n_behind']} of the last {ns['n_last']}</b> clear images"
                   + (" — a steady pattern, not a one-off image" if persistent else " — could still be a one-off" if ns["n_behind"] == 1 else ""))
    why.append(f"<b>{st_.label}</b> — {st_.note}")
    patchy = None
    if not np.isnan(ns.get("spread_me", np.nan)) and not np.isnan(ns.get("spread_comp", np.nan)) and si < 0.95:
        patchy = ns["spread_me"] > max(1.5 * ns["spread_comp"], 0.12)
        why.append("The field is <b>patchy</b> — some parts are much weaker than others; start with the weakest spots" if patchy else
                   "The field is <b>evenly</b> behind — more often water, nutrients, variety or planting date than a local problem")

    # ---- late season / harvested → information, not a scouting call
    if ns["harvested"]:
        return dict(level="info", icon="🚜", headline="Looks harvested or defoliated",
                    reason=f"Greenness on {ns['date']:%b %d} is very low (NDVI {ns['my_ndvi']:.2f}). Late-season differences mostly reflect harvest timing.",
                    why=why[:1], look_for="", conf=conf, conf_why=cw, pct=pct)
    if st_.key in ("late", "post"):
        big_drop = slipping and si < 0.70
        return dict(level="watch" if big_drop else "info", icon="🍂" if not big_drop else "👀",
                    headline="Check for early leaf loss" if big_drop else "Late season — lower greenness is expected",
                    reason=(f"Greenness fell to {pct}% of the best fields — faster than its neighbours. Worth a look for disease or drought leaf loss."
                            if big_drop else f"{st_.note} Use ★ Yield window for how the season went."),
                    why=why, look_for=st_.check, conf=conf, conf_why=cw, pct=pct)

    # ---- the decision
    if si < 0.70 or (si < 0.85 and (persistent or slipping)) or (crit and si < 0.80):
        level, icon, head = "scout", "🔎", "Scout this field"
        bits = [f"it is only {pct}% as green as the best comparable fields"]
        if persistent:
            bits.append(f"it has been behind on {ns['n_behind']} of the last {ns['n_last']} images")
        elif slipping:
            bits.append("it is slipping further behind")
        if crit:
            bits.append("this is the stage that matters most for yield")
        reason = "Because " + (bits[0] if len(bits) == 1 else ", ".join(bits[:-1]) + (", and " if len(bits) > 2 else " and ") + bits[-1]) + "."
    elif si < 0.85 or (si < 0.95 and slipping) or (si < 0.95 and n_comp >= 8 and better_than < 25):
        level, icon, head = "watch", "👀", "Keep an eye on it"
        if si >= 0.85 and better_than < 25 and not slipping:
            reason = (f"It is {pct}% as green as the best fields, but most comparable fields are ahead of it "
                      f"({ns['rank']}{_ord(ns['rank'])} of {ns['n']}). Look again at the next clear image.")
        else:
            reason = (f"It is {pct}% as green as the best fields on this image" + (" and slipping" if slipping else "")
                      + " — not yet a steady pattern. Look again at the next clear image.")
    else:
        level, icon, head = "ok", "✅", "On track — no need to scout for growth"
        reason = (f"It is {pct}% as green as the best comparable fields" + (" and catching up" if gaining else "")
                  + ". Keep your normal scouting routine.")
    return dict(level=level, icon=icon, headline=head, reason=reason, why=why, look_for=st_.check if level != "ok" else "",
                conf=conf, conf_why=cw, pct=pct, patchy=patchy)


def _ord(n: int) -> str:
    return "th" if 10 <= n % 100 <= 20 else {1: "st", 2: "nd", 3: "rd"}.get(n % 10, "th")


# ----------------------------------------------------------------------------- crop-specific "best" index
BEST = "★ Best"
CROP_EVIDENCE = {
    "corn": "Red-edge / chlorophyll indices (CIre, MTCI-type, GCVI) were the most consistent corn yield predictors around the reproductive stage "
            "(Mississippi rainfed trials); swapping red for red-edge beat NDVI, which saturates in dense corn (Brazil, Sentinel-2 + harvester maps); "
            "GNDVI did best at grain fill in Italy; GCVI drives Stanford's SCYM corn yield maps.",
    "soybean": "Peak-season EVI2 had the highest yield correlation for soybean (Croatia; Iowa/Illinois yield-monitor studies); red-edge and "
               "short-wave-infrared (canopy water) bands also carry soybean yield signal — NDMI flags drought stress.",
    "cotton": "NDRE / red-edge chlorophyll indices fitted farm cotton yields best (Uzbekistan); EVI edged NDVI/NDRE in Brazilian harvester maps; "
              "red-edge indices are weak early, so NDVI leads before squaring. Best timing: flowering → boll fill (~75–120 days after sowing).",
    "peanut": "Pods are underground, so healthy green leaf area is the signal: simple-ratio-type indices (WDRVI ≈ simple ratio, red-edge simple ratio) "
              "track peanut leaf area best, NDVI and EVI also predicted yield; mid-season (60–105 days) is the most informative window. "
              "Evidence is thinner than for corn or soybean — the consensus is especially useful here.",
}


def recommended_vi(crop: str, dap: int | float | None = None) -> str:
    """Index research ties most closely to yield for this crop — for the growth stage at `dap` days after planting
    (single-date views), or for the yield-critical stage when dap is None (season views)."""
    if dap is None or pd.isna(dap):
        return critical(crop).vi
    st_ = stage_at(crop, dap)
    if st_ is None or st_.key == "post":
        return critical(crop).vi
    return st_.vi


def resolve_vi(choice: str | None, crop: str, dap=None) -> str:
    """The single index used for SI / "% of the best". ★ Best and ◎ Consensus both use the stage's best index for that."""
    return recommended_vi(crop, dap) if (not choice or choice in (BEST, "◎ Consensus")) else choice


# ----------------------------------------------------------------------------- consensus across indices
# Weight of each index in the consensus, per crop and growth stage — how strongly research ties it to yield at that stage.
# 3 = strongest evidence · 2 = good · 1 = useful · 0.5 = weak (saturates, or sees soil more than crop at that stage).
# Early: canopy is sparse → NDVI/EVI2 (soil-robust) lead, red-edge indices are weak. Reproductive stages: red-edge and
# chlorophyll indices lead in corn and cotton (NDVI saturates), EVI2 in soybean, leaf-area indices (WDRVI ≈ simple ratio) in peanut.
_W = lambda **k: k  # noqa: E731
VI_WEIGHTS = {
    "corn": {
        "early":      _W(NDVI=3, EVI2=3, WDRVI=2, GNDVI=1, GCVI=1, NDRE=0.5, CIre=0.5, NDMI=0.5),
        "vegetative": _W(WDRVI=3, EVI2=2, GCVI=2, NDRE=2, CIre=2, NDVI=1, GNDVI=1, NDMI=1),
        "critical":   _W(CIre=3, NDRE=3, GCVI=2, WDRVI=2, GNDVI=2, EVI2=1, NDMI=1, NDVI=0.5),
        "late":       _W(GCVI=3, GNDVI=3, CIre=2, NDRE=2, WDRVI=1, EVI2=1, NDMI=1, NDVI=0.5),
    },
    "soybean": {
        "early":      _W(NDVI=3, EVI2=3, WDRVI=2, GNDVI=1, GCVI=1, NDRE=0.5, CIre=0.5, NDMI=0.5),
        "vegetative": _W(WDRVI=3, EVI2=3, NDVI=1, GNDVI=1, GCVI=1, NDRE=1, CIre=1, NDMI=1),
        "critical":   _W(EVI2=3, WDRVI=2, NDMI=2, CIre=2, NDRE=2, GCVI=1, GNDVI=1, NDVI=1),
        "late":       _W(NDVI=2, EVI2=2, NDMI=2, WDRVI=1, GNDVI=1, GCVI=1, NDRE=1, CIre=1),
    },
    "cotton": {
        "early":      _W(NDVI=3, EVI2=3, WDRVI=2, GNDVI=1, GCVI=1, NDRE=0.5, CIre=0.5, NDMI=0.5),
        "vegetative": _W(WDRVI=3, EVI2=2, NDVI=2, NDRE=1, CIre=1, GCVI=1, GNDVI=1, NDMI=1),
        "critical":   _W(CIre=3, NDRE=3, EVI2=2, WDRVI=1, GCVI=1, GNDVI=1, NDMI=1, NDVI=0.5),
        "late":       _W(NDVI=2, EVI2=2, NDRE=1, CIre=1, WDRVI=1, GCVI=1, GNDVI=1, NDMI=1),
    },
    "peanut": {
        "early":      _W(NDVI=3, EVI2=3, WDRVI=2, GNDVI=1, GCVI=1, NDRE=0.5, CIre=0.5, NDMI=0.5),
        "vegetative": _W(WDRVI=3, EVI2=2, NDVI=2, GCVI=1, NDRE=1, CIre=1, GNDVI=1, NDMI=1),
        "critical":   _W(WDRVI=3, EVI2=2, NDVI=2, CIre=2, NDRE=1, GCVI=1, GNDVI=1, NDMI=1),
        "late":       _W(NDVI=2, WDRVI=2, EVI2=1, NDMI=1, GNDVI=1, GCVI=1, NDRE=1, CIre=1),
    },
}
CONSENSUS = "◎ Consensus"
VI_EVIDENCE = {   # one line per index: where the yield evidence for it comes from
    "NDVI": "best early (sparse canopy); saturates once rows close",
    "EVI2": "soil- and saturation-robust; peak-season EVI2 tracked soybean yield best (Iowa/Illinois, Croatia); slight edge for cotton in Brazil",
    "WDRVI": "keeps separating dense canopies; best maize yield proxy at peak (Croatia); simple-ratio type indices track peanut leaf area",
    "GNDVI": "chlorophyll-sensitive; best Sentinel-2 index for within-field corn yield at grain fill (Italy)",
    "GCVI": "chlorophyll index behind Stanford's SCYM corn & soybean yield maps",
    "NDRE": "red-edge; highest fit to cotton farm yields (Uzbekistan); red-edge replaces red to beat NDVI saturation in corn (Brazil)",
    "CIre": "red-edge chlorophyll index; among the most consistent corn yield predictors around the reproductive stage (Mississippi rainfed)",
    "NDMI": "canopy water (NIR–SWIR); flags drought stress — SWIR bands carried soybean yield signal",
}


def vi_weights(crop: str, dap) -> dict:
    st_ = stage_at(crop, dap)
    key = st_.key if st_ is not None and st_.key in ("early", "vegetative", "critical", "late") else "critical"
    w = VI_WEIGHTS.get(crop, VI_WEIGHTS["cotton"])[key]
    return {v: float(w.get(v, 1.0)) for v in ALL_VI}


def consensus(per_vi: dict, weights: dict) -> pd.DataFrame:
    """per_vi: {vi: DataFrame[field_id, SI, rank]} — the same fields ranked on each index (same image, same reference rule).
    Each index turns into a percentile (100 = best of the ranked fields); the consensus score is the weight-averaged percentile.
    Rank-based on purpose: indices have different spreads (NDVI saturates, CIre stretches), so their SI values aren't averaged.
    Returns DataFrame[field_id, cons_score, cons_rank, cons_pct, n_vi, rank_<vi>..., pct_<vi>...]."""
    frames = []
    for vi, d in per_vi.items():
        d = d[["field_id", "SI", "rank"]].dropna(subset=["SI"])
        if len(d) < 3:
            continue
        n = len(d)
        p = (1 - (d["rank"].astype(float) - 1) / max(n - 1, 1)) * 100
        frames.append(pd.DataFrame({"field_id": d["field_id"].values, f"rank_{vi}": d["rank"].astype(float).values, f"pct_{vi}": p.values}))
    if not frames:
        return pd.DataFrame(columns=["field_id", "cons_score", "cons_rank", "cons_pct", "n_vi"])
    out = frames[0]
    for f in frames[1:]:
        out = out.merge(f, on="field_id", how="outer")
    vis = [c[4:] for c in out.columns if c.startswith("pct_")]
    P = out[[f"pct_{v}" for v in vis]].to_numpy(dtype=float)
    W = np.array([weights.get(v, 1.0) for v in vis], dtype=float)[None, :] * np.isfinite(P)
    out["n_vi"] = np.isfinite(P).sum(axis=1)
    out["cons_score"] = np.where(W.sum(axis=1) > 0, np.nansum(np.nan_to_num(P) * W, axis=1) / np.maximum(W.sum(axis=1), 1e-9), np.nan)
    ok = out["n_vi"] >= max(2, len(vis) // 2)
    out.loc[~ok, "cons_score"] = np.nan
    r = out["cons_score"].rank(ascending=False, method="first")
    out["cons_rank"] = r
    m = int(out["cons_score"].notna().sum())
    out["cons_pct"] = (1 - (r - 1) / max(m - 1, 1)) * 100
    return out


def agreement(row: pd.Series, weights: dict) -> dict:
    """How the indices see one field: weighted share that put it in the bottom half / bottom quarter, and its rank range."""
    vis = [c[4:] for c in row.index if c.startswith("pct_") and pd.notna(row[c])]
    if not vis:
        return {}
    w = np.array([weights.get(v, 1.0) for v in vis]); p = np.array([row[f"pct_{v}"] for v in vis], dtype=float)
    ranks = [row[f"rank_{v}"] for v in vis]
    return {"n": len(vis), "w_bottom_half": float(w[p < 50].sum() / w.sum()), "w_bottom_quarter": float(w[p < 25].sum() / w.sum()),
            "w_top_half": float(w[p >= 50].sum() / w.sum()), "n_bottom_half": int((p < 50).sum()), "n_bottom_quarter": int((p < 25).sum()),
            "best_rank": int(min(ranks)), "worst_rank": int(max(ranks)),
            "best_vi": vis[int(np.argmax(p))], "worst_vi": vis[int(np.argmin(p))]}
