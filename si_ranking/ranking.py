"""
Step 5 — Cohort construction, virtual reference, sufficiency index and rank.

SI_i = VI_i / VI_ref, where VI_ref is the VIRTUAL_REF_QUANTILE (default 95th
percentile) of field-median VI among the *cohort*: fields of the same crop in
the buffer that pass the user's similarity filters (planting window,
irrigation, elevation band, soil drainage class, AWC band). The target field
is always kept in the cohort so it can be ranked.

Everything here is pure pandas so it can be unit-tested without imagery.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from . import config as C


@dataclass
class CohortFilters:
    planting_window_days: int | None = C.PLANTING_WINDOW_DAYS   # None = ignore planting
    same_irrigation: bool = True
    elevation_band_m: float | None = None     # e.g. 15 -> |Δelev| <= 15 m
    same_drainage_class: bool = False
    awc_band: float | None = None             # e.g. 2 cm -> |Δaws0100| <= 2
    min_cohort: int = C.MIN_COHORT_SIZE
    extra_query: str | None = None            # pandas query string on the merged table


def build_table(fields: pd.DataFrame, latest_vi: pd.DataFrame, planting: pd.DataFrame | None,
                ancillary: pd.DataFrame | None) -> pd.DataFrame:
    """One row per field with VI value, planting estimate and ancillary attributes."""
    t = fields.drop(columns=[c for c in ("geometry", "geometry_inner") if c in fields]).copy()
    t = t.merge(latest_vi[["field_id", "date", "median", "p10", "p90", "n_pix"]]
                .rename(columns={"date": "obs_date", "median": "vi_median", "p10": "vi_p10", "p90": "vi_p90"}),
                on="field_id", how="left")
    if planting is not None and len(planting):
        t = t.merge(planting[["field_id", "emergence_date", "planting_posterior", "planting_sd_days", "method"]],
                    on="field_id", how="left")
    if ancillary is not None and len(ancillary):
        t = t.merge(ancillary.drop(columns=[c for c in ("pivot_like",) if c in ancillary]), on="field_id", how="left")
    return t


def apply_cohort(table: pd.DataFrame, target_id: str, filt: CohortFilters) -> pd.DataFrame:
    tgt = table.loc[table["field_id"] == target_id]
    if tgt.empty:
        raise KeyError(f"target {target_id} not in table")
    tgt = tgt.iloc[0]
    m = (table["crop_latest"] == tgt["crop_latest"]) & table["vi_median"].notna()

    if filt.planting_window_days is not None and "planting_posterior" in table and pd.notna(tgt.get("planting_posterior")):
        pd_t = pd.to_datetime(table["planting_posterior"])
        d = (pd_t - pd.Timestamp(tgt["planting_posterior"])).dt.days.abs()
        m &= d.le(filt.planting_window_days) | table["field_id"].eq(target_id)
    if filt.same_irrigation and "irrigated" in table and pd.notna(tgt.get("irrigated")):
        m &= table["irrigated"].eq(tgt["irrigated"]) | table["field_id"].eq(target_id)
    if filt.elevation_band_m is not None and "elev_m" in table and pd.notna(tgt.get("elev_m")):
        m &= (table["elev_m"] - tgt["elev_m"]).abs().le(filt.elevation_band_m) | table["field_id"].eq(target_id)
    if filt.same_drainage_class and "drclassdcd" in table and pd.notna(tgt.get("drclassdcd")):
        m &= table["drclassdcd"].eq(tgt["drclassdcd"]) | table["field_id"].eq(target_id)
    if filt.awc_band is not None and "aws0100wta" in table and pd.notna(tgt.get("aws0100wta")):
        m &= (table["aws0100wta"] - tgt["aws0100wta"]).abs().le(filt.awc_band) | table["field_id"].eq(target_id)
    cohort = table[m].copy()
    if filt.extra_query:
        keep = cohort.query(filt.extra_query)
        cohort = pd.concat([keep, cohort[cohort["field_id"] == target_id]]).drop_duplicates("field_id")
    return cohort


def rank_cohort(cohort: pd.DataFrame, q: float = C.VIRTUAL_REF_QUANTILE) -> tuple[pd.DataFrame, float]:
    """Add SI, rank and percentile columns. Returns (ranked table, virtual reference VI)."""
    v = cohort["vi_median"].astype(float)
    ref = float(np.nanquantile(v, q)) if v.notna().sum() else np.nan
    out = cohort.copy()
    out["vi_ref"] = ref
    out["SI"] = out["vi_median"] / ref
    out["rank"] = out["vi_median"].rank(ascending=False, method="min").astype("Int64")
    out["percentile"] = out["vi_median"].rank(pct=True) * 100
    out["z"] = (out["vi_median"] - v.mean()) / (v.std(ddof=1) if len(v) > 1 else np.nan)
    return out.sort_values("rank"), ref


def rank_field(table: pd.DataFrame, target_id: str, filt: CohortFilters | None = None) -> dict:
    filt = filt or CohortFilters()
    cohort = apply_cohort(table, target_id, filt)
    ranked, ref = rank_cohort(cohort)
    n = len(ranked)
    tgt = ranked[ranked["field_id"] == target_id].iloc[0]
    return {
        "target_id": target_id, "crop": tgt["crop_latest"], "cohort_size": n,
        "cohort_ok": n >= filt.min_cohort, "vi_ref": ref,
        "target_vi": float(tgt["vi_median"]), "target_SI": float(tgt["SI"]),
        "target_rank": int(tgt["rank"]), "target_percentile": float(tgt["percentile"]),
        "obs_date": tgt.get("obs_date"), "ranked": ranked,
    }
