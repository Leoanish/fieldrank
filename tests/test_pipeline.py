"""Smoke tests that need no network:  python -m pytest tests -q"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from si_ranking.synthetic import run_demo, synth_fields  # noqa: E402
from si_ranking.fields import fields_in_buffer, nearest_field  # noqa: E402
from si_ranking.imagery import _zonal  # noqa: E402


def test_demo_planting_accuracy():
    m, res = run_demo(n_fields=80, seed=1)
    ok = m["planting_posterior"].notna()
    err = (pd.to_datetime(m.loc[ok, "planting_posterior"]) - pd.to_datetime(m.loc[ok, "true_planting"])).dt.days
    assert ok.mean() > 0.9
    assert err.abs().mean() < 6
    assert res["cohort_size"] >= 3 and 0 < res["target_SI"] <= 1.05


def test_buffer_and_nearest():
    f = synth_fields(30, np.random.default_rng(0))
    sel = fields_in_buffer(f, -83.52, 31.45, 5)
    assert (sel["dist_km"] <= 5).all()
    nf = nearest_field(f, f.loc[3, "lon"], f.loc[3, "lat"])
    assert nf["field_id"] == f.loc[3, "field_id"]


def test_zonal():
    labels = np.array([[1, 1, 2], [2, 0, 0]])
    vals = np.array([[0.2, 0.4, 0.6], [0.8, 0.1, 0.1]])
    valid = np.ones_like(labels, bool)
    med, p10, p90, cnt = _zonal(vals, labels, 2, valid)
    assert cnt.tolist() == [2, 2]
    assert abs(med[0] - 0.3) < 1e-9 and abs(med[1] - 0.7) < 1e-9
