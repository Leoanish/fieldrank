"""
Run the example fields (two per crop) once, headless, so their imagery / green-up / profile caches exist before you publish.
Visitors who press an example then get results in seconds instead of a minute.

    bash scripts/fieldrank.sh warm
"""
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from streamlit.testing.v1 import AppTest  # noqa: E402

APP = str(ROOT / "app" / "dashboard.py")
for crop, i in [(c, k) for c in ("cotton", "peanut", "corn", "soybean") for k in (0, 1)]:
    t0 = time.time()
    at = AppTest.from_file(APP, default_timeout=1800)
    at.run()
    at.session_state["crop"] = crop
    at.run()
    pick = [r for r in at.sidebar.radio if r.key == f"ex_pick_{crop}"]
    btn = [b for b in at.sidebar.button if b.key == f"ex_load_{crop}"]
    if not pick or not btn:
        print(f"{crop}: example picker not found"); continue
    pick[0].set_value(i).run()
    btn = [b for b in at.sidebar.button if b.key == f"ex_load_{crop}"]
    btn[0].click().run()                         # step 1: comparable fields
    go = [b for b in at.button if b.key == "rank_open"]
    if go:
        go[0].click().run()                      # step 2: "Rank my field" opens the date pop-up …
        d_ = [d for d in at.date_input if d.key == "dlg_date"]
        if d_:                                   # … confirm the example date (what the pop-up's button does)
            at.session_state["check_date"] = str(d_[0].value)
            at.session_state["now_request"] = True
            at.session_state["rank_on"] = True
            at.run()                             # fills the ranking cache too
    if at.exception:
        print(f"{crop}: ERROR {at.exception[0].message}"); continue
    R = at.session_state["result"] if "result" in at.session_state else None
    n = len(R["allf"]) if R else 0
    print(f"✓ {crop} #{i + 1}: {n} fields analysed · ranked on the example date · {time.time() - t0:.0f} s")
print("Done — commit data/timeseries and data/ancillary/field_profile.parquet with:  bash scripts/fieldrank.sh publish")
