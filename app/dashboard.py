"""
FieldRank — Georgia (corn, cotton, peanut, soybean)

    streamlit run app/dashboard.py

Sidebar : my field (upload / lat-lon / click / draw), crop, planting date, radius → Find comparable fields
          "Compare only with fields that have…" toggles: similar elevation · similar soil · same irrigation
Main    : KPI cards → conical filter funnel + clickable donut → map (comparison / SI) + ranking chart
          → tabs: green-up curves · ranking details · data table
"""
from __future__ import annotations

import datetime as dt
import sys
from concurrent.futures import ThreadPoolExecutor
import threading
import time
from pathlib import Path

import altair as alt
import folium
import geopandas as gpd
import numpy as np
import pandas as pd
import streamlit as st
from folium.plugins import Draw
from shapely.geometry import Point, shape
from streamlit_folium import st_folium

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))      # app/ (crop_icons.py)


def _reload_changed_modules():
    """Streamlit only re-runs this file; modules in si_ranking/ stay in memory. Reload any that changed on disk
    so an update never needs a manual restart."""
    import importlib
    import os
    mt = getattr(sys, "_fieldrank_mtimes", {})
    # every module of the package (config first, so the others re-read it) — a new module can never be left stale
    names = sorted((n for n in list(sys.modules) if n.startswith("si_ranking.")), key=lambda n: (n != "si_ranking.config", n))
    changed = False
    for name in names:
        m = sys.modules.get(name)
        f = getattr(m, "__file__", None) if m else None
        if not f or not os.path.exists(f):
            continue
        t = os.path.getmtime(f)
        if name in mt and mt[name] != t:
            changed = True
        mt[name] = t
    if changed:                       # reload them all: modules import each other, so a partial reload can mix old and new code
        for name in names:
            try:
                importlib.reload(sys.modules[name])
            except Exception:
                pass
    sys._fieldrank_mtimes = mt


_reload_changed_modules()
from si_ranking import config as C  # noqa: E402
from si_ranking.fields import FieldStore  # noqa: E402
import shapely  # noqa: E402
from si_ranking.lite import pick_vi, make_target, score  # noqa: E402
from si_ranking.cohort import planting_cohort, NOMINAL_EMERGENCE_DAYS, DAYS_AFTER  # noqa: E402
from si_ranking.profile import field_profiles, auto_match, lanid_layers, match_flags, MATCH_KEYS  # noqa: E402
from si_ranking.boundary import read_boundary, UPLOAD_TYPES  # noqa: E402
from si_ranking.season import season_series, ts_vs_reference, season_composite, si_trend, rank_on_date, rank_series  # noqa: E402
from si_ranking import growth as G  # noqa: E402
from si_ranking.scenes import refresh_recent  # noqa: E402
from si_ranking import cache as CACHE  # noqa: E402
from si_ranking import plantdate as PD  # noqa: E402

# ============================================================================ page + theme
st.set_page_config(page_title="FieldRank", page_icon="🌱", layout="wide", initial_sidebar_state="auto")
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');
:root { --ink:#0f172a; --muted:#64748b; --line:#e8ebf0; --bg:#f4f6f9; }
html, body, [class*="css"], .stApp { font-family:'Inter',-apple-system,system-ui,sans-serif; color:var(--ink); }
.stApp { background:var(--bg); }
#MainMenu, footer, [data-testid="stToolbar"], [data-testid="stDecoration"], [data-testid="stStatusWidget"] { visibility:hidden; }
/* keep the header (it holds the "open sidebar" arrow) but make it invisible chrome */
header[data-testid="stHeader"] { background:transparent !important; height:2.6rem; pointer-events:none; }
header[data-testid="stHeader"] * { pointer-events:auto; }
[data-testid="stExpandSidebarButton"], [data-testid="stSidebarCollapsedControl"], [data-testid="collapsedControl"],
[data-testid="stSidebarCollapseButton"] { visibility:visible !important; display:flex !important; opacity:1 !important; z-index:1000; }
[data-testid="stExpandSidebarButton"] button, [data-testid="stSidebarCollapsedControl"] button, [data-testid="collapsedControl"] button {
  background:#fff !important; border:1px solid var(--line) !important; border-radius:10px !important; box-shadow:0 4px 12px rgba(15,23,42,.12); }
.block-container { padding-top:1.1rem; padding-bottom:2.5rem; max-width:1560px; }
section[data-testid="stSidebar"] { background:#ffffff; border-right:1px solid var(--line); }
section[data-testid="stSidebar"] .block-container { padding-top:1rem; }
.sb-brand { display:flex; align-items:center; gap:10px; margin:0 0 .6rem; }
.sb-logo { width:34px; height:34px; border-radius:11px; background:linear-gradient(135deg,#22c55e,#065f46); display:grid; place-items:center; color:#fff; font-weight:800; }
.sb-brand b { font-size:1.08rem; letter-spacing:-.01em; }
.sb-h { font-size:.7rem; font-weight:700; letter-spacing:.09em; text-transform:uppercase; color:var(--muted); margin:1.1rem 0 .35rem; }
.sb-note { color:var(--muted); font-size:.78rem; margin:.15rem 0 .4rem; }
.footer { margin:2.2rem 0 .5rem; padding:1rem 0 0; border-top:1px solid var(--line, #e2e8f0); color:var(--muted); font-size:.76rem; line-height:1.5; display:grid; gap:.35rem; }
.stButton > button, .stDownloadButton > button, div[data-testid="stPopover"] > div > button {
  border-radius:999px !important; font-weight:600; padding:.45rem 1.05rem; border:1px solid var(--line); background:#fff; color:var(--ink); }
.stButton > button[kind="primary"] { background:var(--ink); color:#fff; border:none; box-shadow:0 6px 16px rgba(15,23,42,.18); }
.stButton > button[kind="primary"]:hover { background:#1e293b; color:#fff; }
div[data-testid="stPopover"] > div > button { padding:.22rem .8rem; font-size:.78rem; color:var(--muted); }
div[data-testid="stColumn"] .stButton > button:not([kind="primary"]) { padding:.22rem .8rem; font-size:.78rem; color:var(--muted); }
div[data-testid="stVerticalBlockBorderWrapper"] { background:#fff; border:1px solid var(--line) !important; border-radius:22px !important;
  box-shadow:0 1px 2px rgba(16,24,40,.03), 0 10px 28px rgba(16,24,40,.05); }
.stTabs [data-baseweb="tab-list"] { gap:4px; background:#e9edf2; padding:4px; border-radius:999px; width:fit-content; border:none; }
.stTabs [data-baseweb="tab"] { height:36px; padding:0 18px; border-radius:999px; font-weight:600; color:var(--muted); background:transparent; }
.stTabs [aria-selected="true"] { background:#fff; color:var(--ink); box-shadow:0 1px 4px rgba(15,23,42,.12); }
.stTabs [data-baseweb="tab-highlight"], .stTabs [data-baseweb="tab-border"] { display:none; }
.stTabs div[role="tablist"] { gap:4px; background:#e6eaf0; padding:4px; border-radius:999px; width:fit-content; border:none !important; box-shadow:none; margin-bottom:.6rem; }
.stTabs div[role="tablist"] > div[data-testid="stTab"], .stTabs button[role="tab"] { padding:7px 20px !important; border-radius:999px; color:#64748b; border:none !important; background:transparent; }
.stTabs div[data-testid="stTab"] p, .stTabs button[role="tab"] p { font-weight:650; font-size:.9rem; }
.stTabs div[data-testid="stTab"][aria-selected="true"], .stTabs button[role="tab"][aria-selected="true"] { background:#fff; color:#0f172a; box-shadow:0 1px 4px rgba(15,23,42,.14); }
.stTabs div[data-testid="stTab"]::after, .stTabs div[data-testid="stTab"]::before { display:none !important; }
div[data-testid="stTabs"] > div:first-child, .stTabs > div > div:first-child { border-bottom:none !important; }
iframe[title="streamlit_folium.st_folium"] { border-radius:16px; }
section[data-testid="stSidebar"] [data-testid="stFileChip"], section[data-testid="stSidebar"] [data-testid="stFileUploaderFile"],
section[data-testid="stSidebar"] [data-testid="stFileUploaderPagination"],
section[data-testid="stSidebar"] .stFileUploaderFile, section[data-testid="stSidebar"] small[data-testid="stFileUploaderPagination"] { display:none !important; }
section[data-testid="stSidebar"] [data-testid="stFileUploaderDropzoneInstructions"] span { display:none !important; }
section[data-testid="stSidebar"] [data-testid="stFileUploaderDropzone"] { min-height:0; padding:.55rem .7rem; }
.upchip { display:flex; align-items:center; gap:10px; background:#f0fdf4; border:1px solid #bbf7d0; border-radius:14px; padding:9px 12px; margin:.35rem 0 .2rem; }
.upchip .ic { width:30px; height:30px; border-radius:9px; background:#16a34a; color:#fff; display:grid; place-items:center; font-size:15px; flex:none; }
.upchip b { font-size:.84rem; display:block; line-height:1.2; } .upchip span { font-size:.74rem; color:#166534; }
div[data-testid="stDataFrame"] { border-radius:14px; overflow:hidden; }
.hd { display:flex; align-items:center; justify-content:space-between; gap:16px; margin:.1rem 0 1rem; flex-wrap:wrap; }
.hd h1 { font-size:1.9rem; font-weight:800; letter-spacing:-.03em; margin:0; padding:0; }
.hd .sub { color:var(--muted); font-size:.9rem; margin-top:2px; }
.pills { display:flex; gap:8px; flex-wrap:wrap; }
.pill { display:inline-flex; align-items:center; gap:6px; padding:5px 12px; border-radius:999px; background:#fff; border:1px solid var(--line);
  font-size:.78rem; font-weight:600; color:var(--ink); white-space:nowrap; }
.pill.dark { background:var(--ink); color:#fff; border-color:var(--ink); }
.pill.soft { background:#eef2f6; border-color:transparent; color:#334155; }
.pill.g { background:#dcfce7; color:#166534; border-color:transparent; }
.pill.a { background:#fef3c7; color:#92400e; border-color:transparent; }
.pill.r { background:#fee2e2; color:#991b1b; border-color:transparent; }
.pill.b { background:#e0f2fe; color:#075985; border-color:transparent; }
.dot { width:9px; height:9px; border-radius:50%; display:inline-block; }
.kpi { min-height:128px; }
.kpi .lab { color:var(--muted); font-size:.8rem; font-weight:600; }
.kpi .val { font-size:2.15rem; font-weight:800; letter-spacing:-.03em; line-height:1.1; margin:.25rem 0 .2rem; }
.kpi .val small { font-size:.95rem; color:var(--muted); font-weight:700; letter-spacing:0; }
.kpi .sub { color:var(--muted); font-size:.8rem; }
.hero { background:linear-gradient(135deg,#22c55e 0%,#15803d 45%,#064e3b 100%); border-radius:18px; padding:16px 18px; color:#fff; }
.hero .lab, .hero .sub { color:rgba(255,255,255,.88); }
.hero .val small { color:rgba(255,255,255,.82); }
.hero .bar { height:8px; border-radius:999px; background:rgba(255,255,255,.25); margin:.6rem 0 .4rem; overflow:hidden; }
.hero .bar i { display:block; height:100%; background:#fff; border-radius:999px; }
.ctitle { font-weight:700; font-size:1rem; margin:0 0 .1rem; }
.csub { color:var(--muted); font-size:.8rem; margin:0 0 .5rem; }
.legend { display:flex; gap:12px; flex-wrap:wrap; color:var(--muted); font-size:.78rem; margin:.45rem 0 0; }
.legend span { display:inline-flex; align-items:center; gap:6px; }
.note { color:var(--muted); font-size:.78rem; line-height:1.45; margin:.5rem 0 0; }
.note b { color:var(--ink); }
.callout { border-radius:14px; padding:10px 14px; font-size:.85rem; margin:.1rem 0 .9rem; }
.callout.warn { background:#fff7ed; border:1px solid #fed7aa; color:#9a3412; }
.callout.err { background:#fef2f2; border:1px solid #fecaca; color:#991b1b; }
.callout.info { background:#eff6ff; border:1px solid #bfdbfe; color:#1e3a8a; }
.howstrip { display:grid; grid-template-columns:repeat(3, minmax(0,1fr)); gap:.6rem; margin:0 0 .9rem; }
.howstrip .hs { display:flex; gap:.6rem; align-items:flex-start; background:#fff; border:1px solid var(--line); border-radius:14px; padding:.7rem .85rem; font-size:.82rem; color:#334155; line-height:1.45; }
.howstrip .hn { flex:0 0 1.5rem; height:1.5rem; border-radius:999px; background:#0f172a; color:#fff; font-weight:700; font-size:.78rem; display:flex; align-items:center; justify-content:center; }
@media (max-width: 760px) { .howstrip { grid-template-columns:1fr; } }
.st-key-plantcheck { background:#fffbeb; border:1px solid #fcd34d; border-radius:14px; padding:12px 16px; margin:.1rem 0 .9rem; }
.st-key-plantcheck .pc-t { font-weight:700; color:#78350f; margin-bottom:.2rem; }
.st-key-plantcheck .pc-b { font-size:.86rem; color:#78350f; line-height:1.5; }
.datebar { font-size:.9rem; color:#334155; line-height:1.5; padding-top:1.55rem; }
.datebar.warn { color:#92400e; }
.n-eyebrow { font-size:.74rem; font-weight:800; letter-spacing:.08em; text-transform:uppercase; color:#15803d; }
.n-big { font-size:2.6rem; font-weight:800; letter-spacing:-.04em; line-height:1; margin:.35rem 0 .3rem; }
.n-big small { font-size:1rem; color:var(--muted); font-weight:700; letter-spacing:0; }
.n-word { font-size:1.02rem; font-weight:700; margin:.15rem 0 .35rem; }
.n-chg { font-size:.84rem; color:#334155; }
.n-chg b.up { color:#15803d; } .n-chg b.dn { color:#b91c1c; } .n-chg b.fl { color:#475569; }
.tlw { position:relative; margin:30px 2px 4px; }
.tl { display:flex; height:14px; border-radius:999px; overflow:hidden; gap:2px; }
.tl i { display:block; height:100%; }
.tll { display:flex; gap:2px; margin-top:5px; } .tll span { font-size:.68rem; color:var(--muted); text-align:center; overflow:hidden; white-space:nowrap; text-overflow:ellipsis; }
.tll span.on { color:var(--ink); font-weight:800; }
.tlm { position:absolute; top:-26px; transform:translateX(-50%); text-align:center; }
.tlm b { display:inline-block; background:var(--ink); color:#fff; font-size:.7rem; font-weight:700; padding:2px 8px; border-radius:999px; white-space:nowrap; }
.tlm i { display:block; width:2px; height:26px; background:var(--ink); margin:2px auto 0; }
.wbox { background:#f8fafc; border-radius:16px; padding:12px 14px; }
.wbox .lab { color:var(--muted); font-size:.78rem; font-weight:600; } .wbox .val { font-size:1.7rem; font-weight:800; letter-spacing:-.03em; margin:.15rem 0; }
.wbox .val small { font-size:.85rem; color:var(--muted); font-weight:700; letter-spacing:0; } .wbox .sub { color:var(--muted); font-size:.76rem; }
/* fluid feel: soft transitions, hover feedback, no layout jumps */
.stButton > button, div[data-testid="stPopover"] > div > button, .stTabs button[role="tab"], .stTabs div[data-testid="stTab"] { transition:background .15s ease, box-shadow .15s ease, transform .1s ease, color .15s ease; }
.stButton > button:hover { box-shadow:0 4px 12px rgba(15,23,42,.10); }
.stButton > button:active { transform:scale(.98); }
div[data-testid="stVerticalBlockBorderWrapper"] { transition:box-shadow .2s ease; }
.stTabs div[role="tablist"], .stTabs [data-baseweb="tab-list"] { width:100% !important; max-width:100%; overflow-x:auto; scrollbar-width:none; display:flex; }
.stTabs div[role="tablist"] > div[data-testid="stTab"], .stTabs button[role="tab"] { flex:1 1 0; justify-content:center; white-space:nowrap; }
.stTabs [data-baseweb="tab-highlight"], .stTabs [data-baseweb="tab-border"] { display:none !important; }
/* cards side by side: the same height, so rows line up (Filtering, Green-up, Map & ranking, Overview) */
[data-testid="stMain"] [data-testid="stHorizontalBlock"] { align-items:stretch; }
[data-testid="stMain"] [data-testid="stColumn"] > [data-testid="stVerticalBlock"] { height:100%; }
[data-testid="stMain"] [data-testid="stColumn"] > [data-testid="stVerticalBlock"] > [data-testid="stLayoutWrapper"]:only-child { height:100%; }
[data-testid="stMain"] [data-testid="stColumn"] > [data-testid="stVerticalBlock"] > [data-testid="stLayoutWrapper"]:only-child > [data-testid="stVerticalBlock"] { height:100%; }
/* KPI cards: their buttons sit on one line at the bottom */
.st-key-kpirow [data-testid="stLayoutWrapper"] > [data-testid="stVerticalBlock"] { display:flex; flex-direction:column; }
.st-key-kpirow [data-testid="stLayoutWrapper"] > [data-testid="stVerticalBlock"] > :last-child { margin-top:auto; }
.stTabs div[role="tablist"]::-webkit-scrollbar { display:none; }
.cropic { width:18px; height:18px; vertical-align:-4px; margin-right:4px; }
.pill .cropic { width:16px; height:16px; vertical-align:-3px; margin:0; }
.legend .sw { width:12px; height:12px; border-radius:4px; display:inline-block; }
.legend b { color:var(--ink); }
.rankst { font-size:.76rem; color:#334155; line-height:1.4; margin-top:.3rem; }
/* KPI row: every card the same height, whatever it holds */
.st-key-kpirow [data-testid="stHorizontalBlock"] { align-items:stretch !important; }
.st-key-kpirow [data-testid="stColumn"] { display:flex; flex-direction:column; }
.st-key-kpirow [data-testid="stColumn"] > [data-testid="stVerticalBlock"] { flex:1 1 auto; height:100%; }
.st-key-kpirow [data-testid="stColumn"] > [data-testid="stVerticalBlock"] > [data-testid="stLayoutWrapper"] { flex:1 1 auto; height:100%; }
.st-key-kpirow [data-testid="stLayoutWrapper"] > [data-testid="stVerticalBlock"] { height:100%; min-height:300px; }
.st-key-kpirow .kpi { min-height:0; }
.st-key-kpirow [data-testid="stColumn"] [data-testid="stColumn"] button { padding:.3rem .5rem !important; min-width:0; white-space:nowrap; font-size:.8rem; }
.st-key-kpirow .kpi .val { font-size:2rem; }
.rankst.warn { color:#92400e; }
@media (max-width: 1100px) { .kpi .val { font-size:1.8rem; } .n-big { font-size:2.1rem; } .hd h1 { font-size:1.6rem; } }
@media (max-width: 760px) {
  .block-container { padding-left:.8rem; padding-right:.8rem; padding-top:.6rem; }
  .kpi { min-height:0; } .hd { margin-bottom:.6rem; } .pills { gap:6px; } .pill { font-size:.72rem; padding:4px 10px; }
  div[data-testid="stVerticalBlockBorderWrapper"] { border-radius:16px !important; }
  .stTabs div[data-testid="stTab"] p, .stTabs button[role="tab"] p { font-size:.82rem; }
  .stTabs div[role="tablist"] > div[data-testid="stTab"], .stTabs button[role="tab"] { padding:6px 12px !important; }
}
</style>
""", unsafe_allow_html=True)

INK, BLUE, REF_COLOR, SEL_COLOR = "#0f172a", "#2563eb", "#7c3aed", "#0ea5e9"
SI_CLASSES = [("≥ 0.95", 0.95, "#1a9641"), ("0.85–0.95", 0.85, "#a6d96a"), ("0.70–0.85", 0.70, "#fdae61"), ("< 0.70", -1, "#d7191c")]
SI_DOMAIN = [c[0] for c in SI_CLASSES]; SI_RANGE = [c[2] for c in SI_CLASSES]
CATS = [  # (key, label, colour) — one reason per field, shared by the donut and the map
    ("comparable", "Comparable", "#16a34a"),
    ("planting", "Different planting", "#f59e0b"),
    ("uncertain", "Roughly dated (left out)", "#fcd34d"),
    ("irrigation", "Different irrigation", "#ec4899"),
    ("elevation", "Different elevation", "#0ea5e9"),
    ("soil", "Different soil", "#a16207"),
    ("later", "Planted later", "#a78bfa"),
    ("unseen", "Green-up not seen", "#cbd5e1"),
]
CAT_LABEL = {k: lab for k, lab, _ in CATS}; CAT_COLOR = {k: c for k, _, c in CATS}


def si_class(si):
    if si is None or pd.isna(si):
        return None
    for lab, thr, _ in SI_CLASSES:
        if si >= thr:
            return lab
    return SI_CLASSES[-1][0]


def si_color(si):
    c = si_class(si)
    return "#e5e7eb" if c is None else dict(zip(SI_DOMAIN, SI_RANGE))[c]


def html(s):
    st.markdown(s, unsafe_allow_html=True)


def card():
    try:
        return st.container(border=True)
    except TypeError:  # very old Streamlit
        return st.container()


def popover(label):
    return st.popover(label) if hasattr(st, "popover") else st.expander(label)


def segmented(label, options, default, key, format_func=str):
    if hasattr(st, "segmented_control"):
        kw = {} if key in st.session_state else {"default": default}
        v = st.segmented_control(label, options, key=key, label_visibility="collapsed", format_func=format_func, **kw)
        return v if v is not None else default
    kw = {} if key in st.session_state else {"index": options.index(default)}
    return st.radio(label, options, key=key, horizontal=True, label_visibility="collapsed", format_func=format_func, **kw)


# ---------------------------------------------------------------- crop icons (small flat SVGs, used wherever a crop is named)
from crop_icons import ICONS as CROP_SVG  # noqa: E402  (app/crop_icons.py)


def _svg_uri(svg):
    import base64
    return "data:image/svg+xml;base64," + base64.b64encode(svg.encode()).decode()


def crop_img(crop, cls="cropic"):
    return f'<img class="{cls}" src="{_svg_uri(CROP_SVG[crop])}" alt="{crop}">' if crop in CROP_SVG else ""


def _crop_css():
    """Crop picker as a 2 x 2 grid of icon buttons; the example buttons get the same icons."""
    icon = "".join(
        f'.st-key-crop button[role="radio"]:nth-of-type({i + 1}) p::before, .st-key-ex_{c} button p::before, .st-key-ex_load_{c} button p::before '
        f'{{ content:""; display:inline-block; width:20px; height:20px; margin-right:7px; vertical-align:-5px; '
        f'background:url("{_svg_uri(CROP_SVG[c])}") center/contain no-repeat; }}'
        for i, c in enumerate(C.CROPS) if c in CROP_SVG)
    grid = (".st-key-crop, .st-key-crop > div { width:100% !important; }"
            ".st-key-crop [role=radiogroup] { display:grid !important; grid-template-columns:1fr 1fr; gap:6px; width:100%; }"
            ".st-key-crop button[role=radio] { border-radius:12px !important; border:1px solid var(--line) !important; background:#fff; "
            "justify-content:flex-start; padding:.5rem .7rem !important; min-height:2.5rem; margin:0 !important; }"
            ".st-key-crop button[role=radio]:hover { background:#f8fafc; border-color:#cbd5e1 !important; }"
            ".st-key-crop button[role=radio][aria-checked=true] { background:#f0fdf4 !important; border-color:#16a34a !important; box-shadow:0 0 0 1px #16a34a inset; }"
            ".st-key-crop button[role=radio][aria-checked=true] p { color:#14532d; font-weight:700; }")
    st.markdown(f"<style>{icon}{grid}</style>", unsafe_allow_html=True)


_crop_css()


def fmt(v, f=".0f", unit=""):
    try:
        return "—" if v is None or pd.isna(v) else f"{v:{f}}{unit}"
    except (TypeError, ValueError):
        return "—"


# ============================================================================ data + state
@st.cache_resource(show_spinner="Opening the Georgia field index…", max_entries=1)
def _store(_version: str):
    return FieldStore()          # tiny index in memory; field tiles are read only around the chosen field


def _store_version() -> str:
    """Changes whenever the field code or the field tiles change, so a server that keeps running after an update
    never serves an index built by older code."""
    import hashlib
    import si_ranking.fields as _fm
    h = hashlib.md5(Path(_fm.__file__).read_bytes())
    idx = Path(_fm.TILES_DIR) / "index.parquet"
    if idx.exists():
        h.update(str(idx.stat().st_mtime_ns).encode() + str(idx.stat().st_size).encode())
    return h.hexdigest()


try:
    STORE = _store(_store_version())
    if not hasattr(STORE, "label_year"):          # belt and braces: an index from older code
        _store.clear(); STORE = _store(_store_version())
except Exception as e:
    st.error(f"Field tiles missing — run `python -m si_ranking.cli tiles` (or put data/fields/ga_fields.parquet in place). ({e})")
    st.stop()


@st.cache_data(max_entries=12 if C.HOSTED else 64, ttl=3600, show_spinner=False)
def picker_geojson(lon: float, lat: float, radius: float, season_year: int):
    """Fields shown on the picker map, labelled with their crop in the season year — simplified to ~2 m and 5 decimals."""
    near = STORE.near(lon, lat, radius)
    if near.empty:
        return None
    col = STORE.label_year(season_year)[1]
    near = near.assign(crop_latest=near[col] if col in near else near["crop_latest"])
    g = near[["field_id", "crop_latest", "acres", "rotation", "geometry"]].copy()
    g["geometry"] = shapely.set_precision(g.geometry.simplify(0.00002).values, 1e-5)
    g["acres"] = g["acres"].round(0)
    return g.to_json()


FIRST_SEASON = 2024   # oldest season offered: 2024 and 2025 have USDA crop labels; newer seasons use the 2025 labels


# one click examples (2025 season — the crop labels in the field data are 2025).
# Picked from the CSB field layer: labelled with that crop in 2025, 40–160 ac, in the densest same-crop
# area of the state (lots of neighbours to compare with), ≥35 km from metro areas and ≥12 km from towns.
# (field id, place, why, planting, check date)
EXAMPLES = {
    "cotton": [("131825005315166", "Dooly County · near Vienna", "356 cotton fields within 7 km · cotton every year since 2018",
                dt.date(2025, 5, 6), dt.date(2025, 7, 25)),
               ("131825006961251", "Colquitt County · near Doerun", "240 cotton fields within 7 km · continuous cotton",
                dt.date(2025, 5, 8), dt.date(2025, 7, 28))],
    "peanut": [("131825012170614", "Early County · near Donalsonville", "271 peanut fields within 7 km · peanut–cotton rotation",
                dt.date(2025, 5, 8), dt.date(2025, 7, 30)),
               ("131825006455896", "Irwin County · near Ocilla", "241 peanut fields within 7 km · peanut–cotton rotation",
                dt.date(2025, 5, 10), dt.date(2025, 8, 1))],
    "corn": [("131825011700184", "Terrell County · near Dawson", "158 corn fields within 7 km · mostly corn since 2018",
              dt.date(2025, 3, 25), dt.date(2025, 6, 12)),
             ("131825006455913", "Irwin County · near Ocilla", "189 corn fields within 7 km · corn after peanut/cotton",
              dt.date(2025, 3, 28), dt.date(2025, 6, 15))],
    "soybean": [("131825005794950", "Randolph County · near Cuthbert", "96 soybean fields within 7 km (soybean is sparse in GA)",
                 dt.date(2025, 5, 28), dt.date(2025, 8, 12)),
                ("131825012399872", "Brooks County · near Quitman", "71 soybean fields within 7 km",
                 dt.date(2025, 5, 28), dt.date(2025, 8, 12))],
}


def example_center(crop_, i=0):
    g_ = STORE.by_id(EXAMPLES[crop_][i][0])
    return None if g_.empty else (g_.geometry.iloc[0].centroid.y, g_.geometry.iloc[0].centroid.x)


ss = st.session_state
ss.setdefault("geom", None); ss.setdefault("geom_src", None); ss.setdefault("center", (31.45, -83.52)); ss.setdefault("vi_cache", {}); ss.setdefault("season_cache", {}); ss.setdefault("now_cache", {})


def set_geom(geom, src):
    ss.geom, ss.geom_src = geom, src
    ss.center = (geom.centroid.y, geom.centroid.x)
    ss.pop("check_date", None); ss.pop("example", None)


def load_example(crop_, i=0):
    fid, place, _why, pl, asof = EXAMPLES[crop_][i]
    g_ = STORE.by_id(fid)
    if g_.empty:
        return
    r_ = g_.iloc[0]
    reset_results()
    set_geom(r_.geometry, f"CSB field {fid} · {crop_} 2025 · {r_['acres']:.0f} ac")
    ss["crop"], ss["planting"], ss["radius"] = crop_, pl, 7
    ss["_last_crop"] = crop_
    ss.check_date = str(asof); ss.auto_process = True
    ss.example = f"{crop_.title()} field in {place} · 2025 season · check date set to {asof:%b %d, %Y}"


def reset_results():
    for k in ("result", "rank_key", "rank_bars", "donut", "season_key", "season_bars", "season_fetched", "vi_seg", "vi_ts", "vi_season", "now_vi", "rank_on"):
        ss.pop(k, None)
    ss.vi_cache = {}; ss.season_cache = {}; ss.now_cache = {}; ss.pop("now_failed", None)


# ============================================================================ sidebar
with st.sidebar:
    html('<div class="sb-brand"><div class="sb-logo">F</div><b>FieldRank</b></div>')
    html('<div class="sb-h">Your field</div>')
    ss.setdefault("up_nonce", 0)
    ups = st.file_uploader("Upload boundary", type=UPLOAD_TYPES, accept_multiple_files=True, label_visibility="collapsed", key=f"uploader_{ss.up_nonce}",
                           help="Shapefile (select .shp + .shx + .dbf + .prj together, or one .zip) · KML / KMZ · GeoJSON · GeoPackage · "
                                "GML · FlatGeobuf · GeoParquet · CSV with WKT or lat/lon. A point (e.g. a KML pin) picks the field it falls in.")
    if not ups:
        html('<div class="sb-note" style="margin-top:-.35rem">Shapefile (all parts or a .zip) · KML / KMZ · GeoJSON · GeoPackage · CSV — or a pin</div>')
    sig = tuple(sorted((u.name, u.size) for u in ups)) if ups else None
    if sig and sig != ss.get("up_sig"):
        ss.up_sig = sig; ss.pop("up_pick", None)
        try:
            ss.up_feats, ss.up_notes = read_boundary([(u.name, u.getvalue()) for u in ups])
            ss.up_err = None
        except Exception as e:
            ss.up_feats, ss.up_notes, ss.up_err = None, [], str(e)
    if not sig:
        for k_ in ("up_sig", "up_feats", "up_notes", "up_err", "up_pick"):
            ss.pop(k_, None)
    if ss.get("up_err"):
        st.error(ss.up_err)
    feats = ss.get("up_feats")
    if feats is not None and len(feats):
        names = [n for n, _ in sig]
        stems = {}
        for n_ in names:
            stems.setdefault(Path(n_).stem, []).append(Path(n_).suffix.lower())
        groups = []
        for st_, exts_ in stems.items():
            if ".shp" in exts_:
                groups.append(f"{st_}.shp · shapefile ({len(exts_)} files)")
            else:
                groups += [f"{st_}{e}" for e in exts_]
        upname = groups[0] if len(groups) == 1 else f"{len(groups)} files"
        pol_ = feats[feats["kind"] == "polygon"]
        what = (f"{len(pol_)} field{'s' if len(pol_) != 1 else ''} · {pol_['acres'].sum():.0f} ac" if len(pol_) else "point → field it falls in")
        st.markdown(f'<div class="upchip"><div class="ic">✓</div><div><b>{upname}</b><span>{what}</span></div></div>', unsafe_allow_html=True)
        if st.button("Remove upload", key="up_clear", help="Remove the uploaded boundary", use_container_width=True):
            ss.up_nonce += 1
            for k_ in ("up_sig", "up_feats", "up_notes", "up_err", "up_pick"):
                ss.pop(k_, None)
            ss.geom, ss.geom_src = None, None
            reset_results(); st.rerun()
        polys = feats[feats["kind"] == "polygon"]
        if len(polys) > 1:
            opts = [f"{r['name']} · {r['acres']:.0f} ac" for _, r in polys.iterrows()] + [f"All {len(polys)} together (one field)"]
            pick = st.selectbox(f"{len(polys)} fields in the file — which one?", opts, key="up_pick")
            i = opts.index(pick)
            if i < len(polys):
                g_, src_ = polys.geometry.iloc[i], f"upload: {polys['name'].iloc[i]} ({upname})"
            else:
                g_ = polys.geometry.union_all() if hasattr(polys.geometry, "union_all") else polys.geometry.unary_union
                src_ = f"upload: {len(polys)} fields merged ({upname})"
        elif len(polys) == 1:
            g_, src_ = polys.geometry.iloc[0], f"upload: {polys['name'].iloc[0]} ({upname})"
        else:   # only points / lines → the CSB field that contains (or is nearest to) the point
            pt = feats.geometry.iloc[0].representative_point()
            f_ = STORE.at(pt.x, pt.y)
            g_, src_ = f_.geometry, f"CSB field {f_['field_id']} · {f_['crop_latest']} 2025 · {f_['acres']:.0f} ac (from point in {upname})"
            st.caption(f"The upload is a point, so the field it falls in was used ({f_['acres']:.0f} ac).")
        if ss.geom_src != src_:
            set_geom(g_, src_); reset_results()
        for n_ in ss.get("up_notes", []):
            st.caption(f"ℹ️ {n_}")
    # before a field is chosen, the lat/lon boxes point at the selected crop's example field
    _cr = ss.get("crop", "cotton")
    if ss.geom is None and ss.get("_last_crop") != _cr:
        ss["_last_crop"] = _cr
        ss.center = example_center(_cr) or ss.center
    c1, c2 = st.columns(2)
    lat_in = c1.number_input("Latitude", value=float(ss.center[0]), format="%.5f")
    lon_in = c2.number_input("Longitude", value=float(ss.center[1]), format="%.5f")
    if st.button("Use the field at this point", use_container_width=True):
        f = STORE.at(lon_in, lat_in)
        set_geom(f.geometry, f"CSB field {f['field_id']} · {f['crop_latest']} 2025 · {f['acres']:.0f} ac")
        reset_results()
    html('<div class="sb-note">…or click a field / draw a polygon on the map.</div>')

    html('<div class="sb-h">Crop & planting</div>')
    crop = segmented("Crop", C.CROPS, "cotton", "crop", format_func=str.title)
    _exs = EXAMPLES.get(crop, [])
    if _exs:
        with st.expander(f"Try an example {crop} field", expanded=ss.geom is None):
            _lab = [p_ for _, p_, _, _, _ in _exs]
            _i = st.radio("Example", range(len(_exs)), format_func=lambda k: _lab[k], key=f"ex_pick_{crop}", label_visibility="collapsed")
            _f, _p, _w, _pl, _ck = _exs[_i]
            html(f'<div class="sb-note">{_w}. Planted ≈ {_pl:%b %d} · checked {_ck:%b %d, %Y} (2025 season).</div>')
            st.button(f"Load this {crop} field", key=f"ex_load_{crop}", use_container_width=True, on_click=load_example, args=(crop, _i))
    ss.setdefault("planting", dt.date(dt.date.today().year, 5, 5)); ss.setdefault("radius", 7)
    _pmin, _pmax = dt.date(FIRST_SEASON, 1, 1), dt.date(dt.date.today().year, 12, 31)
    if not (_pmin <= ss["planting"] <= _pmax):
        ss["planting"] = min(max(ss["planting"], _pmin), _pmax)
    planting = st.date_input("Planting date", key="planting", min_value=_pmin, max_value=_pmax,
                             help=f"Seasons {FIRST_SEASON}–{dt.date.today().year}. Past seasons compare against the fields that grew "
                                  "your crop that year (USDA crop labels).")
    LBL_YEAR, LBL_COL = STORE.label_year(planting.year)
    LBL_YEAR = LBL_YEAR or 2025
    if planting.year > LBL_YEAR:
        html(f'<div class="sb-note">Crop labels for {planting.year} are not out yet (USDA publishes them after the season) — '
             f'fields are taken as the crop they grew in <b>{LBL_YEAR}</b>; some will have rotated.</div>')
    elif STORE.years:
        html(f'<div class="sb-note">{planting.year} season: fields are matched on the crop they grew in <b>{planting.year}</b>.</div>')
    ss["radius"] = min(ss["radius"], C.MAX_RADIUS_KM)
    radius = st.slider("Search radius (km)", 2, C.MAX_RADIUS_KM, key="radius",
                       help="Every field of your crop inside this circle is compared. Bigger = more fields, a bit slower.")
    process = st.button("Find comparable fields", type="primary", use_container_width=True) or bool(ss.pop("auto_process", False))

    if "result" in ss:
        html('<div class="sb-h">Compare only with fields that have…</div>')
        st.toggle("Similar planting", value=True, disabled=True, help="Always on: estimated planting date within ± the days below of mine")
        window = st.slider("Planting within ± days of mine", 3, 14, C.PLANTING_WINDOW_DAYS, key="plant_window",
                           help="Fields whose estimated planting date is this close to the date you entered — counted in heat-unit days from local weather "
                                "(so a 10-day gap in cool March counts less than in hot June). ±10 days keeps crop stage and size alike; narrow it for a stricter match.")
        _pt = ss["result"]["table"] if "result" in ss else None
        _pc = (_pt.loc[_pt["field_id"] != "MY_FIELD", "plant_conf"].value_counts().to_dict() if _pt is not None and "plant_conf" in _pt else {})
        _plab = {"High confidence": "Well dated only" + (f" · {_pc.get('high', 0)}" if _pc else ""),
                 "All estimated": "+ roughly dated" + (f" · {_pc.get('high', 0) + _pc.get('low', 0)}" if _pc else "")}
        plant_mode = segmented("Which planting dates to use", ["High confidence", "All estimated"], "High confidence", "plant_mode",
                               format_func=lambda k: _plab.get(k, k))
        html('<div class="sb-note" style="margin-top:-.2rem">' + ("<b>Well dated</b>: the satellite saw the crop come up clearly — NDVI and EVI2 agree and "
             "clear images bracket it — so the planting date is good to about a week."
             if plant_mode == "High confidence" else "Adds <b>roughly dated</b> fields: green-up seen but blurred by clouds or a weak signal, "
             "so the date could be off by 1–2 weeks. More fields, less exact.") + '</div>')
        t_irr = st.toggle("Same irrigation", value=False,
                          help="Irrigated vs dryland — LANID 2015–17 + pivot shape" if lanid_layers() else "Irrigated vs dryland — pivot shape only (LANID not installed)")
        t_elev = st.toggle("Similar elevation", value=False, help="Field median elevation within half the local inter-quartile range (3–10 m) of mine")
        t_soil = st.toggle("Similar soil texture", value=False, help="Same USDA texture group, 0–30 cm: coarse · moderately coarse · medium · moderately fine · fine")
        html('<div class="sb-note">Filters apply instantly — no re-processing.</div>')
    else:                                              # nothing to filter yet — defaults until the first search is done
        window, plant_mode, t_irr, t_elev, t_soil = ss.get("plant_window", C.PLANTING_WINDOW_DAYS), (ss.get("plant_mode") or "High confidence"), False, False, False
        html('<div class="sb-note" style="margin-top:.8rem">After the search, filters appear here — planting window, irrigation, '
             'elevation, soil — and apply instantly.</div>')

    with st.expander("Advanced"):
        allow_l = st.toggle("Add Landsat when Sentinel-2 is cloudy", value=True)
        max_days = st.slider("Look back for a clear image (days)", 5, 60, 30)
        q = st.slider("Reference quantile", 0.80, 0.99, C.VIRTUAL_REF_QUANTILE, 0.01)
        CACHE.prune()                                   # keeps the imagery cache under its size budget (checks every 5 min)
        cu = CACHE.usage()
        html(f'<div class="sb-note" style="margin-top:.6rem"><b>Imagery cache</b> · {cu["mb"]:.0f} MB · {cu["scenes"]} scenes '
             f'(limit {C.CACHE_MAX_MB:,} MB, oldest removed first). Re-checking a field or a neighbour reuses it.</div>')
        if st.button("Clear imagery cache", key="cache_clear", use_container_width=True,
                     help="Deletes the stored satellite results. Nothing breaks — the next run just reads the images again."):
            freed = CACHE.clear(); ss.vi_cache = {}; ss.season_cache = {}; ss.now_cache = {}
            st.toast(f"Cleared {freed:.0f} MB of cached imagery.", icon="🧹")
    if "result" in ss and st.button("↺ Start over with a new field", use_container_width=True):
        reset_results(); ss.pop("check_date", None); ss.pop("example", None); st.rerun()

def footer():
    """Data sources, how to cite and the usual caveat — on every page."""
    html('<div class="footer"><div><b>FieldRank</b> · decision support for Georgia corn, cotton, peanut and soybean. It shows <i>where</i> to scout, '
         'not <i>why</i> a field is behind — a field visit tells you the cause. Not a yield estimate and not agronomic advice.</div>'
         '<div>Data: USDA NASS Crop Sequence Boundaries &amp; Cropland Data Layer (field boundaries and crops) · ESA Copernicus Sentinel-2 '
         '(10 m, via Element 84 Earth Search) · NASA/USGS Landsat 8–9 (green-up only) · Open-Meteo weather · USDA SSURGO soils · '
         'Copernicus GLO-30 elevation · LANID irrigation. Imagery © ESA / USGS; basemap Esri, Maxar.</div></div>')


lat0, lon0 = ss.center
has_field = ss.geom is not None
has_result = "result" in ss
VI = G.resolve_vi(ss.get("vi_seg"), crop)

# ============================================================================ header
R0 = ss.get("result", {})
hdr = (f'<span class="pill dark">{crop_img(R0.get("crop") or crop)} {(R0.get("crop") or crop).title()}</span>'
       f'<span class="pill">Planted {R0.get("planting", planting)}</span>'
       f'<span class="pill">{R0.get("radius", radius)} km radius</span>')
if has_field:
    hdr += f'<span class="pill soft">{(ss.geom_src or "")[:46]}</span>'
html(f'<div class="hd"><div><h1>Field ranking</h1><div class="sub">How well is my field doing compared with similar, comparable fields?</div></div>'
     f'<div class="pills">{hdr}</div></div>')
if ss.get("example"):
    html(f'<div class="callout info"><b>Example:</b> {ss.example}. Press <b>Rank my field</b> for the rank and the scout answer — everything is live: switch filters, change the date, explore. '
         f'To check your own field, press <b>↺ Start over</b> in the sidebar.</div>')
progress_slot = st.container()
if ss.pop("stopped", False):
    st.toast("Stopped. Everything already read is kept — running again picks up where it stopped.", icon="⏹️")
    html('<div class="callout warn"><b>Stopped.</b> Nothing was lost: images, green-up and profiles read so far are cached, so pressing '
         '<b>Find comparable fields</b> or <b>Rank my field</b> again continues quickly from there.</div>')


@st.cache_resource
def _job_gate():
    """Shared by every visitor of this server: at most MAX_CONCURRENT_JOBS imagery jobs at once."""
    return threading.BoundedSemaphore(C.MAX_CONCURRENT_JOBS)


def free_memory():
    """Give memory back after a big job (numpy/GDAL buffers) so a long-running server doesn't creep up."""
    import gc
    gc.collect()
    try:
        import ctypes
        ctypes.CDLL("libc.so.6").malloc_trim(0)          # Linux (the hosted server); harmless no-op elsewhere
    except Exception:
        pass


def run_heavy(_show, fn, *a, **k):
    """Run an imagery job when a slot is free; meanwhile show the user they are in line (■ Stop still works)."""
    gate = _job_gate(); t0 = time.time()
    while not gate.acquire(timeout=0.5):
        _show(0.0, f"The server is busy with other users' searches — you're in line ({time.time() - t0:.0f} s). It starts by itself.")
    try:
        return fn(*a, **k)
    finally:
        gate.release()
        free_memory()


OUTAGE_MSG = ("The satellite image service (Element 84 Earth Search on AWS) did not answer. This is usually a short outage on their side — "
              "nothing is wrong with your field or settings. Please try again in a few minutes; everything already read is kept.")


def outage(status, what: str, e: Exception):
    """Show a calm, plain message instead of a traceback when an outside service is down."""
    try:
        status.update(label=f"Could not {what}", state="error", expanded=False)
    except Exception:
        pass
    st.warning(f"**Could not {what}.** {OUTAGE_MSG}", icon="🛰️")
    with st.expander("Technical detail"):
        st.code(f"{type(e).__name__}: {str(e)[:400]}")


def _stopped():
    ss.stopped = True
    ev = ss.get("_cancel_evt")
    if ev is not None:
        ev.set()


def status_panel(title):
    status = progress_slot.status(title, expanded=True)
    c_, b_ = status.columns([4, 1])
    with b_:
        st.button("■ Stop", key=f"stop_{abs(hash(title)) % 10**8}", on_click=_stopped, use_container_width=True,
                  help="Stop now. Images and profiles already read are kept, so running again resumes quickly.")
    with c_:
        st.caption("Working… you can stop at any time.")
    bar = status.progress(0.0); line = status.empty()

    def step(msg): status.write(msg)

    def progress(frac, msg): bar.progress(min(max(frac, 0.0), 1.0)); line.caption(msg)

    def done(label): bar.progress(1.0); line.empty(); status.update(label=label, state="complete", expanded=False)
    return status, step, progress, done


def circle_view(la, lo, rad_km, height_px=580):
    """Leaflet zoom at which the whole search circle (+10 %) fits the map height."""
    m_per_px = 2.2 * rad_km * 1000 / max(height_px - 40, 200)
    z = int(np.floor(np.log2(156543.03 * np.cos(np.radians(la)) / max(m_per_px, 1e-6))))
    return [la, lo], int(min(max(z, 8), 16))


def base_map(center=None, zoom=None, rad=None, height_px=580):
    la, lo = center or (lat0, lon0)
    zoom = zoom or circle_view(la, lo, rad or radius, height_px)[1]
    m = folium.Map(location=[la, lo], zoom_start=zoom, tiles=None, control_scale=True)
    folium.TileLayer("https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}", attr="Esri, Maxar", name="Satellite").add_to(m)
    folium.TileLayer("https://server.arcgisonline.com/ArcGIS/rest/services/Reference/World_Boundaries_and_Places/MapServer/tile/{z}/{y}/{x}",
                     attr="Esri", name="Roads & places", overlay=True, control=True, opacity=0.9).add_to(m)
    r_ = (rad or radius) * 1000
    folium.Circle([la, lo], radius=r_, color="#000", weight=4, opacity=0.25, fill=False).add_to(m)
    folium.Circle([la, lo], radius=r_, color="#ffffff", weight=1.6, fill=False, dash_array="8 6").add_to(m)
    dlat = r_ / 111_320.0; dlon = r_ / (111_320.0 * max(np.cos(np.radians(la)), 0.2))
    m.fit_bounds([[la - dlat, lo - dlon], [la + dlat, lo + dlon]], padding=(12, 12))   # whole search circle in view
    return m


def label_marker(m, lat, lon, text, bg="#ffffff", fg="#0f172a"):
    """Small, readable label on top of a field (short name or rank)."""
    folium.Marker([lat, lon], icon=folium.DivIcon(html=f'<div style="transform:translate(-50%,-50%);display:inline-block;background:{bg};color:{fg};'
                                                     f'font:700 10px Inter,sans-serif;padding:1px 6px;border-radius:999px;white-space:nowrap;'
                                                     f'box-shadow:0 1px 3px rgba(0,0,0,.45)">{text}</div>', icon_size=(0, 0))).add_to(m)


def pill_marker(m, lat, lon, text, color, anchor=(18, 10)):
    folium.Marker([lat, lon], icon=folium.DivIcon(html=f'<div style="background:{color};color:#fff;font:700 11px Inter,sans-serif;padding:3px 9px;'
                                                     f'border-radius:999px;white-space:nowrap;box-shadow:0 2px 6px rgba(0,0,0,.3)">{text}</div>', icon_anchor=anchor)).add_to(m)



def rank_bar_chart(coh_df: pd.DataFrame, ref_v: float, vi: str, key: str, value_label: str):
    """Horizontal SI bars, rank 1 on top; ★ ME / ◆ REF labels; click → selection under `key`."""
    bars = coh_df[["rank", "field_id", "SI", "vi_median", "is_ref_field", "acres"] + [c for c in ("short", "where", "est_planting", "n_obs") if c in coh_df]].copy()
    bars["is_me"] = bars["field_id"] == "MY_FIELD"
    nref = int(bars["is_ref_field"].sum())
    first_ref = bars.loc[bars["is_ref_field"], "rank"].min() if nref else None
    few = len(bars) <= 30
    ref_lab = np.where(bars["is_ref_field"] & (few | (bars["rank"] == first_ref)), "◆ REF" if few else f"◆ {nref} REF", "")
    bars["label"] = np.where(bars["is_me"], "★ ME", ref_lab)
    bars["role"] = np.where(bars["is_me"], "my field", np.where(bars["is_ref_field"], "reference field", "neighbour"))
    bars["si_class"] = bars["SI"].apply(si_class); bars["rank"] = bars["rank"].astype(int)
    bars["outline"] = np.where(bars["is_me"], BLUE, np.where(bars["is_ref_field"], REF_COLOR, "transparent"))
    bars["outline_w"] = np.where(bars["is_me"] | bars["is_ref_field"], 3, 0)
    bars["label_color"] = np.where(bars["is_me"], BLUE, REF_COLOR)
    if "est_planting" in bars:
        bars["est_planting"] = bars["est_planting"].astype(str)
    pick = alt.selection_point(name="pick", fields=["field_id"], on="click", clear="dblclick")
    y_enc = alt.Y("rank:O", sort="ascending", title=None, axis=alt.Axis(labelFontSize=10, ticks=False, domain=False, labelColor="#64748b"))
    tips = ([alt.Tooltip("short:N", title="field"), alt.Tooltip("where:N", title="where")] if "short" in bars else []) + [
        alt.Tooltip("rank:O"), alt.Tooltip("role:N", title=""), alt.Tooltip("SI:Q", format=".2f"), alt.Tooltip("vi_median:Q", format=".3f", title=value_label)]
    tips += [alt.Tooltip("est_planting:N", title="est. planting")] if "est_planting" in bars else []
    tips += [alt.Tooltip("n_obs:Q", title="clear looks")] if "n_obs" in bars else []
    tips += [alt.Tooltip("acres:Q", format=".0f")]
    ch = alt.Chart(bars).mark_bar(cornerRadiusEnd=6, cursor="pointer").encode(
        y=y_enc, x=alt.X("SI:Q", title="sufficiency index", scale=alt.Scale(domain=[0, max(1.12, float(bars["SI"].max()) + 0.08)]), axis=alt.Axis(grid=False)),
        color=alt.Color("si_class:N", scale=alt.Scale(domain=SI_DOMAIN, range=SI_RANGE), legend=None),
        stroke=alt.Stroke("outline:N", scale=None, legend=None), strokeWidth=alt.StrokeWidth("outline_w:Q", scale=None, legend=None),
        opacity=alt.condition(pick, alt.value(1.0), alt.value(0.35)), tooltip=tips).add_params(pick)
    tx = alt.Chart(bars[bars["label"] != ""]).mark_text(align="left", dx=5, fontSize=11, fontWeight="bold").encode(
        y=y_enc, x="SI:Q", text="label:N", color=alt.Color("label_color:N", scale=None, legend=None))
    rl = alt.Chart(pd.DataFrame({"x": [1.0]})).mark_rule(color=REF_COLOR, strokeDash=[4, 3], strokeWidth=2).encode(x="x:Q")
    chart = (ch + tx + rl).properties(height=max(320, min(17 * len(bars) + 40, 560))).configure_view(strokeWidth=0)
    try:
        st.altair_chart(chart, use_container_width=True, on_select="rerun", selection_mode="pick", key=key)
    except TypeError:
        st.altair_chart(chart, use_container_width=True)


CONS_CLASSES = [("top 25 %", 75, "#1a9641"), ("50–75 %", 50, "#a6d96a"), ("25–50 %", 25, "#fdae61"), ("bottom 25 %", -1, "#d7191c")]


def cons_class(p):
    if p is None or pd.isna(p):
        return None
    return next(lab for lab, thr, _ in CONS_CLASSES if p >= thr)


def cons_color(p):
    c = cons_class(p)
    return "#e5e7eb" if c is None else {lab: col for lab, _, col in CONS_CLASSES}[c]


def consensus_bar_chart(coh_df: pd.DataFrame, key: str):
    """Consensus score bars (weighted percentile over all indices), best on top; ★ ME marked; click → zoom on the map."""
    bars = coh_df[coh_df["cons_score"].notna()].copy().sort_values("cons_rank")
    bars["is_me"] = bars["field_id"] == "MY_FIELD"
    bars["cons_rank"] = bars["cons_rank"].astype(int)
    bars["cls"] = bars["cons_pct"].apply(cons_class)
    bars["label"] = np.where(bars["is_me"], "★ ME", "")
    bars["outline"] = np.where(bars["is_me"], BLUE, "transparent"); bars["outline_w"] = np.where(bars["is_me"], 3, 0)
    rk_cols = [c for c in bars if c.startswith("rank_") and c[5:] in G.ALL_VI]
    bars["spread"] = bars[rk_cols].min(axis=1).astype("Int64").astype(str) + "–" + bars[rk_cols].max(axis=1).astype("Int64").astype(str) if rk_cols else ""
    pick = alt.selection_point(name="pick", fields=["field_id"], on="click", clear="dblclick")
    y_enc = alt.Y("cons_rank:O", sort="ascending", title=None, axis=alt.Axis(labelFontSize=10, ticks=False, domain=False, labelColor="#64748b"))
    tips = ([alt.Tooltip("short:N", title="field"), alt.Tooltip("where:N", title="where")] if "short" in bars else []) + [
        alt.Tooltip("cons_rank:O", title="consensus rank"), alt.Tooltip("cons_score:Q", format=".0f", title="score /100"),
        alt.Tooltip("spread:N", title="rank range over indices"), alt.Tooltip("n_vi:Q", title="indices")]
    ch = alt.Chart(bars).mark_bar(cornerRadiusEnd=6, cursor="pointer").encode(
        y=y_enc, x=alt.X("cons_score:Q", title="consensus score (stage-weighted percentile over all indices)", scale=alt.Scale(domain=[0, 105]), axis=alt.Axis(grid=False)),
        color=alt.Color("cls:N", scale=alt.Scale(domain=[c[0] for c in CONS_CLASSES], range=[c[2] for c in CONS_CLASSES]), legend=None),
        stroke=alt.Stroke("outline:N", scale=None, legend=None), strokeWidth=alt.StrokeWidth("outline_w:Q", scale=None, legend=None),
        opacity=alt.condition(pick, alt.value(1.0), alt.value(0.35)), tooltip=tips).add_params(pick)
    tx = alt.Chart(bars[bars["label"] != ""]).mark_text(align="left", dx=5, fontSize=11, fontWeight="bold", color=BLUE).encode(y=y_enc, x="cons_score:Q", text="label:N")
    chart = (ch + tx).properties(height=max(320, min(17 * len(bars) + 40, 560))).configure_view(strokeWidth=0)
    try:
        st.altair_chart(chart, use_container_width=True, on_select="rerun", selection_mode="pick", key=key)
    except TypeError:
        st.altair_chart(chart, use_container_width=True)


def bar_selection(key: str):
    try:
        pts = (ss.get(key) or {}).get("selection", {}).get("pick", [])
        return pts[0].get("field_id") if pts else None
    except Exception:
        return None


def si_map(g_rank: gpd.GeoDataFrame, vi_label: str, key: str, sel_id=None, center=None, rad=None, height=560, consensus=False):
    """SI choropleth of ranked fields with REF / ME pills; zooms to `sel_id` if given."""
    mm = base_map(center=center, rad=rad, height_px=height)
    gg = g_rank[g_rank["field_id"] != "MY_FIELD"].copy()
    for c_ in ("SI", "vi_median"):
        gg[c_] = pd.to_numeric(gg[c_], errors="coerce").round(3)
    if consensus:
        gg["cons_pct"] = pd.to_numeric(gg.get("cons_pct"), errors="coerce")
        gg["rank"] = gg.get("cons_rank"); gg["SI"] = pd.to_numeric(gg.get("cons_score"), errors="coerce").round(0)
    gg["rank"] = gg["rank"].astype("Float64")
    if "short" not in gg:
        gg["short"] = gg["field_id"]; gg["where"] = ""
    gg["acres"] = pd.to_numeric(gg["acres"], errors="coerce").round(0)
    folium.GeoJson(gg[["field_id", "short", "where", "SI", "rank", "vi_median", "in_cohort", "is_ref_field", "acres", "geometry"] + (["cons_pct"] if consensus else [])].to_json(),
                   style_function=lambda f: {"fillColor": (cons_color(f["properties"]["cons_pct"]) if consensus else si_color(f["properties"]["SI"])) if f["properties"]["in_cohort"] else "#e5e7eb",
                                             "color": REF_COLOR if f["properties"]["is_ref_field"] else "#1f2937",
                                             "weight": 3 if f["properties"]["is_ref_field"] else (1.0 if f["properties"]["in_cohort"] else 0.3),
                                             "fillOpacity": 0.78 if f["properties"]["in_cohort"] else 0.12},
                   highlight_function=lambda f: {"weight": 3.5, "color": "#facc15"},
                   tooltip=folium.GeoJsonTooltip(fields=["short", "where", "rank", "SI"], aliases=["field", "where", "rank", "score /100" if consensus else "SI"]),
                   popup=folium.GeoJsonPopup(fields=["short", "where", "rank", "SI", "vi_median", "acres", "field_id"],
                                             aliases=["field", "where", "rank", "SI", vi_label, "acres", "CSB id"])).add_to(mm)
    coh_rows = gg[gg["in_cohort"] & gg["rank"].notna() & ~gg["is_ref_field"]]
    if len(coh_rows) <= 150:
        for _, r_ in coh_rows.iterrows():
            rp = r_.geometry.representative_point(); label_marker(mm, rp.y, rp.x, f"#{int(r_['rank'])}")
    for _, rf in g_rank[g_rank["is_ref_field"]].iterrows():
        cc = rf.geometry.centroid; pill_marker(mm, cc.y, cc.x, f"REF · #{int(rf['rank'])}", REF_COLOR, (22, 10))
    centre, zoom = circle_view(*(center or (lat0, lon0)), rad or radius, height)     # default: the whole search circle
    if sel_id is not None and (g_rank["field_id"] == sel_id).any():
        srow = g_rank[g_rank["field_id"] == sel_id].iloc[0]; cc = srow.geometry.centroid
        folium.GeoJson(srow.geometry.__geo_interface__, style_function=lambda f: {"fillColor": "none", "color": SEL_COLOR, "weight": 5, "dashArray": "6 4"}).add_to(mm)
        pill_marker(mm, cc.y, cc.x, f"#{int(srow['rank']) if pd.notna(srow['rank']) else '—'} · SI {fmt(srow['SI'], '.2f')}", SEL_COLOR, (30, 28))
        centre, zoom = [cc.y, cc.x], 16 if srow["acres"] < 60 else 15
    me_ = g_rank[g_rank["field_id"] == "MY_FIELD"]
    folium.GeoJson(me_[["field_id", "geometry"]].to_json(), style_function=lambda f: {"fillColor": "none", "color": "#60a5fa", "weight": 4}).add_to(mm)
    mc = me_.geometry.iloc[0].centroid
    r_ = me_["cons_rank"].iloc[0] if consensus and "cons_rank" in me_ else me_["rank"].iloc[0]
    pill_marker(mm, mc.y, mc.x, "ME" + (f" · #{int(r_)}" if pd.notna(r_) else ""), BLUE)
    folium.LayerControl(collapsed=True).add_to(mm)
    st_folium(mm, height=height, use_container_width=True, key=key, returned_objects=[], center=centre, zoom=zoom)
    html('<div class="legend">' + ("".join(f'<span><i class="dot" style="background:{c}"></i>consensus {lab}</span>' for lab, _, c in CONS_CLASSES) if consensus else
                                   "".join(f'<span><i class="dot" style="background:{c}"></i>SI {lab}</span>' for lab, _, c in SI_CLASSES))
         + f'<span><i class="dot" style="background:{REF_COLOR}"></i>reference field</span><span><i class="dot" style="background:{BLUE}"></i>my field</span></div>')


# ============================================================================ before processing: pick the field
if not has_result:
    with card():
        e0, e1 = st.columns([1.15, 2])
        with e0:
            html('<div class="ctitle" style="font-size:1.1rem">New here? Try an example field</div>'
                 '<div class="csub" style="margin:0">One click loads a real Georgia field from the 2025 season and finds its comparable fields. '
                 'Then press <b>Rank my field</b> — the date is already set — for the rank and the “should I scout?” answer.</div>')
        with e1:
            ec = st.columns(4)
            for col_, (cr_, exs_) in zip(ec, EXAMPLES.items()):
                col_.button(cr_.title(), key=f"ex_{cr_}", help=f"{exs_[0][1]} · {exs_[0][2]}",
                            use_container_width=True, on_click=load_example, args=(cr_,))
    html(f'<div class="pills" style="margin:.2rem 0 .8rem"><span class="pill {"g" if has_field else "dark"}">1 · Pick your field {"✓" if has_field else ""}</span>'
         f'<span class="pill {"dark" if has_field else "soft"}">2 · Crop & planting date</span><span class="pill soft">3 · Find comparable fields</span></div>')
    with card():
        m = base_map()
        gj = picker_geojson(round(lon0, 4), round(lat0, 4), radius, planting.year)
        if gj:
            folium.GeoJson(gj, name="Fields",
                           style_function=lambda f: {"color": "#4ade80" if f["properties"]["crop_latest"] == crop else "#e2e8f0", "weight": 1.2,
                                                     "fillColor": "#22c55e" if f["properties"]["crop_latest"] == crop else "#cbd5e1", "fillOpacity": 0.18},
                           highlight_function=lambda f: {"weight": 3, "color": "#ffffff", "fillOpacity": 0.35},
                           tooltip=folium.GeoJsonTooltip(fields=["field_id", "crop_latest", "acres", "rotation"], aliases=["field", f"crop {LBL_YEAR}", "acres", "rotation 2018→"])).add_to(m)
        if has_field:
            folium.GeoJson(gpd.GeoSeries([ss.geom]).__geo_interface__, style_function=lambda f: {"fillColor": "none", "color": "#60a5fa", "weight": 4}).add_to(m)
            pill_marker(m, ss.geom.centroid.y, ss.geom.centroid.x, "MY FIELD", BLUE)
        Draw(export=False, draw_options={"polyline": False, "circle": False, "marker": False, "circlemarker": False}).add_to(m)
        folium.LayerControl().add_to(m)
        out = st_folium(m, height=600, use_container_width=True, key="picker", returned_objects=["last_active_drawing", "last_clicked"])
        html(f'<div class="legend"><span><i class="dot" style="background:#16a34a"></i>{crop} fields ({LBL_YEAR})</span><span><i class="dot" style="background:#94a3b8"></i>other crops</span>'
             f'<span>◌ {radius} km search radius</span><span>click a field or draw your boundary</span></div>')
    if out and out.get("last_active_drawing"):
        g = shape(out["last_active_drawing"]["geometry"])
        if ss.geom is None or not g.equals(ss.geom):
            set_geom(g, "drawn polygon"); st.rerun()
    elif out and out.get("last_clicked"):
        p = out["last_clicked"]; f = STORE.at(p["lng"], p["lat"])
        if f.geometry.contains(Point(p["lng"], p["lat"])) and (ss.geom is None or not f.geometry.equals(ss.geom)):
            set_geom(f.geometry, f"CSB field {f['field_id']} · {f['crop_latest']} 2025 · {f['acres']:.0f} ac"); st.rerun()

# ============================================================================ process
if process:
    if ss.geom is None:
        st.sidebar.warning("Pick, draw or upload your field first."); st.stop()
    geom = ss.geom; c = geom.centroid
    tgt = make_target(geom, crop)
    src = ss.geom_src or ""
    hit = STORE.by_id(src.split(" ")[2]) if src.startswith("CSB field") else STORE.by_id("-")
    if len(hit):
        tgt["pivot_like"] = bool(hit.iloc[0]["pivot_like"])
    buf = STORE.near(c.x, c.y, radius, crop, planting.year)          # every field growing this crop in this season's labels
    buf = buf[~buf.geometry.intersects(geom)]                          # never my own field (or a polygon overlapping it)
    n_crop_total = len(buf)
    buf = buf[buf["acres"] >= C.CROP_PARAMS[crop].min_acres_10m]
    n_capped = 0
    allf = gpd.GeoDataFrame(pd.concat([tgt, buf], ignore_index=True), geometry="geometry", crs="EPSG:4326")
    allf["geometry_inner"] = gpd.GeoSeries(allf["geometry_inner"], crs="EPSG:4326")
    allf = allf[~allf["geometry_inner"].is_empty].reset_index(drop=True)
    t0 = dt.datetime.now()
    status, step, prog, done = status_panel(f"Finding comparable {crop} fields…")
    step(f"**Fields** — {n_crop_total} {crop} fields within {radius} km ({LBL_YEAR} crop labels) · {len(allf) - 1} ≥ {C.CROP_PARAMS[crop].min_acres_10m:.0f} ac kept")
    # elevation / soil / irrigation are fetched in the background while the imagery is read
    pool = ThreadPoolExecutor(max_workers=1)
    ss._cancel_evt = threading.Event()                 # set by ■ Stop → the background profile job stops too
    fut_prof = pool.submit(field_profiles, allf, lambda s: None, None, 3, True, ss._cancel_evt)
    try:
        coh, info, gts = run_heavy(prog, planting_cohort, allf, "MY_FIELD", planting, window, log=lambda s: None, progress=prog, allow_landsat=allow_l)
    except Exception as e:
        ss._cancel_evt.set(); pool.shutdown(wait=False)
        outage(status, "read the satellite images", e)
        st.stop()
    sc = info["status_counts"]
    step(f"**Green-up** — {info['n_scenes']} clear Sentinel-2 dates" + (f" + {info['n_landsat']} Landsat" if info["n_landsat"] else "") +
         f" · seen on {sc.get('crossing', 0)} fields · mine on {info['anchor_greenup']} ({info['target_conf_label']} confidence)")
    t_w = dt.datetime.now()
    while not fut_prof.done():                              # poll (not block) so ■ Stop can interrupt here too
        prog(0.99, f"Finishing elevation / soil / irrigation… {(dt.datetime.now() - t_w).total_seconds():.0f} s")
        time.sleep(0.4)
    try:
        prof = fut_prof.result()
        diag = prof.attrs.get("diag", {})
    except Exception as e:
        prof = pd.DataFrame({"field_id": allf["field_id"], "irrigated": allf["pivot_like"].astype(bool), "irr_source": "pivot shape"})
        diag = {"n": len(allf), "n_elev": 0, "n_soil": 0, "n_lanid": 0, "errors": [f"{type(e).__name__}: {e}"[:200]], "elev_src": "", "soil_src": ""}
    pool.shutdown(wait=False)
    step(f"**Field profiles** — elevation {diag.get('n_elev', 0)}/{len(allf)} ({diag.get('elev_src') or '—'}) · "
         f"soil {diag.get('n_soil', 0)}/{len(allf)} ({diag.get('soil_src') or '—'}) · irrigation layer {diag.get('n_lanid', 0)}/{len(allf)}")
    for e in diag.get("errors", [])[:3]:
        step(f"⚠️ {e}")
    table = allf.drop(columns=["geometry", "geometry_inner"]).merge(coh.drop(columns=[c for c in ("irrigated",) if c in coh]), on="field_id", how="left")
    table = table.merge(prof.drop_duplicates("field_id"), on="field_id", how="left")
    table["irrigated"] = table["irrigated"].fillna(table["pivot_like"]).astype(bool)
    prog(0.995, "Estimating planting dates from green-up + local weather…")
    wx = PD.weather_for(c.x, c.y, planting, log=lambda s_: None)
    pdet, pinfo = PD.estimate(coh, crop, planting, "MY_FIELD", wx)
    table = table.drop(columns=[c_ for c_ in ("est_planting",) if c_ in table]).merge(
        pdet[["field_id", "est_planting", "plant_lo", "plant_hi", "plant_unc", "plant_conf", "plant_why", "lag_days"] + (["delta_tdays"] if "delta_tdays" in pdet else [])], on="field_id", how="left")
    lag_days = pinfo.get("my_lag_days")
    step(f"**Planting dates** — {pinfo['n_high']} well dated · {pinfo['n_low']} roughly dated · {pinfo['n_none']} not visible · "
         + (f"timing from local weather ({'calibrated on my field' if pinfo['calibrated'] else 'crop default'})" if pinfo["method"] == "degree-days"
            else f"weather unavailable → {'my field' if pinfo['calibrated'] else 'typical'} timing in days"))
    done(f"{len(allf) - 1} {crop} fields analysed in {(dt.datetime.now() - t0).total_seconds():.0f} s")
    reset_results()
    ss.result = {"allf": allf, "table": table, "info": info, "gts": gts, "crop": crop, "planting": planting, "window": window, "pinfo": pinfo,
                 "radius": radius, "lag_days": lag_days, "n_crop_total": n_crop_total, "w_end": planting + dt.timedelta(days=DAYS_AFTER),
                 "diag": diag, "center": (c.y, c.x), "n_capped": n_capped}
    st.rerun()

if not has_result:
    footer()
    st.stop()

# ============================================================================ results: live filtering
R = ss.result; T = R["table"].copy(); allf = R["allf"]; info = R["info"]; gts = R["gts"]; diag = R.get("diag", {})
if "map_geom" not in R:          # light copy of the boundaries for the maps (~2 m simplify, 5 decimals): far less to send to the browser
    import shapely
    _mg = allf[["field_id", "geometry", "acres"]].copy()
    _mg["geometry"] = shapely.set_precision(_mg.geometry.simplify(0.00002, preserve_topology=True).values, 1e-5)
    R["map_geom"] = gpd.GeoDataFrame(_mg, geometry="geometry", crs="EPSG:4326")
GEO = R["map_geom"]
lat0, lon0 = R.get("center", (lat0, lon0))
is_me = T["field_id"].eq("MY_FIELD")
for c_ in ("elev_median", "texture", "texture_group", "irr_source", "om", "drainage", "slope_median", "elev_range", "aws_0_30cm", "sand", "clay"):
    if c_ not in T:
        T[c_] = np.nan
me = T[is_me].iloc[0]
PINFO = R.get("pinfo", {})
if "plant_conf" not in T:            # results made before planting-date estimation existed
    T["plant_conf"] = np.where(T["greenup_status"].eq("crossing"), "high", None); T["plant_why"] = ""; T["plant_unc"] = np.nan
    T["plant_lo"] = T["plant_hi"] = T["est_planting"]
ep_ = pd.to_datetime(T["est_planting"], errors="coerce")
T["delta_days"] = (ep_ - pd.Timestamp(R["planting"])).dt.days          # estimated planting − my planting date
has_date = ep_.notna() & T["plant_conf"].notna()
trusted = has_date & (T["plant_conf"].eq("high") | (plant_mode == "All estimated"))
# matched in heat-unit days when local weather is available (same crop development, not just same calendar gap), else calendar days
T["delta_match"] = T["delta_tdays"].where(T["delta_tdays"].notna(), T["delta_days"]) if "delta_tdays" in T else T["delta_days"]
HEAT_MATCH = "delta_tdays" in T and T["delta_tdays"].notna().any()
T["similar_planting"] = (trusted & T["delta_match"].abs().le(window)) | is_me
final, STAGES, prof_reason = auto_match(T, me, T["similar_planting"], elevation=t_elev, soil=t_soil, irrigation=t_irr)
T["in_window"] = final
# reference = the q-quantile of ALL comparable fields (never mine) — the same set every field is ranked in, so with q = 0.98
# only the top ~2 % of comparable fields can score above 1
T["ref_ok"] = final
status_ = T["greenup_status"].fillna("too_few_obs")
T["category"] = np.select([final, prof_reason.ne(""), has_date & ~trusted, has_date, status_.isin(["not_emerged", "still_emerging"])],
                          ["comparable", prof_reason, "uncertain", "planting", "later"], "unseen")
T.loc[is_me, "category"] = "me"
MATCH = match_flags(T, me, T["similar_planting"])          # each criterion on its own, for the similarity map
_cen = allf.set_index("field_id").geometry.to_crs(5070).centroid
_dx = _cen.x - _cen["MY_FIELD"].x; _dy = _cen.y - _cen["MY_FIELD"].y
_dist = np.hypot(_dx, _dy) / 1000.0
_brg = (np.degrees(np.arctan2(_dx, _dy)) + 360) % 360
_dirs = np.array(["N", "NE", "E", "SE", "S", "SW", "W", "NW"])
_order = _dist.drop("MY_FIELD").sort_values().index
SHORT = {fid: f"F{i + 1:02d}" for i, fid in enumerate(_order)}; SHORT["MY_FIELD"] = "ME"
WHERE = {fid: ("my field" if fid == "MY_FIELD" else f"{_dist[fid]:.1f} km {_dirs[int(((_brg[fid] + 22.5) // 45) % 8)]}") for fid in _dist.index}
T["short"] = T["field_id"].map(SHORT); T["where"] = T["field_id"].map(WHERE)
nb = T[~is_me]
n_all = len(nb)
n_plant = int(nb["similar_planting"].sum())
n_final = int(nb["in_window"].sum())
anchor = pd.Timestamp(info["anchor_greenup"])
lag = (anchor.date() - R["planting"]).days
n_hi_all = int((nb["plant_conf"] == "high").sum()); n_lo_all = int((nb["plant_conf"] == "low").sum())

observed = bool(info.get("anchor_observed", "not visible" not in str(info.get("anchor_source", ""))))
DEFAULT_RANK_DATE = min(dt.date.today(), R["planting"] + dt.timedelta(days=75))


IMG_COLS = ["field_id", "date", "scene_id", "n_pix"] + [c for k in C.VI_ENGINE for c in (k, f"{k}_p10", f"{k}_p90")]


def full_cols(df):
    """Every image table gets the same columns (missing ones as empty) + the derived indices, so ranking on any index,
    any date, after any number of changes can never fail on a missing column."""
    if df is None:
        return None
    df = df.copy()
    for c_ in IMG_COLS:
        if c_ not in df:
            df[c_] = pd.Series(dtype="datetime64[ns]") if c_ == "date" else (pd.Series(dtype=object) if c_ in ("field_id", "scene_id") else np.nan)
    for c_ in IMG_COLS[3:]:
        df[c_] = pd.to_numeric(df[c_], errors="coerce")
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    return G.add_derived(df) if len(df) else df.assign(**{v_: np.nan for v_ in G.ALL_VI if v_ not in df})


# ============================================================================ rank dialog (pop-up)
def _rank_form():
    st.markdown("Choose the date you want to check. FieldRank uses the newest **clear** Sentinel-2 image of your field on or before that date — "
                "the rank, the scout answer, the map and the ranking all follow it.")
    d = st.date_input("Check date", value=pd.Timestamp(ss["check_date"]).date() if ss.get("check_date") else min(DEFAULT_RANK_DATE, dt.date.today()),
                      key="dlg_date", max_value=dt.date.today(), help="The newest clear image on or before this date is used.")
    dap_ = (d - R["planting"]).days
    st_d = G.stage_at(R["crop"], dap_)
    cur_ = ss.get("vi_seg") or G.BEST
    vi_ = G.resolve_vi(cur_, R["crop"], dap_)
    st.caption(f"Day {dap_} after planting" + (f" · {st_d.label}" if st_d else "") + f" · index **{'★ ' + vi_ if cur_ == G.BEST else vi_}**"
               + (f" (★ Best for {R['crop']} at this stage)" if cur_ == G.BEST else "") + " — switch the index in *Map & ranking* any time, it re-ranks instantly.")
    if st.button("Rank & check my field", type="primary", use_container_width=True, key="dlg_go"):
        if d >= dt.date.today():
            ss.pop("check_date", None)
        else:
            ss.check_date = str(d)
        ss.now_request = True; ss.rank_on = True
        st.rerun()


if hasattr(st, "dialog"):
    rank_dialog = st.dialog("Rank my field")(_rank_form)
elif hasattr(st, "experimental_dialog"):
    rank_dialog = st.experimental_dialog("Rank my field")(_rank_form)
else:
    rank_dialog = None

# ---------------------------------------------------------------- season pop-up (one read feeds both season tabs)
def _season_form():
    st.markdown("Pick the part of the season to analyse. FieldRank reads every **clear** Sentinel-2 look of your field and of the "
                "similar-planting fields in this range (once — later views are instant).")
    prev = ss.get("season_key")
    d0 = pd.Timestamp(prev[0]).date() if prev else max(R["planting"], (anchor + pd.Timedelta(days=7)).date())
    d1 = pd.Timestamp(prev[1]).date() if prev else min(dt.date.today(), R["planting"] + dt.timedelta(days=150),
                                                         pd.Timestamp(ss.get("check_date") or dt.date.today()).date())
    rng = st.date_input("Date range", value=(d0, d1), key="dlg_season")
    n_f = int(T["in_window"].sum())
    st.caption(f"{n_f} fields: mine + the {n_f - 1} comparable fields under the current filters. "
               "Tip: start ~1 week after green-up so bare-soil dates don't dominate.")
    if st.button("Load season", type="primary", use_container_width=True, key="dlg_season_go"):
        if isinstance(rng, (list, tuple)) and len(rng) == 2 and rng[1] > rng[0]:
            ss.season_request = (str(rng[0]), str(rng[1]))
            st.rerun()
        else:
            st.warning("Pick a start and an end date.")


season_dialog = (st.dialog("Season analysis")(_season_form) if hasattr(st, "dialog")
                 else st.experimental_dialog("Season analysis")(_season_form) if hasattr(st, "experimental_dialog") else None)

need_ids = set(T.loc[T["in_window"] | is_me, "field_id"])          # = current comparable set + my field
if ss.get("season_request"):
    rkey = tuple(ss.pop("season_request"))
    have = ss.season_cache.get(rkey)
    have_ids = set(have["field_id"].unique()) if have is not None and len(have) else set()
    todo = need_ids - have_ids if have is not None else need_ids
    if todo:
        t0 = dt.datetime.now()
        sub = allf[allf["field_id"].isin(todo)].reset_index(drop=True)
        status, step, prog, done = status_panel(f"Reading the season {rkey[0]} → {rkey[1]} for {len(sub)} field{'s' if len(sub) != 1 else ''}…")
        try:
            S_, n_found, n_used = run_heavy(prog, season_series, sub, pd.Timestamp(rkey[0]).date(), pd.Timestamp(rkey[1]).date(), progress=prog, log=lambda s_: None)
        except Exception as e:
            outage(status, "read the season's images", e)
            S_, n_found, n_used = pd.DataFrame(columns=IMG_COLS), -1, 0
        merged = pd.concat([have, S_], ignore_index=True) if have is not None and len(have) else S_
        ss.season_cache[rkey] = merged
        for k_ in list(ss.season_cache)[:-2]:           # keep the last 2 season ranges per visitor
            ss.season_cache.pop(k_, None)
        if n_found >= 0:
            ss.setdefault("season_fetched", {})[rkey] = (have_ids | set(sub["field_id"]))
        if n_found >= 0:
            done(f"Season ready · {n_used} clear dates of {n_found} scenes · {len(sub)} field{'s' if len(sub) != 1 else ''} read · {(dt.datetime.now() - t0).total_seconds():.0f} s")
    ss.season_key = rkey; ss.pop("season_bars", None)
S_season = full_cols(ss.season_cache.get(ss.get("season_key"))) if ss.get("season_key") else None
season_missing = set()
if S_season is not None:
    fetched = ss.get("season_fetched", {}).get(ss.season_key, set(S_season["field_id"].unique()))
    season_missing = need_ids - fetched                               # filters were loosened after loading
    S_season = full_cols(S_season[S_season["field_id"].isin(need_ids)])   # only today's comparable set is used

# ============================================================================ ONE check date drives everything
# rank card, "should I scout?", map, ranking bars and charts all use the same image and the same numbers
if "vi_pending" in ss:
    ss["vi_seg"] = ss.pop("vi_pending")
PLANTING = {fid: d for fid, d in zip(T["field_id"], pd.to_datetime(T["est_planting"], errors="coerce")) if pd.notna(d)}
CHECK_DATE = pd.Timestamp(ss.get("check_date") or dt.date.today()).date()
NOW_ASOF = CHECK_DATE
my_plant = R["planting"]
st_today = G.stage_at(R["crop"], (CHECK_DATE - my_plant).days)
in_season = st_today is not None and st_today.key not in ("pre", "post")
now_key = (str(CHECK_DATE), R["crop"], int(max_days))
now_have = ss.now_cache.get(now_key)
now_missing = need_ids - (now_have[1] if now_have else set())
# nothing is read until the user ranks; once a rank exists for this date, loosened filters top it up automatically
want_now = bool(ss.pop("now_request", False)) or (now_have is not None and bool(now_missing) and not ss.get("now_failed", {}).get(now_key))
if want_now and now_missing and st_today.key != "pre":
    t0 = dt.datetime.now()
    sub = allf[allf["field_id"].isin(now_missing)].reset_index(drop=True)
    rngs = [(CHECK_DATE - dt.timedelta(days=int(max_days)), CHECK_DATE)]          # only the look-back window
    status, step, prog, done = status_panel(f"Ranking my field · clear images {rngs[0][0]:%b %d} → {CHECK_DATE:%b %d, %Y} · {len(sub)} field{'s' if len(sub) != 1 else ''}…")
    parts, n_used = [], 0
    try:
        for a_, b_ in rngs:
            S_, _, u_ = run_heavy(prog, season_series, sub, a_, b_, progress=prog, log=lambda s_: None)
            parts.append(S_); n_used += u_
        S_new = pd.concat([p_ for p_ in parts if len(p_)], ignore_index=True) if any(len(p_) for p_ in parts) else pd.DataFrame()
        old = now_have[0] if now_have else None
        S_all = pd.concat([x_ for x_ in (old, S_new) if x_ is not None and len(x_)], ignore_index=True) if (old is not None and len(old)) or len(S_new) else pd.DataFrame()
        if len(S_all):
            S_all = S_all.sort_values("n_pix", ascending=False).drop_duplicates(["field_id", "date"]).sort_values(["field_id", "date"])
        ss.now_cache[now_key] = (S_all, (now_have[1] if now_have else set()) | set(sub["field_id"]))
        for k_ in list(ss.now_cache)[:-3]:              # each visitor keeps only their last 3 check dates in memory
            ss.now_cache.pop(k_, None)
        done(f"Ready · {n_used} clear date{'s' if n_used != 1 else ''} · {' + '.join(f'{a_:%b %d}–{b_:%b %d}' for a_, b_ in rngs)} · "
             f"{(dt.datetime.now() - t0).total_seconds():.0f} s")
    except Exception as e:  # network down etc. — never break the page, and don't retry on every rerun
        ss.setdefault("now_failed", {})[now_key] = True
        outage(status, "read the images for this date", e)
    ss.pop("rank_bars", None)
    now_have = ss.now_cache.get(now_key)

comp_now = T.loc[T["in_window"] & ~is_me, "field_id"]
ref_now = T.loc[T["ref_ok"] & ~is_me, "field_id"]
min_n = 5 if len(comp_now) >= 10 else 3
S_now = None
if now_have is not None and len(now_have[0]):
    S_now = full_cols(now_have[0])
    S_now = S_now[S_now["field_id"].isin(need_ids)]
IMG_DATE, IMG_N = G.pick_image_date(S_now, comp_now, "MY_FIELD", CHECK_DATE, int(max_days), min_n) if S_now is not None else (None, 0)
VI_CHOICE = ss.get("vi_seg") or G.BEST
RANK_DATE = IMG_DATE or CHECK_DATE
RANK_STAGE = G.stage_at(R["crop"], (RANK_DATE - R["planting"]).days)
if VI_CHOICE not in [G.BEST, G.CONSENSUS] + list(G.ALL_VI):      # a value left over from an older version of the app
    VI_CHOICE = G.BEST; ss["vi_seg"] = G.BEST
VI = G.resolve_vi(VI_CHOICE, R["crop"], (RANK_DATE - R["planting"]).days)
VI_TAG = f"★ {VI}" if VI_CHOICE in (G.BEST, G.CONSENSUS) else VI
USE_CONS = VI_CHOICE == G.CONSENSUS


def best_note(choice, vi, when_txt):
    """One line under an index selector: which index ★ Best picked and why (crop-specific research)."""
    crop_ = R["crop"]
    if choice == G.BEST:
        return (f'<div class="note" style="margin:.1rem 0 .5rem">★ Best for {crop_} {when_txt}: <b>{vi}</b> — {G.VI_HELP.get(vi, "")}. '
                f'<br><span style="color:#64748b">Why: {G.CROP_EVIDENCE.get(crop_, "")}</span></div>')
    return (f'<div class="note" style="margin:.1rem 0 .5rem"><b>{vi}</b> — {G.VI_HELP.get(vi, "")}. '
            f'Choose ★ Best for the index research ties most closely to {crop_} yield.</div>')


# ranking — EVERY comparable field, each on its clear image nearest the check image, compared through SI
# (value ÷ that day's reference), so cloud-free days of different fields can be compared fairly.
# Re-scored on every toggle / index change (no download).
RANK_WINDOW = min(8, int(max_days))
ranked = ref = mr = None
IMG_IDS = []
N_COMP = int(len(comp_now))                     # comparable fields, never counting mine
N_RANKED = N_NOIMG = 0
RT = None
if IMG_DATE is not None:
    S_rank = S_now
    try:
        RT = rank_on_date(S_rank, comp_now, ref_now, "MY_FIELD", VI, IMG_DATE,
                          min(CHECK_DATE, IMG_DATE + dt.timedelta(days=RANK_WINDOW)), q, window=RANK_WINDOW)
        if RT is None or not (RT["field_id"] == "MY_FIELD").any():
            raise ValueError("my field has no value on this image")
    except Exception as e:
        st.warning(f"Could not rank on the {IMG_DATE:%b %d} image with {VI} ({type(e).__name__}). Try another date or index — "
                   "nothing else is affected.", icon="⚠️")
        IMG_DATE, IMG_N, RT = None, 0, None
if IMG_DATE is not None:
    IMG_IDS = sorted(S_now.loc[S_now["date"] == pd.Timestamp(IMG_DATE), "scene_id"].astype(str).unique())
    ranked = T.drop(columns=[c for c in ("vi_median", "vi_p10", "vi_p90", "n_pix", "obs_date", "scene_id", "SI", "rank") if c in T]).merge(RT, on="field_id", how="left")
    ranked["in_cohort"] = ranked["in_window"].fillna(False).astype(bool)
    ranked["is_ref_field"] = ranked["in_ref"].fillna(False).astype(bool) & (ranked["SI"] >= 1 - 1e-9) & (ranked["field_id"] != "MY_FIELD")
    ranked["vi_ref"] = ranked["ref_day"]
    ranked = ranked.sort_values(["in_cohort", "SI"], ascending=[False, False])
    mr = ranked[ranked["field_id"] == "MY_FIELD"].iloc[0]
    ref = float(mr["ref_day"]) if pd.notna(mr["ref_day"]) else float("nan")
    N_RANKED = int(mr["n_ranked"]) if pd.notna(mr["n_ranked"]) else 0
    N_NOIMG = max(N_COMP - max(N_RANKED - 1, 0), 0)

# ◎ consensus: the same ranking on every index (same image, same calibration, same reference rule), combined by
# stage-weighted percentile — a robustness check on the single ★ index. Memoised: re-ranking 8 indices is cheap but not free.
CONS = MYC = AGREE = None
W_NOW = G.vi_weights(R["crop"], (RANK_DATE - R["planting"]).days)
if IMG_DATE is not None:
    ckey = (str(IMG_DATE), str(CHECK_DATE), q, RANK_WINDOW, hash(tuple(sorted(comp_now))), hash(tuple(sorted(ref_now))), id(now_have[0]))
    if ss.get("cons_key") != ckey:
        per_vi = {}
        for v_ in G.ALL_VI:
            try:
                per_vi[v_] = rank_on_date(S_rank, comp_now, ref_now, "MY_FIELD", v_, IMG_DATE,
                                          min(CHECK_DATE, IMG_DATE + dt.timedelta(days=RANK_WINDOW)), q, window=RANK_WINDOW)
            except Exception:
                pass
        ss.cons_key, ss.cons_val = ckey, per_vi
    CONS = G.consensus(ss.cons_val, W_NOW)
    if len(CONS) and (CONS["field_id"] == "MY_FIELD").any():
        MYC = CONS[CONS["field_id"] == "MY_FIELD"].iloc[0]
        AGREE = G.agreement(MYC, W_NOW) if pd.notna(MYC["cons_score"]) else None
    ranked = ranked.merge(CONS[["field_id", "cons_score", "cons_rank", "cons_pct", "n_vi"] + [c for c in CONS if c.startswith("rank_")]], on="field_id", how="left")
N_CONS = int(CONS["cons_score"].notna().sum()) if CONS is not None and len(CONS) else 0
NS = None
if S_now is not None:
    TSR, SI_ALL = (rank_series(S_rank, comp_now, ref_now, "MY_FIELD", VI, IMG_DATE, q, window=RANK_WINDOW, all_fields=True)
                   if IMG_DATE is not None else (None, None))                       # same method on every date
    NS = G.now_status(S_now, comp_now, ref_now, "MY_FIELD", R["crop"], my_plant, CHECK_DATE, q,
                      min_n=max(1, min(min_n, IMG_N)) if IMG_DATE else min_n, vi=VI, planting=PLANTING, at_date=IMG_DATE,
                      ts_override=TSR)
elif now_have is not None:
    NS = {"ok": False, "why": "no clear images in the look-back window", "as_of": CHECK_DATE, "stage_today": st_today}
if NS and NS.get("ok") and ranked is not None and pd.notna(mr["rank"]):
    cr_ = ranked[ranked["in_cohort"] & ranked["SI"].notna()]
    NS.update(SI=float(mr["SI"]), rank=int(mr["rank"]), n=N_RANKED, me=float(mr["vi_median"]), ref=float(mr["ref_day"]),
              day=pd.DataFrame({"field_id": cr_["field_id"], "value": cr_["vi_median"], "pct": cr_["SI"] * 100, "is_me": cr_["field_id"].eq("MY_FIELD")}),
              window=NS.get("window"), n_noimg=N_NOIMG, rank_window=RANK_WINDOW)
    NS["word"], NS["tone"] = G.verdict(NS["SI"])
    # ★ yield-critical stage so far — mean of each field's calibrated SI over my images inside the stage (same method)
    cw0 = G.critical(R["crop"])
    w0_, w1_ = my_plant + dt.timedelta(days=cw0.start), my_plant + dt.timedelta(days=cw0.end - 1)
    NS["window"] = None
    if SI_ALL is not None and len(SI_ALL):
        Wd = SI_ALL[(SI_ALL["date"] >= pd.Timestamp(w0_)) & (SI_ALL["date"] <= pd.Timestamp(w1_))]
        if len(Wd) and (Wd["field_id"] == "MY_FIELD").any():
            mean_ = Wd.groupby("field_id")["SI"].mean()
            rk_ = mean_.rank(ascending=False, method="first")
            NS["window"] = {"SI": float(mean_["MY_FIELD"]), "rank": int(rk_["MY_FIELD"]), "n": int(len(mean_)),
                            "looks": int(Wd.loc[Wd["field_id"] == "MY_FIELD", "date"].nunique()), "vi": VI, "label": cw0.label,
                            "start": w0_, "end": w1_}


def _check_now():
    ss.rank_on = True; ss.now_request = True


def open_rank():
    """The ONE way to rank or change the date: the pop-up."""
    if rank_dialog:
        rank_dialog()
    else:
        _check_now(); st.rerun()


def why_rows(items) -> str:
    """Numbered/iconed rows — one look for every reasons list in the app."""
    return "".join(f'<div style="display:flex;gap:10px;margin:0 0 .55rem;font-size:.88rem;line-height:1.4;color:#334155">'
                   f'<span style="flex:none;width:22px;height:22px;border-radius:50%;background:#eef2f6;display:grid;place-items:center;'
                   f'font-size:.72rem;font-weight:800;color:#0f172a">{mark}</span><span>{txt}</span></div>' for mark, txt in items)


def rank_status_html():
    """One line under my rank: which date and image everything below uses, or why there is no rank."""
    if IMG_DATE is not None and (mr is None or pd.isna(mr["rank"])):
        return (f'<div class="rankst warn">My image {IMG_DATE:%b %d} is clear, but fewer than 3 comparable fields were — no reference to rank against. '
                f'Widen <b>Planting within ± days</b>, use <b>All estimated</b> planting dates, or switch a filter off.</div>')
    if IMG_DATE is not None:
        gap_ = (CHECK_DATE - IMG_DATE).days
        when_ = "same day" if not gap_ else f"{gap_} d before"
        return f'<div class="rankst">📅 <b>{CHECK_DATE:%b %d, %Y}</b> → my image <b>{IMG_DATE:%b %d}</b></div>'
    if now_have is not None:
        return (f'<div class="rankst warn">No clear image of my field {CHECK_DATE - dt.timedelta(days=int(max_days)):%b %d} → '
                f'{CHECK_DATE:%b %d, %Y} (clouds). Pick a later date, or a longer look-back under Advanced.</div>')
    if st_today is not None and st_today.key == "pre":
        return f'<div class="rankst warn">Not planted yet on {CHECK_DATE:%b %d, %Y} — pick a date after {my_plant:%b %d}.</div>'
    return ('<div class="rankst">Comparable fields are ready. Pick a date — rank, scout answer, map and ranking all follow it, '
            'and images are read only for that date.</div>')


# ============================================================================ KPI row (always visible)
with st.container(key="kpirow"):
    k1, k2, k3, k4 = st.columns([1.2, 1.05, 1.2, 1.0])
with k1, card():
    active = ["planting"] + [s_[0] for s_ in STAGES]
    html(f'<div class="kpi hero"><div class="lab">Comparable fields <span style="opacity:.75">(not counting mine)</span></div><div class="val">{n_final} <small>of {n_all} {R["crop"]} fields</small></div>'
         f'<div class="bar"><i style="width:{max(n_final / max(n_all, 1) * 100, 2):.0f}%"></i></div>'
         f'<div class="sub">matched on {" · ".join(active)} · within {R["radius"]} km</div></div>')
    if n_final < 8 and n_all >= 10:
        # few left: say which step removed the most, and what to loosen
        _nd = int((nb["plant_conf"].eq("high") | (nb["plant_conf"].eq("low") & (plant_mode == "All estimated"))).sum())
        _steps = [("planting date found", n_all - _nd, "choose “+ roughly dated” in the sidebar" if plant_mode == "High confidence" else "pick a later check date or a wider radius"),
                  (f"planted within ± {window} d", _nd - n_plant, "widen “Planting within ± days”, or check the planting date you entered")]
        _prev = n_plant
        for _, lab_, n_, _r in STAGES:
            _steps.append((lab_.lower(), _prev - n_, f"switch off “{lab_}”")); _prev = n_
        big_ = max(_steps, key=lambda x: x[1])
        if big_[1] > 0:
            html(f'<div class="rankst warn">Most fields were dropped at <b>{big_[0]}</b> (−{big_[1]}). To compare with more: {big_[2]}.</div>')
    with popover("How 'similar' is decided"):
        st.markdown(f"- **Planting** — estimated planting date within ± {window} {'heat-unit ' if HEAT_MATCH else ''}days of the date I entered (always on) · "
                    f"{'well-dated fields only' if plant_mode == 'High confidence' else 'well and roughly dated fields'}\n"
                    "- **Elevation** — field median within half the local inter-quartile range (3–10 m)\n"
                    "- **Soil** — same USDA texture group, 0–30 cm\n"
                    "- **Irrigation** — same irrigated / dryland status\n\n"
                    + "  \n".join(f"**{lab}:** {n} fields · {rule}" for _, lab, n, rule in STAGES))
with k2, card():
    pc_ = PINFO.get("my_conf")
    if observed:
        sat_ = PINFO.get("my_sat_planting")
        html(f'<div class="kpi"><div class="lab">My green-up</div><div class="val">{anchor.strftime("%b %d")}</div>'
             f'<div class="sub">{lag:+d} d after planting'
             + {"confirmed": ' · <b style="color:#15803d">confirms my date ✓</b>', "partly": ' · partly confirms my date'}.get(PINFO.get("cal_state"), "")
             + (f'<br>satellite: planted ≈ <b>{sat_:%b %d}</b> ± {PINFO.get("my_sat_unc", 0):.0f} d' if sat_ else "") + '</div>'
             f'<div class="sub" style="margin-top:.3rem"><span class="pill {"g" if pc_ == "high" else "a"}">{"Well dated" if pc_ == "high" else "Roughly dated"}</span>'
             + (f' &nbsp;NDVI {pd.Timestamp(T.loc[is_me, "onset_ndvi"].iloc[0]):%b %d} · EVI2 {pd.Timestamp(T.loc[is_me, "onset_evi2"].iloc[0]):%b %d}'
                if {"onset_ndvi", "onset_evi2"} <= set(T) and T.loc[is_me, ["onset_ndvi", "onset_evi2"]].notna().all(axis=1).iloc[0] else "")
             + '</div></div>')
    else:
        html(f'<div class="kpi"><div class="lab">My green-up</div><div class="val">not seen</div>'
             f'<div class="sub"><span class="pill a">assumed {anchor.strftime("%b %d")}</span> &nbsp;planting + {NOMINAL_EMERGENCE_DAYS} d</div></div>')
    with popover("Planting dates"):
        sc = info.get("status_counts", {})
        _cs = {"confirmed": "confirmed by my field", "partly": f"{PINFO.get('cal_weight', 0) * 100:.0f}% my field + crop default"}.get(PINFO.get("cal_state"), "crop default")
        meth_ = (f"local weather (degree-days), {_cs} — "
                 f"{PINFO.get('target_gdd') or 0:.0f} °C-days from planting to visible green-up" if PINFO.get("method") == "degree-days"
                 else f"{'my field' if PINFO.get('calibrated') else 'typical'} timing in days (weather not available)")
        st.markdown(
            f"**Green-up** = the day the crop starts to show: a smooth growth curve is fitted through each field's clear images, once with NDVI "
            f"and once with EVI2 (less fooled by wet or bright soil), and green-up is where the curve has made 20 % of its rise. "
            f"Mine: **{anchor.date()}** ({info['anchor_source']}).  \n"
            f"**Planting** is earlier: seed → emergence → a canopy big enough to see at 10 m. How long that takes depends on heat, so "
            f"each field's planting date is worked back from its green-up with **{meth_}**.  \n"
            f"**Well dated** = the curve fits, NDVI and EVI2 agree within {PD.AGREE} days, and clear images ≤ {PD.HIGH_MAX_GAP} days apart bracket "
            f"the green-up → planting good to about a week. **Roughly dated** = green-up seen but blurred (cloudy weeks, too few looks for a "
            f"curve, the two indices disagree, or a weak rise) → could be off by 1–2 weeks.  \n"
            f"**Neighbours:** {n_hi_all} well dated · {n_lo_all} roughly dated · {sc.get('not_emerged', 0) + sc.get('still_emerging', 0)} planted later · "
            f"{sc.get('already_green', 0)} green before planting · {sc.get('too_few_obs', 0)} too cloudy  \n"
            f"**Imagery:** {info['n_scenes']} clear Sentinel-2 dates" + (f" + {info['n_landsat']} Landsat ({info['tier2_reason']})" if info.get("n_landsat") else "")
            + ("  \n**Landsat lined up with Sentinel-2:** NASA's bandpass adjustment, then " + " · ".join(
                f"{k.upper()} {v['pairs']} same-field pairs within 2 days, gap {v['gap_before']:+.3f} → {v['gap_after']:+.3f}"
                if v.get("pairs") else f"{k.upper()}: too few same-day pairs, bandpass adjustment only" for k, v in info.get("landsat_cal", {}).items())
               if info.get("landsat_cal") else "")
            + "  \n**Ranking and trends use Sentinel-2 only** (10 m, one sensor family) — Landsat only helps find green-up in cloudy weeks.")
with k3, card():
    if ranked is not None and pd.notna(mr["rank"]):
        sic = si_class(mr["SI"])
        cons_ok = MYC is not None and pd.notna(MYC["cons_score"])
        si_pill = f'<span class="pill {"g" if sic in ("≥ 0.95", "0.85–0.95") else "a" if sic == "0.70–0.85" else "r"}">SI {mr["SI"]:.2f}</span>'
        if USE_CONS and cons_ok:
            head_ = (f'<div class="lab">My rank · ◎ Consensus of {int(MYC["n_vi"])}</div>'
                     f'<div class="val">#{int(MYC["cons_rank"])} <small>of {N_CONS}</small></div>'
                     f'<div class="sub"><span class="pill {"g" if MYC["cons_pct"] >= 50 else "a" if MYC["cons_pct"] >= 25 else "r"}">score {MYC["cons_score"]:.0f}</span>'
                     f' &nbsp;★ {VI}: #{int(mr["rank"])} · SI {mr["SI"]:.2f}</div>')
        else:
            head_ = (f'<div class="lab">My rank · {VI_TAG}</div>'
                     f'<div class="val">#{int(mr["rank"])} <small>of {int(mr["n_ranked"])}</small></div>'
                     f'<div class="sub">{si_pill} &nbsp;better than {max(mr["percentile"], 0):.0f}%</div>'
                     + (f'<div class="sub" style="margin-top:.2rem">◎ consensus #{int(MYC["cons_rank"])} of {N_CONS}</div>' if cons_ok else ""))
        html(f'<div class="kpi">{head_}<div class="sub" style="margin-top:.2rem">me + {N_RANKED - 1} of {N_COMP} comparable'
             + (f' · {N_NOIMG} cloudy' if N_NOIMG else '') + '</div>' + rank_status_html() + '</div>')
        b1, b2 = st.columns(2, gap="small")
        if b1.button("📅 Date", use_container_width=True, key="rank_open", help="Change the check date — rank, scout answer, map and ranking all follow"):
            open_rank()
        with b2:
            with popover("Details"):
                st.markdown(f"**My {VI}:** {mr['vi_median']:.3f} (image {mr['obs_date']})  \n**Reference (q{int(q * 100)}):** {ref:.3f}  \n"
                            f"**Each field's value:** the **median** of its clear 10 m pixels at least {C.INWARD_BUFFER_M:.0f} m inside the boundary "
                            f"(edge trees, roads, ditches and mixed border pixels are left out; clouds and shadows masked; ≥ {C.MIN_PIXELS_10M} clear pixels needed)  \n"
                            + (f"**Stage:** {RANK_STAGE.label}  \n" if RANK_STAGE else "") +
                            f"**Sufficiency index:** {mr['SI']:.2f} = my value ÷ reference  \n"
                            f"**Compared:** my field ({IMG_DATE:%b %d}) + {N_RANKED - 1} of the {N_COMP} comparable fields, each on its clear image "
                            f"within ±{RANK_WINDOW} days" + (f" — {N_NOIMG} had no clear image (clouds)" if N_NOIMG else "") + "  \n"
                            f"**How fields seen on different days are compared:** SI = a field's value ÷ the reference of *that same day*, so light, "
                            f"atmosphere and a few days of growth cancel out  \n**Reference:** the q{int(q * 100)} of the other fields clear that day (never includes mine)  \n"
                            f"**Index:** change it in *Map & ranking* — re-ranks instantly, nothing is downloaded")
                if CHECK_DATE >= dt.date.today() - dt.timedelta(days=3) and st.button("↻ Look for newer images", key="chk_update", use_container_width=True):
                    refresh_recent(); ss.now_cache.pop(now_key, None); ss.get("now_failed", {}).pop(now_key, None); ss.now_request = True; st.rerun()
    else:
        html('<div class="kpi"><div class="lab">My rank</div><div class="val">Not ranked</div>' + rank_status_html() + '</div>')
        if st.button("Rank my field", type="primary", use_container_width=True, key="rank_open"):
            open_rank()
        if not rank_dialog:  # very old Streamlit: inline form
            with popover("Pick the date"):
                _rank_form()
with k4, card():
    tex = me.get("texture") if isinstance(me.get("texture"), str) else "—"
    irr_txt = "Irrigated" if bool(me.get("irrigated")) else "Dryland"
    html(f'<div class="kpi"><div class="lab">My field</div><div class="val">{fmt(me.get("acres"))} <small>ac</small></div>'
         f'<div class="pills" style="margin-top:.3rem"><span class="pill b">⛰ {fmt(me.get("elev_median"), ".0f", " m")}</span>'
         f'<span class="pill a">{tex}</span><span class="pill {"b" if irr_txt == "Irrigated" else "soft"}">💧 {irr_txt}</span></div></div>')
    with popover("Field details"):
        st.markdown(f"**Elevation:** median {fmt(me.get('elev_median'), '.1f', ' m')} · range {fmt(me.get('elev_range'), '.1f', ' m')} · slope {fmt(me.get('slope_median'), '.1f', ' %')}  \n"
                    f"**Soil 0–30 cm:** {tex} ({me.get('texture_group') if isinstance(me.get('texture_group'), str) else '—'}) · sand {fmt(me.get('sand'), '.0f', ' %')} · "
                    f"clay {fmt(me.get('clay'), '.0f', ' %')} · OM {fmt(me.get('om'), '.2f', ' %')} · water {fmt(me.get('aws_0_30cm'), '.1f', ' cm')} · "
                    f"{me.get('drainage') if isinstance(me.get('drainage'), str) else '—'}  \n"
                    f"**Irrigation:** {me.get('irr_source') if isinstance(me.get('irr_source'), str) else '—'}  \n**Boundary:** {ss.geom_src or '—'}")
        st.caption(f"Sources — elevation: {diag.get('elev_src') or 'not available'} · soil: {diag.get('soil_src') or 'not available'} · "
                   f"irrigation: {'LANID 2015–17 + pivot shape' if diag.get('n_lanid') else 'pivot shape'}")
        for e in diag.get("errors", []):
            st.caption(f"⚠️ {e}")

# callouts only when something needs attention
miss = [(lab, n) for lab, n, on in (("elevation", diag.get("n_elev", 0), t_elev), ("soil", diag.get("n_soil", 0), t_soil)) if on and n < len(allf) * 0.8]
if miss:
    html('<div class="callout err">' + " · ".join(f"<b>{lab.title()}</b> is available for only {n}/{len(allf)} fields — fields without it are excluded by that filter." for lab, n in miss)
         + (f" ({diag['errors'][0][:220]})" if diag.get("errors") else "")
         + (" <br><b>Fix:</b> run <code>bash scripts/fieldrank.sh soil</code> once (builds the soil table from Planetary Computer, "
            "so the USDA server is no longer needed), then press <b>Find comparable fields</b> again." if any(l_ == "soil" for l_, _ in miss) else "")
         + "</div>")
PLANT_OFF_DAYS = 20          # entered planting vs what the satellite + weather suggest — beyond this we ask the user to check


def _use_sat_planting(d_):
    ss["planting"] = d_; ss["auto_process"] = True
    ss.pop("check_date", None)


_sat = PINFO.get("my_sat_planting")
_off = (R["planting"] - _sat).days if _sat else None
if PINFO.get("warn") or (_off is not None and abs(_off) >= PLANT_OFF_DAYS):
    _lag = PINFO.get("my_lag_days")
    _wxt = "in this season's weather" if PINFO.get("method") == "degree-days" else "with typical timing"
    with st.container(key="plantcheck"):
        cA, cB = st.columns([3.2, 1], vertical_alignment="center")
        with cA:
            html('<div class="pc-t">📅 Is the planting date right?</div><div class="pc-b">'
                 f'You entered <b>{R["planting"]:%b %d, %Y}</b>. The satellite saw your field green up on <b>{anchor:%b %d}</b>'
                 + (f' — {abs(_lag)} days {"after" if _lag >= 0 else "<u>before</u>"} that date' if _lag is not None else "") + '. '
                 + (f'For {R["crop"]} {_wxt}, that points to planting around <b>{_sat:%b %d}</b> '
                    f'(± {PINFO.get("my_sat_unc") or 0:.0f} days) — <b>{abs(_off)} days {"earlier" if _off > 0 else "later"}</b> than entered. ' if _sat else "")
                 + ('Because the gap is outside the usual range, neighbours are dated with typical crop timing, so <b>few of them may match '
                    'your date</b>. ' if PINFO.get("warn") else
                    'Neighbours are still matched fairly — on how far their green-up is from yours — but a typo here also shifts the season '
                    'stage and the ★ index. ')
                 + 'If the date was mistyped, use the satellite date; if you replanted or know it\'s right, keep yours.</div>')
        with cB:
            if _sat:
                st.button(f"Use {_sat:%b %d} and search again", key="use_sat_plant", type="primary", use_container_width=True,
                          on_click=_use_sat_planting, args=(_sat,))
elif not observed:
    html(f'<div class="callout warn">My field\'s green-up could not be seen in the images ({str(me["greenup_status"]).replace("_", " ")}), '
         f'so matching assumes it emerged {NOMINAL_EMERGENCE_DAYS} days after the planting date you entered.</div>')

# ============================================================================ should I scout? (always visible)


def stage_timeline(crop, dap):
    segs = G.stages(crop)
    span = [(s_, (min(s_.end, s_.start + 25) if s_.key == "post" else s_.end) - s_.start) for s_ in segs]
    tot = sum(w_ for _, w_ in span)
    col = {"early": "#d9f99d", "vegetative": "#86efac", "critical": "#16a34a", "late": "#fcd34d", "post": "#e2e8f0"}
    short = {"early": "Early", "vegetative": "Growth", "critical": "★ Yield-critical", "late": "Late", "post": "After"}
    cur = G.stage_at(crop, dap)
    bar = "".join(f'<i title="{s_.label} · day {s_.start}–{s_.end if s_.key != "post" else ""}" style="width:{w_ / tot * 100:.1f}%;background:{col[s_.key]}"></i>' for s_, w_ in span)
    lab = "".join(f'<span class="{"on" if cur and cur.key == s_.key else ""}" style="width:{w_ / tot * 100:.1f}%">{short[s_.key]}</span>' for s_, w_ in span)
    mk = ""
    if dap is not None and dap >= 0:
        x_ = 0.0
        for s_, w_ in span:
            if dap < s_.start + w_ or s_.key == "post":
                x_ += min(max(dap - s_.start, 0), w_); break
            x_ += w_
        mk = f'<div class="tlm" style="left:{min(x_ / tot * 100, 99):.1f}%"><b>day {dap}</b><i></i></div>'
    return f'<div class="tlw">{mk}<div class="tl">{bar}</div><div class="tll">{lab}</div></div>'


TONE = {"scout": ("#fef2f2", "#fecaca", "#991b1b"), "watch": ("#fffbeb", "#fde68a", "#92400e"),
        "ok": ("#f0fdf4", "#bbf7d0", "#166534"), "info": ("#f8fafc", "#e2e8f0", "#334155")}
BAND_DF = pd.DataFrame({"x0": [0, 70, 85, 95], "x1": [70, 85, 95, 130], "c": ["< 0.70", "0.70–0.85", "0.85–0.95", "≥ 0.95"]})


def where_you_stand(day: pd.DataFrame, date_):
    d = day.copy(); d["jit"] = np.random.default_rng(7).uniform(0.15, 0.85, len(d)); d.loc[d["is_me"], "jit"] = 0.5
    d["who"] = np.where(d["is_me"], "my field", "comparable field")
    lo = float(max(0, min(d["pct"].min() - 6, 60))); hi = float(max(106, d["pct"].max() + 4))
    xax = alt.Axis(values=[v_ for v_ in (40, 50, 60, 70, 85, 100) if lo <= v_ <= hi], labelFontSize=10, grid=False, title="% of the best comparable fields")
    xs = alt.X("pct:Q", scale=alt.Scale(domain=[lo, hi], nice=False), axis=xax)
    bg = alt.Chart(BAND_DF).mark_rect(opacity=0.13, clip=True).encode(x=alt.X("x0:Q", scale=alt.Scale(domain=[lo, hi], nice=False), axis=xax), x2="x1:Q",
                                                           color=alt.Color("c:N", scale=alt.Scale(domain=SI_DOMAIN, range=SI_RANGE), legend=None))
    oth = alt.Chart(d[~d["is_me"]]).mark_circle(size=55, color="#94a3b8", opacity=0.75).encode(
        x=xs, y=alt.Y("jit:Q", axis=None, scale=alt.Scale(domain=[0, 1])), tooltip=[alt.Tooltip("pct:Q", format=".0f", title="% of best")])
    me_ = alt.Chart(d[d["is_me"]]).mark_circle(size=320, color=BLUE, stroke="#fff", strokeWidth=2.5, opacity=1).encode(
        x=xs, y=alt.Y("jit:Q", axis=None), tooltip=[alt.Tooltip("pct:Q", format=".0f", title="my field, % of best")])
    lab = alt.Chart(d[d["is_me"]]).mark_text(dy=-19, fontWeight="bold", fontSize=12, color=BLUE).encode(x=xs, y="jit:Q", text=alt.value("You"))
    best = alt.Chart(pd.DataFrame({"pct": [100.0]})).mark_rule(color=REF_COLOR, strokeDash=[4, 3], strokeWidth=2).encode(x=xs)
    bl = alt.Chart(pd.DataFrame({"pct": [100.0], "y": [0.97]})).mark_text(align="left", dx=4, fontSize=10, color=REF_COLOR).encode(
        x=xs, y=alt.Y("y:Q", axis=None), text=alt.value("best fields"))
    st.altair_chart((bg + oth + best + bl + me_ + lab).properties(height=150).configure_view(strokeWidth=0), use_container_width=True)


def few_weeks(ts_: pd.DataFrame):
    t_ = ts_.copy(); t_["pct"] = t_["SI"] * 100; t_["si_class"] = t_["SI"].apply(si_class)
    lo = float(max(0, min(t_["pct"].min() - 8, 62))); hi = float(max(104, t_["pct"].max() + 4))
    yax = alt.Axis(values=[v_ for v_ in (40, 50, 60, 70, 85, 100) if lo <= v_ <= hi], labelFontSize=10, title="% of best")
    ys = alt.Y("pct:Q", scale=alt.Scale(domain=[lo, hi], nice=False, zero=False), axis=yax)
    xs = alt.X("date:T", title=None, axis=alt.Axis(format="%b %d", grid=False, labelFontSize=10, tickCount=5))
    bands = BAND_DF.rename(columns={"x0": "y0", "x1": "y1"})
    bg = alt.Chart(bands).mark_rect(opacity=0.13, clip=True).encode(y=alt.Y("y0:Q", scale=alt.Scale(domain=[lo, hi], nice=False, zero=False), axis=yax), y2="y1:Q",
                                                         color=alt.Color("c:N", scale=alt.Scale(domain=SI_DOMAIN, range=SI_RANGE), legend=None))
    ln_ = alt.Chart(t_).mark_line(color=BLUE, strokeWidth=2.5).encode(x=xs, y=ys)
    pt_ = alt.Chart(t_).mark_circle(size=70, color=BLUE, stroke="#fff", strokeWidth=1.5, opacity=1).encode(
        x=xs, y=ys, tooltip=[alt.Tooltip("date:T", format="%b %d"), alt.Tooltip("pct:Q", format=".0f", title="% of best"),
                             alt.Tooltip("rank:Q", title="rank"), alt.Tooltip("n:Q", title="comparable clear")])
    one_ = alt.Chart(pd.DataFrame({"pct": [100.0]})).mark_rule(color=REF_COLOR, strokeDash=[4, 3]).encode(y=ys)
    st.altair_chart((bg + one_ + ln_ + pt_).properties(height=150).configure_view(strokeWidth=0), use_container_width=True)


AD = G.advise(NS, st_today) if NS is not None else None
if AD is not None and AGREE and AD.get("level") in ("scout", "watch", "ok") and AD.get("conf"):
    # second opinion from all indices: agreement keeps the confidence, disagreement lowers it one step and says why
    down = {"high": "medium", "medium": "low", "low": "low"}
    nb_, n_ = AGREE["n_bottom_half"], AGREE["n"]
    AD["why"] = list(AD["why"]) + [f"◎ Across all {n_} indices (weighted for this stage): <b>{int(MYC['cons_rank'])}{G._ord(int(MYC['cons_rank']))} of {N_CONS}</b> — "
                                  f"{nb_} of {n_} put my field in the bottom half"]
    if AD["level"] in ("scout", "watch") and AGREE["w_top_half"] >= 0.6:
        AD["conf"] = down[AD["conf"]]; AD["conf_why"] = list(AD.get("conf_why", [])) + [f"the indices disagree — most of the weight puts my field in the top half"]
    elif AD["level"] == "ok" and AGREE["w_bottom_quarter"] >= 0.5:
        AD["conf"] = down[AD["conf"]]; AD["conf_why"] = list(AD.get("conf_why", [])) + [f"half of the index weight puts my field in the bottom quarter"]
    elif AD["level"] in ("scout", "watch") and AGREE["w_bottom_half"] >= 0.75 or AD["level"] == "ok" and AGREE["w_top_half"] >= 0.75:
        AD["conf_why"] = list(AD.get("conf_why", [])) + ["the other indices agree"]


def index_strip():
    """My field's standing on every index (percentile among the ranked fields), heavier = more weight at this stage."""
    rows = [{"vi": v, "pct": float(MYC[f"pct_{v}"]), "rank": int(MYC[f"rank_{v}"]), "w": W_NOW.get(v, 1.0),
             "why": G.VI_EVIDENCE.get(v, ""), "best": v == VI} for v in G.ALL_VI if f"pct_{v}" in MYC and pd.notna(MYC.get(f"pct_{v}"))]
    if not rows:
        return
    d = pd.DataFrame(rows).sort_values(["w", "pct"], ascending=[False, False])
    d["label"] = d["vi"] + np.where(d["best"], " ★", "") + "  (×" + d["w"].map(lambda x: f"{x:g}") + ")"
    order = list(d["label"])
    xs = alt.X("pct:Q", scale=alt.Scale(domain=[0, 100]), title="my field's percentile among the ranked fields (100 = best)",
               axis=alt.Axis(values=[0, 25, 50, 75, 100], grid=False, labelFontSize=10))
    ys = alt.Y("label:N", sort=order, title=None, axis=alt.Axis(labelFontSize=11, ticks=False, domain=False))
    bands = alt.Chart(pd.DataFrame({"x0": [0, 25, 50, 75], "x1": [25, 50, 75, 100], "c": ["#fee2e2", "#ffedd5", "#ecfccb", "#dcfce7"]})).mark_rect(opacity=0.7).encode(
        x=alt.X("x0:Q", scale=alt.Scale(domain=[0, 100])), x2="x1:Q", color=alt.Color("c:N", scale=None))
    dots = alt.Chart(d).mark_circle(stroke="#fff", strokeWidth=1.5).encode(
        x=xs, y=ys, size=alt.Size("w:Q", scale=alt.Scale(domain=[0.5, 3], range=[60, 360]), legend=None),
        color=alt.condition("datum.best", alt.value(BLUE), alt.value(INK)),
        tooltip=[alt.Tooltip("vi:N", title="index"), alt.Tooltip("rank:Q", title="my rank"), alt.Tooltip("pct:Q", format=".0f", title="percentile"),
                 alt.Tooltip("w:Q", title="weight now"), alt.Tooltip("why:N", title="evidence")])
    cons = alt.Chart(pd.DataFrame({"pct": [float(MYC["cons_score"])]})).mark_rule(color="#7c3aed", strokeWidth=2.5, strokeDash=[5, 3]).encode(x="pct:Q")
    st.altair_chart((bands + cons + dots).properties(height=34 * len(d) + 30).configure_view(strokeWidth=0), use_container_width=True)
tab_o, tab_f, tab_g, tab_m, tab_ts, tab_s, tab_d = st.tabs(
    ["Overview", "Filtering", "Green-up", "Map & ranking", "TS vs reference", "Season summary", "Data"])
tab_det = tab_o            # details sit right under the answer in the same tab
with tab_o, card():
    is_today = CHECK_DATE >= dt.date.today()
    eyebrow_ = (f"Check date · today, {CHECK_DATE:%b %d, %Y}" if is_today else f"Check date · {CHECK_DATE:%b %d, %Y} (looking back)")
    sub_ = (f"My field's image of <b>{IMG_DATE:%b %d}</b> against {N_RANKED - 1} of the {N_COMP} comparable fields (each on its clear image "
            f"within ±{RANK_WINDOW} days) — exactly the numbers of my rank, the map and the ranking." if IMG_DATE is not None else
            "Press <b>Rank my field</b> in the <i>My rank</i> card and pick a date — this answer, my rank, the map and the ranking all follow it.")
    html(f'<div class="n-eyebrow">{eyebrow_}</div>'
         f'<div class="ctitle" style="font-size:1.2rem;margin-top:.15rem">Should I scout my field?</div>'
         f'<div class="csub">{sub_}</div>')

    stg = (NS or {}).get("stage") or st_today
    if AD is None:  # not ranked yet → a light placeholder, nothing is read
        d1, d2 = st.columns([1.25, 1])
        with d1:
            html('<div style="background:#f8fafc;border:1px dashed #cbd5e1;border-radius:18px;padding:14px 18px">'
                 '<div style="font-size:1.1rem;font-weight:800;color:#334155">🛰️ Rank my field to get the answer</div>'
                 '<div style="color:#475569;font-size:.88rem;margin-top:.3rem">The scout answer, “where my field stands” and the trend of the last weeks '
                 'are worked out only for the date you rank on — nothing is downloaded before that. '
                 'Start with <b>Rank my field</b> in the <i>My rank</i> card above.</div></div>')
        with d2:
            if st_today and st_today.key != "pre":
                html(f'<div class="ctitle" style="margin-top:.2rem">Season stage on {CHECK_DATE:%b %d}</div>{stage_timeline(R["crop"], (CHECK_DATE - my_plant).days)}')
    else:
        bgc, bdc, fgc = TONE[AD["level"]]
        conf_pill = ({"high": ("g", "High confidence"), "medium": ("a", "Medium confidence"), "low": ("r", "Low confidence")}.get(AD["conf"]) if AD.get("conf") else None)
        d1, d2 = st.columns([1.2, 1])
        with d1:
            html(f'<div style="background:{bgc};border:1px solid {bdc};border-radius:18px;padding:16px 18px;min-height:178px">'
                 f'<div style="font-size:1.55rem;font-weight:800;letter-spacing:-.02em;color:{fgc}">{AD["icon"]} {AD["headline"]}</div>'
                 f'<div style="color:#334155;font-size:.95rem;line-height:1.45;margin:.4rem 0 .7rem">{AD["reason"]}</div>'
                 '<div class="pills">'
                 + (f'<span class="pill {conf_pill[0]}" title="{" · ".join(AD.get("conf_why", []))}">{conf_pill[1]}</span>' if conf_pill else "")
                 + (f'<span class="pill">🛰️ image {NS["date"]:%b %d}</span>' if NS.get("ok") else "")
                 + (f'<span class="pill dark">{stg.label}</span>' if stg else "") + '</div>'
                 + (f'<div style="margin-top:.75rem;font-size:.86rem;color:#334155"><b>What to look for:</b> {AD["look_for"]}.</div>' if AD.get("look_for") else "")
                 + '</div>')
        with d2:
            if AD["why"]:
                html('<div class="ctitle" style="margin:.2rem 0 .45rem">Why we say this</div>'
                     + why_rows((i + 1, w_) for i, w_ in enumerate(AD["why"]))
                     + (f'<div class="note" style="margin-top:.2rem">How sure: {" · ".join(AD.get("conf_why", []))}.</div>' if AD.get("conf_why") else ""))
            elif stg:
                html(f'<div class="ctitle" style="margin-top:.2rem">Season stage</div>{stage_timeline(R["crop"], (NOW_ASOF - my_plant).days)}')
        if NS and NS.get("ok"):
            v1, vR = st.columns([1.5, 1])
            with v1:
                html(f'<div class="ctitle" style="margin-top:.6rem">Where my field stands · {NS["date"]:%b %d}</div>'
                     f'<div class="csub">each dot = one comparable field (its clear image nearest {NS["date"]:%b %d}) · 100% = the best fields</div>')
                where_you_stand(NS["day"], NS["date"])
                with popover("ⓘ What does “% of the best” mean?"):
                    st.markdown(
                        f"**% of the best = my field's greenness ÷ the greenness of the best comparable fields × 100** (the SI, as a %).\n\n"
                        f"- **The best comparable fields** = the level the top {100 - q * 100:.0f} % of the other comparable fields reach (their {q * 100:.0f}th percentile, "
                        f"q{int(q * 100)}), on the {VI_TAG} index, from the image of {NS['date']:%b %d}. My own field is never part of it.\n"
                        f"- **100 %** → my field is as green as the best fields · **80 %** → 20 % less green · above 100 % → greener than that level.\n"
                        f"- **Example now:** my field {NS['me']:.3f} ÷ best fields {NS['ref']:.3f} = **{NS['SI'] * 100:.0f} %**.\n"
                        f"- Using the top fields (not the average) asks *how close am I to what is achievable here this season*, "
                        f"and the {q * 100:.0f}th percentile instead of the single best field keeps one odd field from setting the bar. "
                        f"It is taken over all comparable fields, so only about {100 - q * 100:.0f} % of them can score above 100 %.\n"
                        f"- Colours: green ≥ 95 % · light green 85–95 % · orange 70–85 % · red under 70 %.")
            with vR:
                html(f'<div class="ctitle" style="margin-top:.6rem">Season stage</div>{stage_timeline(R["crop"], NS["dap"])}'
                     '<div class="note" style="margin-top:1rem">Below: the trend of the last weeks, the ★ yield-critical stage so far, '
                     'and how the answer is decided.</div>')
# ============================================================================ sub-tabs (Overview / Details first)

# ---------------------------------------------------------------- Filtering

with tab_det:
    if not (NS and NS.get("ok")):
        pass                                   # the Overview card above already says "Rank my field" and shows the stage
    else:
        with card():
            html(f'<div class="ctitle" style="font-size:1.15rem">Details behind the answer · image {NS["date"]:%b %d}</div>'
                 f'<div class="csub">same numbers as Overview, my rank, the map and the ranking</div>')
            v2, v3 = st.columns([1.3, 1])
            with v2:
                html('<div class="ctitle" style="margin-top:.2rem">The last few weeks</div><div class="csub">my field as % of the best comparable fields on each of its clear images — same method as my rank</div>')
                if len(NS["ts"]) >= 2:
                    few_weeks(NS["ts"])
                else:
                    html('<div class="note">Only one clear image so far — the trend appears after the next one.</div>')
            with v3:
                w_ = NS.get("window"); cw_ = G.critical(R["crop"])
                if w_:
                    since_ = CHECK_DATE - dt.timedelta(days=int(max_days))
                    part_ = (f' · images since {since_:%b %d} (full window: Season summary)' if w_["start"] < since_ else "")
                    html(f'<div class="wbox" style="margin-top:.3rem"><div class="lab">★ Yield-critical stage so far</div>'
                         f'<div class="val">{round(w_["SI"] * 100)}% <small>of the best · {w_["rank"]}{G._ord(w_["rank"])} of {w_["n"]}</small></div>'
                         f'<div class="sub">{cw_.label} · {w_["start"]:%b %d} – {w_["end"]:%b %d} · mean over my {w_["looks"]} clear image{"s" if w_["looks"] != 1 else ""} in this stage · '
                         f'{w_["n"]} = my field + {w_["n"] - 1} of {N_COMP} comparable{part_}</div></div>')
                else:
                    start_ = my_plant + dt.timedelta(days=cw_.start)
                    html(f'<div class="wbox" style="margin-top:.3rem"><div class="lab">★ Yield-critical stage</div>'
                         f'<div class="val" style="font-size:1.2rem">{"starts " + start_.strftime("%b %d") if NOW_ASOF < start_ else "no clear look yet"}</div>'
                         f'<div class="sub">{cw_.label} · the part of the season most tied to yield</div></div>')
        with st.expander("How the answer is decided · second opinion from all indices", expanded=False):
            with card():
                html('<div class="ctitle">How the answer is decided</div><div class="csub">the same rules for every field</div>'
                     '<div style="margin-top:.5rem"></div>' + why_rows([
                         ("🔎", "<b>Scout this field</b> — under 70% of the best fields · or under 85% <i>and</i> behind on 2 of the last 3 images or "
                                "slipping 5+ points · or under 80% during the ★ yield-critical stage"),
                         ("👀", "<b>Keep an eye on it</b> — under 85% on one image · or under 95% and slipping · or among the least green quarter of "
                                "the comparable fields"),
                         ("✅", "<b>On track</b> — everything else. Late season and harvested fields get information, not a scouting call — "
                                "greenness falls there for normal reasons.")]) +
                     f'<div class="note">Index: {VI_TAG} — {G.VI_HELP.get(VI, "")}. Reference = the q{int(q * 100)} (top {100 - q * 100:.0f} %) of the other '
                     'comparable fields. Stages are typical Georgia timings counted from planting — approximate.</div>')
            if MYC is not None and pd.notna(MYC["cons_score"]):
                with card():
                    html(f'<div class="ctitle">Second opinion · all {int(MYC["n_vi"])} indices on {IMG_DATE:%b %d}</div>'
                         f'<div class="csub">my field ranked on every index with the same image and the same comparable fields · bigger dot = more weight '
                         f'for {R["crop"]} at this stage ({RANK_STAGE.label if RANK_STAGE else "—"}) · blue = ★ {VI} · purple line = ◎ consensus score '
                         f'{MYC["cons_score"]:.0f} → <b>#{int(MYC["cons_rank"])} of {N_CONS}</b></div>')
                    index_strip()
                    if AGREE:
                        html(f'<div class="note">{AGREE["n_bottom_half"]} of {AGREE["n"]} indices put my field in the bottom half '
                             f'({AGREE["w_bottom_half"] * 100:.0f}% of the weight) · best on <b>{AGREE["best_vi"]}</b> (#{AGREE["best_rank"]}), '
                             f'weakest on <b>{AGREE["worst_vi"]}</b> (#{AGREE["worst_rank"]}). Ranks are combined, not SI values — each index spreads '
                             f'fields differently (NDVI bunches up in dense canopies, CIre stretches them). Choose <b>◎ Consensus</b> in Map & ranking '
                             f'to rank every field this way.</div>')
        with st.expander("New here? How to read this check", expanded=False):
            st.markdown(
                "- **Comparable fields** are the same crop, planted within a few days of mine, close by — plus any filters switched on in the sidebar "
                "(elevation, soil, irrigation). Comparing with them removes weather and planting-date differences.\n"
                f"- **% of the best** = my field's greenness ÷ the greenness of the best comparable fields (the top {100 - q * 100:.0f} %) on the image my field is ranked on — "
                "fields seen on a nearby clear day are first calibrated to that image. 100% means as green as the best; 80% means 20% less green.\n"
                "- **Colours:** green ≥ 95% · light green 85–95% · orange 70–85% · red under 70%.\n"
                "- **Greenness is not yield**, but during the ★ yield-critical stage it is the best early sign there is. A satellite can show *that* a field is behind, "
                "not *why* — the check tells you where to spend scouting time; the field visit tells you the cause.\n"
                "- **One date for everything:** the date you pick with **Rank my field** (or **📅 Change date** in the *My rank* card) drives this answer, "
                "my rank, the map and the ranking — all from the newest clear image on or before it, so the numbers always agree. Pick today for the "
                "latest images (new clear images arrive every few days when it isn't cloudy), or any earlier date to look back.")

with tab_f:
    f1, f2 = st.columns([1.5, 1])
    with f1, card():
        html('<div class="ctitle" style="font-size:1.15rem">How the comparison set is built</div>'
             '<div class="csub">each step keeps only the fields that are similar to mine — switch filters on or off in the sidebar</div>')
        n_dated = int((nb["plant_conf"].eq("high") | (nb["plant_conf"].eq("low") & (plant_mode == "All estimated"))).sum())
        stages = [("In search area", n_all, f"{R['crop']} · ≥ {C.CROP_PARAMS[R['crop']].min_acres_10m:.0f} ac · within {R['radius']} km"),
                  ("Planting date found", n_dated, "well dated" if plant_mode == "High confidence" else "well or roughly dated"),
                  ("Similar planting", n_plant, f"planted within ± {window} days of mine")]
        stages += [(lab, n, rule) for _, lab, n, rule in STAGES]
        pal = [("#f1f5f9", "#e2e8f0", "#0f172a"), ("#f0fdf4", "#dcfce7", "#14532d"), ("#dcfce7", "#bbf7d0", "#14532d"), ("#bbf7d0", "#86efac", "#14532d"),
               ("#86efac", "#4ade80", "#14532d"), ("#4ade80", "#22c55e", "#052e16")]
        W, H, GAP, LBL = 760, 84, 12, 150
        FW = W - LBL                                       # funnel area; right column holds the "dropped" badges
        n0 = max(stages[0][1], 1)
        widths = [max(0.52, (s_[1] / n0) ** 0.5) * FW for s_ in stages]
        for i in range(1, len(widths)):
            widths[i] = min(widths[i], widths[i - 1] * 0.9)
        widths = [max(w_, 0.46 * FW) for w_ in widths]
        tot_h = len(stages) * (H + GAP)
        svg = [f'<svg viewBox="0 0 {W} {tot_h}" width="100%" style="display:block">', "<defs>"]
        for i in range(len(stages)):
            last = i == len(stages) - 1 and i > 0
            c1, c2, _ = ("#16a34a", "#065f46", "#fff") if last else pal[min(i, len(pal) - 1)]
            svg.append(f'<linearGradient id="fg{i}" x1="0" y1="0" x2="0" y2="1"><stop offset="0" stop-color="{c1}"/><stop offset="1" stop-color="{c2}"/></linearGradient>')
        svg.append('<filter id="sh" x="-5%" y="-5%" width="110%" height="130%"><feDropShadow dx="0" dy="3" stdDeviation="4" flood-opacity=".12"/></filter></defs>')
        for i, (lab, n, rule) in enumerate(stages):
            last = i == len(stages) - 1 and i > 0
            txt = "#ffffff" if last else pal[min(i, len(pal) - 1)][2]
            top = widths[i]; bot = widths[i + 1] if i + 1 < len(widths) else max(top * 0.88, 0.42 * FW)
            y = i * (H + GAP); cx = FW / 2
            r_ = 14
            svg.append(f'<path filter="url(#sh)" fill="url(#fg{i})" d="M{cx - top / 2 + r_},{y} L{cx + top / 2 - r_},{y} Q{cx + top / 2},{y} {cx + top / 2 - 2},{y + r_} '
                       f'L{cx + bot / 2 + 2},{y + H - r_} Q{cx + bot / 2},{y + H} {cx + bot / 2 - r_},{y + H} L{cx - bot / 2 + r_},{y + H} '
                       f'Q{cx - bot / 2},{y + H} {cx - bot / 2 - 2},{y + H - r_} L{cx - top / 2 + 2},{y + r_} Q{cx - top / 2},{y} {cx - top / 2 + r_},{y} Z"/>')
            svg.append(f'<text x="{cx}" y="{y + 36}" text-anchor="middle" font-family="Inter,sans-serif" fill="{txt}">'
                       f'<tspan font-size="26" font-weight="800">{n}</tspan><tspan font-size="16" font-weight="700" dx="10">{lab}</tspan></text>')
            rule_s = rule if len(rule) <= 52 else rule[:50] + "…"
            svg.append(f'<text x="{cx}" y="{y + 60}" text-anchor="middle" font-family="Inter,sans-serif" font-size="13" fill="{txt}" opacity=".78">{rule_s}</text>')
            if i > 0:
                dropped = stages[i - 1][1] - n
                pct = dropped / max(stages[i - 1][1], 1) * 100
                bx = FW + 18
                svg.append(f'<rect x="{bx}" y="{y + H / 2 - 18}" rx="18" width="{LBL - 26}" height="36" fill="#fff1f2"/>')
                svg.append(f'<text x="{bx + (LBL - 26) / 2}" y="{y + H / 2 + 5}" text-anchor="middle" font-family="Inter,sans-serif" font-size="13" font-weight="700" fill="#be123c">'
                           f'−{dropped} ({pct:.0f}%)</text>')
            else:
                bx = FW + 18
                svg.append(f'<text x="{bx + (LBL - 26) / 2}" y="{y + H / 2 + 5}" text-anchor="middle" font-family="Inter,sans-serif" font-size="12" font-weight="600" fill="#94a3b8">dropped</text>')
        svg.append("</svg>")
        html("".join(svg))
        off = [lab for lab, on in (("irrigation", t_irr), ("elevation", t_elev), ("soil", t_soil)) if not on]
        if off:
            html(f'<div class="legend"><span>Turn on {" · ".join(off)} in the sidebar to narrow further</span></div>')
    with f2, card():
        html('<div class="ctitle" style="font-size:1.15rem">Where the fields went</div><div class="csub">click a slice — those fields are highlighted on the map below</div>')
        dd = nb["category"].value_counts().rename_axis("key").reset_index(name="n")
        dd = dd[dd["n"] > 0]
        dd["category"] = dd["key"].map(CAT_LABEL)
        dd["pct"] = dd["n"] / max(dd["n"].sum(), 1)
        order = [lab for k, lab, _ in CATS if lab in set(dd["category"])]
        dd["order"] = dd["category"].map({lab: i for i, lab in enumerate(order)})
        pick_cat = alt.selection_point(name="pick_cat", fields=["category"], on="click", clear="dblclick")
        arc = alt.Chart(dd).mark_arc(innerRadius=105, outerRadius=165, cornerRadius=6, padAngle=0.018, cursor="pointer").encode(
            theta=alt.Theta("n:Q", stack=True), order=alt.Order("order:Q"),
            color=alt.Color("category:N", scale=alt.Scale(domain=order, range=[CAT_COLOR[k] for k, lab, _ in CATS if lab in order]),
                            legend=alt.Legend(title=None, orient="bottom", columns=2, labelFontSize=12, symbolType="circle", symbolSize=140)),
            opacity=alt.condition(pick_cat, alt.value(1), alt.value(0.3)),
            tooltip=[alt.Tooltip("category:N", title=""), alt.Tooltip("n:Q", title="fields"), alt.Tooltip("pct:Q", title="share", format=".0%")]).add_params(pick_cat)
        ctr = alt.Chart(pd.DataFrame({"t": [str(n_final)], "s": ["comparable fields"], "u": [f"of {n_all}"]}))
        donut = (arc + ctr.mark_text(fontSize=44, fontWeight=800, dy=-12, color=INK).encode(text="t:N")
                 + ctr.mark_text(fontSize=13, dy=20, color="#64748b").encode(text="s:N")
                 + ctr.mark_text(fontSize=12, dy=38, color="#94a3b8").encode(text="u:N")).properties(height=480).configure_view(strokeWidth=0)
        sel_cat = None
        try:
            ev = st.altair_chart(donut, use_container_width=True, on_select="rerun", selection_mode="pick_cat", key="donut")
            pts = ev.get("selection", {}).get("pick_cat", []) if ev else []
            sel_cat = pts[0].get("category") if pts else None
        except TypeError:
            st.altair_chart(donut, use_container_width=True)

# ---------------------------------------------------------------- shared map pieces
g = GEO.merge(T[["field_id", "short", "where", "category", "est_planting", "delta_days", "elev_median", "texture", "irr_source", "irrigated"]], on="field_id", how="left")
g["category_label"] = g["category"].map({**CAT_LABEL, "me": "my field"})
g["elev_median"] = pd.to_numeric(g["elev_median"], errors="coerce").round(0)
g["acres"] = g["acres"].round(0)
g["irrigation"] = np.where(g["irrigated"].fillna(False).astype(bool), "irrigated", "dryland")
for col_ in ("est_planting", "texture", "irr_source"):
    g[col_] = g[col_].astype(str).replace({"None": "—", "nan": "—", "NaT": "—"})


def me_outline(mm):
    myg = g[g["field_id"] == "MY_FIELD"]
    folium.GeoJson(myg[["field_id", "geometry"]].to_json(), style_function=lambda f: {"fillColor": "none", "color": "#60a5fa", "weight": 4}).add_to(mm)
    mc = myg.geometry.iloc[0].centroid
    pill_marker(mm, mc.y, mc.x, "ME" + (f" · #{int(mr['rank'])}" if (mr is not None and pd.notna(mr["rank"])) else ""), BLUE)


# ---------------------------------------------------------------- Filtering: where the fields went, on the map (linked to the donut)
with tab_f, card():
    html('<div class="ctitle">Where the fields went, on the map</div>' + (f'<div class="csub">showing <b>{sel_cat}</b> · double-click the donut to clear</div>' if sel_cat
                                                     else '<div class="csub">each field coloured by the step that left it out · click a donut slice to show just those fields</div>'))
    mm = base_map(center=(lat0, lon0), rad=R["radius"], height_px=520)
    gg = g[g["field_id"] != "MY_FIELD"]

    def sty(f):
        p = f["properties"]; col = CAT_COLOR.get(p["category"], "#cbd5e1")
        comp = p["category"] == "comparable"
        if sel_cat:
            on = p["category_label"] == sel_cat
            return {"fillColor": col if on else "#e5e7eb", "color": "#ffffff" if on else "#cbd5e1", "weight": 1.6 if on else 0.4, "fillOpacity": 0.75 if on else 0.08}
        return {"fillColor": col, "color": "#ffffff" if comp else col, "weight": 1.8 if comp else 0.8, "fillOpacity": 0.62 if comp else 0.35}
    cols_ = ["short", "where", "category_label", "acres", "est_planting", "delta_days", "elev_median", "texture", "irr_source", "field_id"]
    alias_ = ["field", "where", "", "acres", "est. planting", "planted Δ days vs mine", "elevation m", "soil", "irrigation", "CSB id"]
    folium.GeoJson(gg[cols_ + ["category", "geometry"]].to_json(), style_function=sty,
                   highlight_function=lambda f: {"weight": 3.5, "color": "#facc15", "fillOpacity": 0.8},
                   tooltip=folium.GeoJsonTooltip(fields=cols_[:4], aliases=alias_[:4]),
                   popup=folium.GeoJsonPopup(fields=cols_, aliases=alias_)).add_to(mm)
    lab_rows = gg[gg["category"] == "comparable"] if not sel_cat else gg[gg["category_label"] == sel_cat]
    if len(lab_rows) <= 150:
        for _, r_ in lab_rows.iterrows():
            rp = r_.geometry.representative_point(); label_marker(mm, rp.y, rp.x, r_["short"])
    me_outline(mm)
    folium.LayerControl(collapsed=True).add_to(mm)
    c_, z_ = circle_view(lat0, lon0, R["radius"], 520)
    st_folium(mm, height=520, use_container_width=True, key="map_cmp", returned_objects=[], center=c_, zoom=z_)
    html('<div class="legend">' + "".join(f'<span><i class="dot" style="background:{c}"></i>{lab}</span>' for k, lab, c in CATS if (nb["category"] == k).any())
         + f'<span><i class="dot" style="background:{BLUE}"></i>my field</span><span>F01… = fields by distance from mine · click a field for details</span></div>')

# ---------------------------------------------------------------- Map & ranking
MATCH_LABEL = {"planting": "Planting", "irrigation": "Irrigation", "elevation": "Elevation", "soil": "Soil"}
MATCH_HELP = {"planting": f"planted within ± {window} days of mine", "irrigation": "same irrigated / dryland status",
              "elevation": "field elevation close to mine", "soil": "same soil texture group (0–30 cm)"}
COMBO_COLORS = {  # one colour per combination; "all" is always the strong green
    ("planting",): "#f59e0b", ("irrigation",): "#3b82f6", ("elevation",): "#a855f7", ("soil",): "#a16207",
    ("planting", "irrigation"): "#14b8a6", ("planting", "elevation"): "#ec4899", ("planting", "soil"): "#ea580c",
    ("irrigation", "elevation"): "#6366f1", ("irrigation", "soil"): "#0891b2", ("elevation", "soil"): "#be185d",
}


def combo_frame(keys):
    """Per field: which of the chosen criteria it matches → a label and a colour."""
    M = MATCH.set_index("field_id")[list(keys)]
    n_ = M.sum(axis=1)
    lab_, col_, ordr = [], [], []
    for fid_, row_ in M.iterrows():
        hit = tuple(k for k in keys if row_[k])
        if len(hit) == len(keys):
            lab_.append("All: " + " + ".join(MATCH_LABEL[k] for k in keys) if len(keys) > 1 else MATCH_LABEL[keys[0]]); col_.append("#15803d"); ordr.append(0)
        elif not hit:
            lab_.append("None of these"); col_.append("#cbd5e1"); ordr.append(9)
        else:
            lab_.append(" + ".join(MATCH_LABEL[k] for k in hit) + " only"); col_.append(COMBO_COLORS.get(hit, "#94a3b8")); ordr.append(len(keys) - len(hit))
    return pd.DataFrame({"field_id": M.index, "match": lab_, "match_color": col_, "match_order": ordr, "n_match": n_.values})


sel_id = None
if ranked is not None:
    try:
        ptsb = (ss.get("rank_bars") or {}).get("selection", {}).get("pick", [])
        sel_id = ptsb[0].get("field_id") if ptsb else None
    except Exception:
        sel_id = None
MAP_H = 700 if ranked is not None else 580       # fills its card next to the ranking chart
with tab_m:
    m1, m2 = st.columns([1.55, 1])
    with m1, card():
        hcol, vcol = st.columns([1.2, 1])
        with hcol:
            html('<div class="ctitle">Map</div><div class="csub">hover a field for details · click it for more</div>')
        with vcol:
            views = ["Similarity", "Ranking (SI)"] if ranked is not None else ["Similarity"]
            view = segmented("View", views, views[-1], f"map_view_{len(views)}")
        if view == "Similarity" or ranked is None:
            default_keys = ["planting"] + [k for k, on in (("irrigation", t_irr), ("elevation", t_elev), ("soil", t_soil)) if on]
            if len(default_keys) == 1:
                default_keys.append("irrigation")            # the two matches farmers care about most
            opts_ = list(MATCH_KEYS)
            if hasattr(st, "pills"):
                kw_ = {} if "match_keys" in ss else {"default": default_keys[:3]}
                pick_ = st.pills("Colour fields by what they share with mine", opts_, selection_mode="multi", key="match_keys",
                                 format_func=lambda k: MATCH_LABEL[k], **kw_)
            else:
                pick_ = st.multiselect("Colour fields by what they share with mine", opts_, default=default_keys[:3], key="match_keys",
                                       format_func=lambda k: MATCH_LABEL[k])
            keys_ = [k for k in opts_ if k in (pick_ or [])][:3] or ["planting"]
            if pick_ and len(pick_) > 3:
                st.caption("Up to three at a time keeps the colours readable — showing the first three.")
            CB = combo_frame(keys_)
            gm = g.merge(CB, on="field_id", how="left")
            gm["match"] = gm["match"].fillna("None of these"); gm["match_color"] = gm["match_color"].fillna("#cbd5e1")
            mm = base_map(center=(lat0, lon0), rad=R["radius"], height_px=MAP_H)
            gg = gm[gm["field_id"] != "MY_FIELD"]
            cols_ = ["short", "where", "match", "acres", "est_planting", "delta_days", "elev_median", "texture", "irrigation", "field_id"]
            alias_ = ["field", "where", "shares", "acres", "est. planting", "planted Δ days vs mine", "elevation m", "soil", "irrigation", "CSB id"]
            folium.GeoJson(gg[cols_ + ["match_color", "n_match", "geometry"]].to_json(),
                           style_function=lambda f: {"fillColor": f["properties"]["match_color"],
                                                     "color": "#ffffff" if f["properties"]["match_color"] == "#15803d" else f["properties"]["match_color"],
                                                     "weight": 1.8 if f["properties"]["match_color"] == "#15803d" else 0.8,
                                                     "fillOpacity": 0.72 if f["properties"]["match_color"] != "#cbd5e1" else 0.22},
                           highlight_function=lambda f: {"weight": 3.5, "color": "#facc15", "fillOpacity": 0.85},
                           tooltip=folium.GeoJsonTooltip(fields=cols_[:4], aliases=alias_[:4]),
                           popup=folium.GeoJsonPopup(fields=cols_, aliases=alias_)).add_to(mm)
            top_ = gg[gg["match_order"] == 0]
            if len(top_) <= 150:
                for _, r_ in top_.iterrows():
                    rp = r_.geometry.representative_point(); label_marker(mm, rp.y, rp.x, r_["short"])
            me_outline(mm)
            folium.LayerControl(collapsed=True).add_to(mm)
            c_, z_ = circle_view(lat0, lon0, R["radius"], MAP_H)
            st_folium(mm, height=MAP_H, use_container_width=True, key="map_sim", returned_objects=[], center=c_, zoom=z_)
            cnt_ = gg.groupby(["match", "match_color", "match_order"]).size().rename("n").reset_index().sort_values(["match_order", "n"], ascending=[True, False])
            html('<div class="legend">' + "".join(f'<span><i class="sw" style="background:{r_.match_color}"></i>{r_.match} <b>{r_.n}</b></span>' for r_ in cnt_.itertuples())
                 + f'<span><i class="sw" style="background:{BLUE}"></i>my field</span></div>'
                 + '<div class="note">' + " · ".join(f"<b>{MATCH_LABEL[k]}</b> = {MATCH_HELP[k]}" for k in keys_)
                 + f". Each criterion is judged on its own here — the sidebar filters decide the comparable set ({n_final} fields).</div>")
        else:
            g_r = g.merge(ranked[["field_id", "SI", "rank", "vi_median", "is_ref_field", "in_cohort"] + [c for c in ("cons_score", "cons_rank", "cons_pct") if c in ranked]],
                          on="field_id", how="left")
            g_r["is_ref_field"] = g_r["is_ref_field"].fillna(False).astype(bool); g_r["in_cohort"] = g_r["in_cohort"].fillna(False).astype(bool)
            si_map(g_r, VI, "map_si", sel_id=sel_id, center=(lat0, lon0), rad=R["radius"], height=MAP_H, consensus=USE_CONS and "cons_pct" in g_r)

    with m2, card():
        if ranked is not None:
            coh_ = ranked[ranked["in_cohort"] & ranked["vi_median"].notna() & ranked["rank"].notna()].sort_values("rank")
            html(f'<div class="ctitle">Ranking</div><div class="csub">my field + {N_RANKED - 1} of {N_COMP} comparable fields · click a bar to zoom on the map</div>')
            if True:
                if "vi_seg" in ss and ss["vi_seg"] not in [G.BEST, G.CONSENSUS] + list(G.ALL_VI):
                    ss["vi_seg"] = G.BEST
                _bv = G.recommended_vi(R["crop"], (RANK_DATE - R["planting"]).days)
                _lbl = {G.BEST: f"★ Best now ({_bv})", G.CONSENSUS: f"◎ Consensus of all {len(G.ALL_VI)}"}
                st.selectbox("Rank by", [G.BEST, G.CONSENSUS] + list(G.ALL_VI), key="vi_seg", label_visibility="collapsed",
                             format_func=lambda v: _lbl.get(v, v), **({} if "vi_seg" in ss else {"index": 0}))
            if USE_CONS:
                wtxt = " · ".join(f"{v} ×{w:g}" for v, w in sorted(W_NOW.items(), key=lambda x: -x[1]))
                html(f'<div class="note" style="margin:.1rem 0 .5rem">◎ <b>Consensus</b> — every field ranked on all {len(G.ALL_VI)} indices, '
                     f'turned into percentiles and averaged with weights for {R["crop"]} at {RANK_STAGE.label if RANK_STAGE else "this stage"}: {wtxt}. '
                     f'The scout answer still reads “% of the best” on ★ {VI}; the consensus checks it.</div>')
            else:
                html(best_note(VI_CHOICE, VI, f"on {RANK_DATE:%b %d}" + (f" ({RANK_STAGE.label})" if RANK_STAGE else "")))
            if USE_CONS and "cons_score" in coh_ and coh_["cons_score"].notna().any():
                consensus_bar_chart(coh_, "rank_bars")
            elif len(coh_):
                rank_bar_chart(coh_, ref, VI, "rank_bars", VI)
            else:
                html(f'<div class="callout warn">Fewer than 3 comparable fields had a clear image near {RANK_DATE:%b %d}, so there is no reference to rank against. '
                     f'Widen <b>Planting within ± days</b>, choose <b>All estimated</b> planting dates, switch a filter off, or pick another date.</div>')
            if sel_id is not None and st.button("↺ Reset map zoom"):
                ss.pop("rank_bars", None); st.rerun()
            refs = ranked[ranked["is_ref_field"]]
            _ab = int((coh_["SI"] > 1).sum()) if "SI" in coh_ else 0
            html(f'<div class="note">{_ab} of {len(coh_)} ranked fields are above the reference (SI > 1) — with q{int(q * 100)} about '
                 f'{max(round(len(coh_) * (1 - q)), 1)} is expected. ' 
                 f'Dashed line = reference (SI 1 = {VI} {ref:.3f}, q{int(q * 100)} of comparable fields) · '
                 f'{len(refs)} reference field{"s" if len(refs) != 1 else ""} (◆ REF) · image: {", ".join(IMG_IDS) or "—"}</div>')
        else:
            html('<div class="ctitle">Ranking</div><div class="csub">not ranked yet</div>')
            html('<div class="note" style="margin-bottom:.8rem">Press <b>Rank my field</b> in the <i>My rank</i> card above and pick a date — the ranking of all '
                 'comparable fields appears here, and the map gets a <b>Ranking (SI)</b> view.</div>')

# ---------------------------------------------------------------- TS vs reference
def _season_missing_banner(key):
    if season_missing:
        cA, cB = st.columns([3, 1])
        cA.markdown(f'<div class="callout warn" style="margin:0">The filters now include {len(season_missing)} field'
                    f'{"s" if len(season_missing) != 1 else ""} that were not loaded for this season — they are left out until you update.</div>', unsafe_allow_html=True)
        if cB.button("Load them", key=key, use_container_width=True):
            ss.season_request = ss.season_key; st.rerun()


def _season_cta(key):
    html('<div class="ctitle">Season not loaded yet</div>'
         '<div class="note" style="margin-bottom:.9rem">Pick a date range — FieldRank reads every clear Sentinel-2 look of your field and the similar-planting '
         'fields once, then shows <b>TS vs reference</b> and the <b>Season summary</b> instantly.</div>')
    if season_dialog and st.button("Choose season range", type="primary", key=key):
        season_dialog()


comp_ids = T.loc[T["in_window"] & ~is_me, "field_id"]
ref_ids = T.loc[T["ref_ok"] & ~is_me, "field_id"]
with tab_ts, card():
    if S_season is None:
        _season_cta("season_open_ts")
    else:
        _season_missing_banner("season_more_ts")
        h1, h2, h3 = st.columns([1.6, 1.2, 0.6])
        with h1:
            html(f'<div class="ctitle" style="font-size:1.15rem">My field vs the reference, date by date</div>'
                 f'<div class="csub">{ss.season_key[0]} → {ss.season_key[1]} · reference = q{int(q * 100)} of comparable fields on each clear date</div>')
        with h2:
            vi_ts_choice = segmented("Index", [G.BEST] + list(G.ALL_VI), G.BEST, "vi_ts")
            vi_ts = G.resolve_vi(vi_ts_choice, R["crop"])
        with h3:
            if season_dialog and st.button("📅 Range", key="season_change_ts", use_container_width=True):
                season_dialog()
        html(best_note(vi_ts_choice, vi_ts, f"over the season (yield-critical stage: {G.critical(R['crop']).label})"))
        min_n = 5 if len(comp_ids) >= 10 else 3
        ts = ts_vs_reference(S_season, comp_ids, ref_ids, "MY_FIELD", vi_ts, q, min_n=min_n)
        if len(comp_ids) < 8:
            html(f'<div class="callout warn">Only {len(comp_ids)} comparable fields with the current filters — the reference on each date rests on very few fields. '
                 'Turning a filter off in the sidebar gives a steadier line.</div>')
        tr = si_trend(ts)
        if ts.empty or tr["n"] == 0:
            st.info(f"Not enough clear dates with my field and at least {min_n} comparable fields in this range — try a longer range or fewer filters.")
        else:
            arrow = {"improving": "↗", "declining": "↘", "steady": "→"}.get(tr["word"], "")
            cols4 = st.columns(4)
            for col_, (lab, val, sub) in zip(cols4, [
                    ("Season SI (mean)", f"{tr['mean']:.2f}", f"over {tr['n']} clear dates"),
                    ("Latest SI", f"{tr['latest']:.2f}", f"on {ts.dropna(subset=['SI'])['date'].max():%b %d}"),
                    ("Trend", f"{arrow} {tr['word']}", "last 6 dates" + (f" · {tr['slope10']:+.02f} SI / 10 d" if pd.notna(tr["slope10"]) else "")),
                    ("Dates at SI ≥ 0.85", f"{tr['share_good']:.0%}", "of my clear dates")]):
                with col_:
                    html(f'<div class="kpi" style="min-height:0;background:#f8fafc;border-radius:16px;padding:12px 14px">'
                         f'<div class="lab">{lab}</div><div class="val" style="font-size:1.6rem">{val}</div><div class="sub">{sub}</div></div>')
            tsl = ts.copy(); tsl["si_class"] = tsl["SI"].apply(si_class)
            x = alt.X("date:T", title=None, axis=alt.Axis(format="%b %d", grid=False))
            band = alt.Chart(tsl).mark_area(opacity=0.18, color="#16a34a").encode(x=x, y=alt.Y("p25:Q", title=vi_ts), y2="p75:Q")
            med = alt.Chart(tsl).mark_line(color="#16a34a", strokeDash=[2, 3], strokeWidth=1.5).encode(x=x, y="p50:Q")
            refl = alt.Chart(tsl).mark_line(color=REF_COLOR, strokeWidth=2.5, point=alt.OverlayMarkDef(size=28, color=REF_COLOR)).encode(
                x=x, y="ref:Q", tooltip=[alt.Tooltip("date:T", format="%b %d"), alt.Tooltip("ref:Q", format=".3f", title="reference"), alt.Tooltip("n:Q", title="comparable clear")])
            mypts = alt.Chart(tsl.dropna(subset=["me"])).mark_circle(size=70, color=BLUE).encode(
                x=x, y="me:Q", tooltip=[alt.Tooltip("date:T", format="%b %d"), alt.Tooltip("me:Q", format=".3f", title="my field"), alt.Tooltip("SI:Q", format=".2f"),
                                        alt.Tooltip("rank:Q", title="rank that day"), alt.Tooltip("n:Q", title="of")])
            mytrend = alt.Chart(tsl.dropna(subset=["me"])).transform_loess("date", "me", bandwidth=0.6).mark_line(color=BLUE, strokeWidth=3).encode(x=x, y="me:Q")
            lgd = pd.DataFrame({"k": ["my field", "reference (q%d)" % int(q * 100), "comparable fields (middle 50%)"], "c": [BLUE, REF_COLOR, "#16a34a"]})
            leg = alt.Chart(lgd).mark_point(opacity=0).encode(color=alt.Color("k:N", scale=alt.Scale(domain=list(lgd["k"]), range=list(lgd["c"])),
                                                                              legend=alt.Legend(title=None, orient="top", symbolType="circle", symbolOpacity=1, labelFontSize=12)))
            st.altair_chart((band + med + refl + mytrend + mypts + leg).properties(height=330, title=alt.TitleParams(f"{vi_ts} over the season", anchor="start", fontSize=13))
                            .configure_view(strokeWidth=0), use_container_width=True)
            bands = pd.DataFrame({"y0": [0.0, 0.70, 0.85, 0.95], "y1": [0.70, 0.85, 0.95, 1.30], "c": ["< 0.70", "0.70–0.85", "0.85–0.95", "≥ 0.95"]})
            bg = alt.Chart(bands).mark_rect(opacity=0.10).encode(y=alt.Y("y0:Q", scale=alt.Scale(domain=[max(0.0, float(tsl["SI"].min()) - 0.1), min(1.3, max(1.05, float(tsl["SI"].max()) + 0.05))], clamp=True), title="SI"),
                                                               y2="y1:Q", color=alt.Color("c:N", scale=alt.Scale(domain=SI_DOMAIN, range=SI_RANGE), legend=None))
            one = alt.Chart(pd.DataFrame({"y": [1.0]})).mark_rule(color=REF_COLOR, strokeDash=[4, 3]).encode(y="y:Q")
            sip = alt.Chart(tsl.dropna(subset=["SI"])).mark_circle(size=90, stroke="#fff", strokeWidth=1.5, opacity=1).encode(
                x=x, y="SI:Q", color=alt.Color("si_class:N", scale=alt.Scale(domain=SI_DOMAIN, range=SI_RANGE), legend=None),
                tooltip=[alt.Tooltip("date:T", format="%b %d"), alt.Tooltip("SI:Q", format=".2f"), alt.Tooltip("rank:Q", title="rank that day"), alt.Tooltip("n:Q", title="of")])
            sit = alt.Chart(tsl.dropna(subset=["SI"])).transform_loess("date", "SI", bandwidth=0.6).mark_line(color=INK, strokeWidth=2.5).encode(x=x, y="SI:Q")
            st.altair_chart((bg + one + sit + sip).properties(height=250, title=alt.TitleParams("Sufficiency index (mine ÷ reference) — trend line = smoothed", anchor="start", fontSize=13))
                            .configure_view(strokeWidth=0), use_container_width=True)
            html(f'<div class="note">Only dates where my field and at least {min_n} comparable fields were clear are shown · comparable = current filters '
                 f'({n_final} fields) · {S_season["date"].nunique()} clear dates read in this range.</div>')

# ---------------------------------------------------------------- Season summary
with tab_s:
    if S_season is None:
        with card():
            _season_cta("season_open_s")
    else:
        _season_missing_banner("season_more_s")
        c1_, c2_, c3_, c4_ = st.columns([1.4, 1.1, 1.1, 0.55])
        with c1_:
            html(f'<div class="ctitle" style="font-size:1.15rem;margin-top:.4rem">Whole-season ranking</div>'
                 f'<div class="csub">{ss.season_key[0]} → {ss.season_key[1]} · each field\'s season value from all its clear looks</div>')
        with c2_:
            stat_lbl = segmented("Statistic", ["Season median", "Season peak", "★ Yield window"], "Season median", "season_stat")
        with c3_:
            vi_s_choice = segmented("Index", [G.BEST] + list(G.ALL_VI), G.BEST, "vi_season")
            vi_s = G.resolve_vi(vi_s_choice, R["crop"])
        with c4_:
            if season_dialog and st.button("📅 Range", key="season_change_s", use_container_width=True):
                season_dialog()
        html(best_note(vi_s_choice, vi_s, f"over the season (yield-critical stage: {G.critical(R['crop']).label})"))
        stat = {"Season median": "median", "Season peak": "peak"}.get(stat_lbl, "window")
        cw_s = G.critical(R["crop"])
        comp_s = (G.window_composite(S_season, PLANTING, R["crop"], vi_s, "mean", min_obs=2) if stat == "window"
                  else season_composite(S_season, vi_s, stat))
        Ts = T.drop(columns=[c for c in ("vi_median", "n_obs") if c in T]).merge(comp_s, on="field_id", how="left")
        mask_s = Ts["in_window"] & Ts["vi_median"].notna()
        ranked_s, ref_s = score(Ts, mask_s, "MY_FIELD", q, ref_mask=(Ts["ref_ok"] | Ts["field_id"].eq("MY_FIELD")) & Ts["vi_median"].notna())
        ranked_s["is_ref_field"] = ranked_s["in_ref"] & (ranked_s["vi_median"] >= ref_s - 1e-9) & (ranked_s["field_id"] != "MY_FIELD")
        ms = ranked_s[ranked_s["field_id"] == "MY_FIELD"].iloc[0]
        k_ = st.columns(4)
        sic = si_class(ms["SI"])
        kp = [("My season rank", f"#{int(ms['rank'])} <small>of {int(ms['n_ranked'])}</small>" if pd.notna(ms["rank"]) else "—",
               f"better than {max(ms['percentile'], 0):.0f}% of comparable fields" if pd.notna(ms["rank"]) else "too few clear looks of my field"),
              ("Season SI", f"{ms['SI']:.2f}" if pd.notna(ms["SI"]) else "—", f'<span class="pill {"g" if sic in ("≥ 0.95", "0.85–0.95") else "a" if sic == "0.70–0.85" else "r"}">{sic or "—"}</span>'),
              (f"My season {vi_s}", fmt(ms["vi_median"], ".3f"), f"reference {ref_s:.3f} (q{int(q * 100)})"),
              ("Clear looks of my field", fmt(ms.get("n_obs"), ".0f"), f"{ms.get('first_obs', '—')} → {ms.get('last_obs', '—')}")]
        for col_, (lab, val, sub) in zip(k_, kp):
            with col_, card():
                html(f'<div class="kpi" style="min-height:0"><div class="lab">{lab}</div><div class="val" style="font-size:1.8rem">{val}</div><div class="sub">{sub}</div></div>')
        sm1, sm2 = st.columns([1.55, 1])
        coh_s = ranked_s[ranked_s["in_cohort"] & ranked_s["vi_median"].notna() & ranked_s["rank"].notna()].sort_values("rank")
        sel_s = bar_selection("season_bars")
        with sm1, card():
            html('<div class="ctitle">Season map</div><div class="csub">coloured by season SI · click a bar to zoom</div>')
            g_s = GEO.merge(ranked_s[["field_id", "short", "where", "SI", "rank", "vi_median", "is_ref_field", "in_cohort"]], on="field_id", how="left")
            g_s["is_ref_field"] = g_s["is_ref_field"].fillna(False).astype(bool); g_s["in_cohort"] = g_s["in_cohort"].fillna(False).astype(bool)
            si_map(g_s, f"season {vi_s}", "season_map", sel_id=sel_s, center=(lat0, lon0), rad=R["radius"], height=540)
        with sm2, card():
            html(f'<div class="ctitle">Season ranking · {vi_s} {stat}</div><div class="csub">{len(coh_s)} comparable fields with ≥ 3 clear looks</div>')
            if len(coh_s):
                rank_bar_chart(coh_s, ref_s, vi_s, "season_bars", f"season {vi_s}")
            if stat == "window":
                html(f'<div class="note">★ Yield window = mean of each field\'s clear looks during <b>{cw_s.label}</b> (day {cw_s.start}–{cw_s.end - 1} after <i>that field\'s</i> '
                     f'estimated planting) — the part of the season most tied to yield. Fields need ≥ 2 clear looks in their window; widen the season range if many are missing. '
                     f'Dashed line = reference (SI 1).</div>')
            else:
                html(f'<div class="note">Season value = {"median" if stat == "median" else "90th percentile"} of each field\'s per-date field medians over clear looks '
                     f'(a field-level composite — same ranking as a pixel median composite, far faster). Dashed line = reference (SI 1).</div>')

# ---------------------------------------------------------------- Green-up
with tab_g:
    _lag_ = PINFO.get("my_lag_days")
    _how = "degree-days (heat units) from local weather" if PINFO.get("method") == "degree-days" else "days"
    html('<div class="howstrip">'
         f'<div class="hs"><div class="hn">1</div><div><b>You entered planting · {R["planting"]:%b %d}</b><br>'
         'This date sets the season stage, the ★ index and the match window.</div></div>'
         f'<div class="hs"><div class="hn">2</div><div><b>Satellite: your field greened up · {anchor:%b %d}</b><br>'
         + (f'{_lag_} days later. ' if _lag_ is not None else 'Not visible — a typical delay is assumed. ')
         + ({"confirmed": f'A typical delay — this <b>confirms your date</b>, and the timing (in {_how}) is learned from <i>your</i> field for this season.',
             "partly": f'The timing (in {_how}) blends your field ({PINFO.get("cal_weight", 0) * 100:.0f}%) with the {R["crop"]} default, '
                       'because the green-up was blurred or the delay unusual.'}.get(PINFO.get("cal_state"),
                                                                                    f'Typical {R["crop"]} timing (in {_how}) is used — your field could not confirm it.'))
         + '</div></div>'
         f'<div class="hs"><div class="hn">3</div><div><b>Neighbours dated the same way</b><br>Each neighbour\'s green-up is turned back into '
         f'a planting date; those planted within <b>± {window} days</b> of yours'
         + (' — counted in heat units, so they are at the same crop stage —' if HEAT_MATCH else '')
         + ' are comparable (green shading below).</div></div>'
         '</div>')
with tab_g, card():
    if gts.empty:
        st.info("No clear observations in the green-up window.")
    else:
        lcol, rcol = st.columns([1.6, 1])
        with lcol:
            ranked_line = IMG_DATE is not None and ranked is not None and pd.notna(mr["rank"])
            gi_ = segmented("Green-up index", ["NDVI", "EVI2"], "NDVI", "gu_index") if "evi2" in gts else "NDVI"
            gcol_ = gi_.lower()
            html(f'<div class="ctitle">{gi_} after planting</div><div class="csub">blue = my field · green = comparable fields · '
                 f'black dashed = the planting date you entered, shaded ± {window} days (the planting match) · blue dashed = my green-up'
                 + (f' · <b style="color:#ea580c">orange line = the image my field is ranked on ({IMG_DATE:%b %d})</b>' if ranked_line else
                    ' · rank my field to mark the ranking date here') + '</div>')
            g_src = gts[["field_id", "date", gcol_] + (["sensor"] if "sensor" in gts else [])].rename(columns={gcol_: "ndvi"})
            if S_now is not None and len(S_now) and gi_ in S_now:   # continue the curves with the ranking images, up to the ranked date
                ext = S_now[["field_id", "date", gi_]].rename(columns={gi_: "ndvi"}).assign(sensor="S2")
                ext = ext[pd.to_numeric(ext["ndvi"], errors="coerce").notna()]
                g_src = pd.concat([g_src, ext], ignore_index=True).drop_duplicates(["field_id", "date"]).sort_values(["field_id", "date"])
            gp = g_src.merge(T[["field_id", "category"]], on="field_id", how="left")      # the SAME comparable set as everywhere else
            gp["group"] = np.where(gp["field_id"] == "MY_FIELD", "my field", np.where(gp["category"].eq("comparable"), "comparable", "other"))
            if "sensor" not in gp:
                gp["sensor"] = "S2"
            gp["sensor"] = gp["sensor"].fillna("S2").map({"S2": "Sentinel-2", "L89": "Landsat (adjusted)"}).fillna("Sentinel-2")
            sc_ = alt.Scale(domain=["my field", "comparable", "other"], range=[BLUE, "#16a34a", "#e2e8f0"])
            _pl = pd.Timestamp(R["planting"])
            band = alt.Chart(pd.DataFrame({"s": [_pl - pd.Timedelta(days=window)], "e": [_pl + pd.Timedelta(days=window)],
                                           "t": [f"planted ± {window} d"]})).mark_rect(opacity=0.14, color="#16a34a").encode(
                x="s:T", x2="e:T", tooltip=alt.Tooltip("t:N", title=""))
            ln = alt.Chart(gp[gp["field_id"] != "MY_FIELD"]).mark_line(opacity=0.45, strokeWidth=1).encode(
                x=alt.X("date:T", title=None, axis=alt.Axis(format="%b %d")), y=alt.Y("ndvi:Q", title=gi_, scale=alt.Scale(domain=[0, 1])), detail="field_id:N",
                color=alt.Color("group:N", scale=sc_, legend=alt.Legend(title=None, orient="top")))
            my_ = gp[gp["field_id"] == "MY_FIELD"]
            mine = (alt.Chart(my_).mark_line(strokeWidth=3.5).encode(x="date:T", y="ndvi:Q", color=alt.Color("group:N", scale=sc_, legend=None))
                    + alt.Chart(my_).mark_point(size=70, filled=True, strokeWidth=2).encode(
                        x="date:T", y="ndvi:Q", color=alt.value(BLUE),
                        shape=alt.Shape("sensor:N", scale=alt.Scale(domain=["Sentinel-2", "Landsat (adjusted)"], range=["circle", "diamond"]),
                                        legend=alt.Legend(title=None, orient="top") if (my_["sensor"] != "Sentinel-2").any() else None),
                        fill=alt.condition(alt.datum.sensor == "Sentinel-2", alt.value(BLUE), alt.value("#ffffff")),
                        tooltip=[alt.Tooltip("date:T", format="%b %d"), alt.Tooltip("ndvi:Q", format=".3f", title=gi_), alt.Tooltip("sensor:N")]))
            rules = alt.Chart(pd.DataFrame({"x": [pd.Timestamp(R["planting"]), anchor], "l": ["planting", "my green-up"]})).mark_rule(strokeDash=[5, 4], strokeWidth=2).encode(
                x="x:T", color=alt.Color("l:N", scale=alt.Scale(domain=["planting", "my green-up"], range=[INK, BLUE]), legend=alt.Legend(title=None, orient="top")))
            layers = band + ln + mine + rules
            if ranked_line:
                rk_ = pd.DataFrame({"x": [pd.Timestamp(IMG_DATE)], "y": [0.98],
                                    "t": [f"ranked · {IMG_DATE:%b %d} · #{int(mr['rank'])} of {N_RANKED}"]})
                layers = (layers + alt.Chart(rk_).mark_rule(color="#ea580c", strokeWidth=3).encode(x="x:T")
                          + alt.Chart(rk_).mark_text(align="right", dx=-6, dy=4, fontSize=11, fontWeight="bold", color="#ea580c").encode(x="x:T", y="y:Q", text="t:N"))
            st.altair_chart(layers.resolve_scale(color="independent").properties(height=360).configure_view(strokeWidth=0), use_container_width=True)
        with rcol:
            html(f'<div class="ctitle">When neighbours were planted (estimated)</div><div class="csub">blue line = my planting date · shaded = ± {window} days · '
                 'each field worked back from its green-up</div>')
            gd = nb[pd.to_datetime(nb["est_planting"], errors="coerce").notna() & nb["plant_conf"].notna()].copy()
            gd["d"] = pd.to_datetime(gd["est_planting"]).dt.normalize()
            gd["group"] = np.where(gd["category"].eq("comparable"), "comparable", np.where(gd["plant_conf"].eq("high"), "well dated · not matched", "roughly dated · not matched"))
            dom_ = ["comparable", "well dated · not matched", "roughly dated · not matched"]
            gd = gd.groupby(["d", "group"]).size().rename("fields").reset_index()
            mp_ = pd.Timestamp(R["planting"])
            bnd = alt.Chart(pd.DataFrame({"s": [mp_ - pd.Timedelta(days=window)], "e": [mp_ + pd.Timedelta(days=window)]})).mark_rect(opacity=0.12, color="#16a34a").encode(x="s:T", x2="e:T")
            hist = alt.Chart(gd).mark_bar(cornerRadiusTopLeft=3, cornerRadiusTopRight=3, size=7).encode(
                x=alt.X("d:T", title=None, axis=alt.Axis(format="%b %d")), y=alt.Y("fields:Q", stack=True, title="fields"),
                tooltip=[alt.Tooltip("d:T", format="%b %d", title="est. planting"), alt.Tooltip("group:N", title=""), "fields:Q"],
                color=alt.Color("group:N", scale=alt.Scale(domain=dom_, range=["#16a34a", "#94a3b8", "#fcd34d"]),
                                legend=alt.Legend(title=None, orient="top", columns=1, labelFontSize=11, symbolType="square")))
            rule_ = alt.Chart(pd.DataFrame({"x": [mp_]})).mark_rule(color=BLUE, strokeWidth=2.5).encode(x="x:T")
            st.altair_chart((bnd + hist + rule_).properties(height=320).configure_view(strokeWidth=0), use_container_width=True)
            sat_ = PINFO.get("my_sat_planting")
            html(f'<div class="note">{n_hi_all} neighbours well dated · {n_lo_all} roughly dated'
                 + (f' · satellite alone puts my planting at <b>{sat_:%b %d}</b> ± {PINFO.get("my_sat_unc", 0):.0f} d (entered: {R["planting"]:%b %d})' if sat_ else "")
                 + '. Choose <b>+ roughly dated</b> in the sidebar to include the blurred ones.</div>')

# ---------------------------------------------------------------- Data
with tab_d, card():
    D = (ranked if ranked is not None else T).copy()
    D["status"] = D["field_id"].map(T.set_index("field_id")["category"]).map({**CAT_LABEL, "me": "my field"})
    D["reference"] = D["is_ref_field"] if "is_ref_field" in D else False
    if "reason" in D:
        D["note"] = np.where(D["status"].eq("Comparable") & D["rank"].isna(), "no clear image near the check date", "")
    for c_ in ("greenup_date", "onset_ndvi", "onset_evi2", "plant_conf", "plant_unc", "plant_why", "lag_days"):
        if c_ not in D and c_ in T:
            D[c_] = D["field_id"].map(T.set_index("field_id")[c_])
    cols = [c for c in ["rank", "short", "where", "status", "note", "reference", "obs_date", "vi_raw", "calibration", "vi_median", "SI",
                        "cons_rank", "cons_score"] + [f"rank_{v}" for v in G.ALL_VI] + [
                        "est_planting", "plant_unc", "plant_conf", "plant_why", "delta_days", "delta_match", "greenup_date", "onset_ndvi", "onset_evi2", "lag_days",
                        "lat", "lon", "acres", "elev_median", "texture", "irrigated", "field_id"] if c in D]
    D = D[cols].sort_values("rank" if "rank" in cols else "status", na_position="last")
    for c_ in ("obs_date", "est_planting", "greenup_date", "onset_ndvi", "onset_evi2"):
        if c_ in D:
            D[c_] = pd.to_datetime(D[c_], errors="coerce").dt.strftime("%Y-%m-%d").fillna("")
    if "plant_conf" in D:
        D["plant_conf"] = D["plant_conf"].map({"high": "well dated", "low": "roughly dated"})
    D = D.rename(columns={"cons_rank": "◎ consensus rank", "cons_score": "◎ score /100", **{f"rank_{v}": f"rank {v}" for v in G.ALL_VI},
                          "short": "field", "field_id": "CSB id", "delta_days": "planted Δ days vs mine", "delta_match": "planted Δ heat-unit days (used to match)", "plant_unc": "planting ± days",
                          "plant_conf": "planting date", "plant_why": "why roughly dated", "greenup_date": "green-up",
                          "onset_ndvi": "green-up (NDVI)", "onset_evi2": "green-up (EVI2)", "lag_days": "planting → green-up days", "elev_median": "elevation m", "obs_date": "image date",
                          "vi_raw": f"{VI} (raw)", "calibration": "image calibration",
                          "vi_median": f"{VI} (on check image scale)" if ranked is not None else "vi"})
    if ranked is not None:
        st.caption(f"Each field's value is read from its clear image nearest {IMG_DATE:%b %d}; values from other days are put on the scale of the "
                   f"{IMG_DATE:%b %d} image with the fields clear on both days (image calibration), then SI = value ÷ the one reference "
                   f"(q{int(q * 100)} of the other comparable fields). Rank, map, scout answer and trend all use these numbers.")
    st.dataframe(D.reset_index(drop=True), use_container_width=True, height=480, hide_index=True,
                 column_config={"lat": st.column_config.NumberColumn(format="%.5f"), "lon": st.column_config.NumberColumn(format="%.5f"),
                                "acres": st.column_config.NumberColumn(format="%.0f"), "SI": st.column_config.NumberColumn(format="%.2f"),
                                "elevation m": st.column_config.NumberColumn(format="%.0f"), "rank": st.column_config.NumberColumn(format="%d")})
    st.download_button("Download CSV", D.to_csv(index=False).encode(), "fieldrank.csv", "text/csv")

footer()
