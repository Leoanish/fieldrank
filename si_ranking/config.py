"""
Central configuration for the Georgia SI field-ranking platform.

Everything a researcher is likely to tune lives here: crop codes, size
thresholds, GDD-to-emergence parameters, planting-date priors, VI bands.
All thermal-time numbers are starting points taken from extension-style
rules of thumb; calibrate them against known planting dates (see README).
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

# ----------------------------------------------------------------------------
# Paths
# ----------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
RAW_DIR = DATA_DIR / "raw"          # CSB zip / gdb goes here
FIELDS_DIR = DATA_DIR / "fields"    # filtered GA field polygons (GeoParquet)
TS_DIR = DATA_DIR / "timeseries"    # per-field VI time series (parquet)
WX_DIR = DATA_DIR / "weather"       # daily weather per grid cell (parquet)
ANC_DIR = DATA_DIR / "ancillary"    # elevation / soils / irrigation (parquet)
for _d in (RAW_DIR, FIELDS_DIR, TS_DIR, WX_DIR, ANC_DIR):
    _d.mkdir(parents=True, exist_ok=True)

# Hosted mode (Streamlit Community Cloud checks the repo out under /mount/src, ~1-2.7 GB RAM, shared CPU):
# fewer parallel scene reads, a smaller maximum search radius, no national soil-table build.
import os as _os
HOSTED = ROOT.as_posix().startswith("/mount/src") or _os.environ.get("FIELDRANK_HOSTED") == "1"
WORKERS = 2 if HOSTED else 4
MAX_RADIUS_KM = 20 if HOSTED else 40
# Scale guards — a 10 km circle in the cotton belt can hold 1000+ fields.
MAX_FIELDS = 450 if HOSTED else 1200        # nearest same-crop fields kept for the comparison (plenty for a q95 reference)
MAX_WINDOW_PX = 700_000 if HOSTED else 1_500_000   # largest single raster window read at once (pixels); bigger areas are tiled
CACHE_MAX_MB = 250 if HOSTED else 1500
MAX_CONCURRENT_JOBS = 2 if HOSTED else 4   # imagery jobs running at once on one server; more users wait in line instead of crashing it       # imagery cache on disk is trimmed (oldest first) above this

STATE_FIPS = "13"            # Georgia
STATE_NAME = "GEORGIA"
# Rough Georgia bounding box (lon_min, lat_min, lon_max, lat_max)
GA_BBOX = (-85.65, 30.35, -80.75, 35.05)

# ----------------------------------------------------------------------------
# Crops (CDL codes) and crop-specific parameters
# ----------------------------------------------------------------------------
CDL_CODES = {"corn": 1, "cotton": 2, "soybean": 5, "peanut": 10}
CODE_TO_CROP = {v: k for k, v in CDL_CODES.items()}
CROPS = list(CDL_CODES)


@dataclass
class CropParams:
    name: str
    cdl_code: int
    # Thermal time from planting to emergence, degree-days in °C.
    # (corn ~110 GDD °F base 50 -> ~61 °C-days; cotton ~55 DD60 -> ~31 °C-days)
    gdd_base_c: float
    gdd_to_emergence: float
    gdd_upper_c: float = 30.0          # cap on daily mean temp (°C)
    # Planting-date prior for Georgia, day-of-year mean and sd (Gaussian).
    # Replace with NASS Crop Progress-derived values when available.
    prior_doy_mean: float = 130.0
    prior_doy_sd: float = 15.0
    # Plausible emergence search window (DOY) for the detector
    emerg_doy_min: int = 60
    emerg_doy_max: int = 220
    # Minimum field size (acres) for ranking at 10 m and at 20 m (NDRE)
    min_acres_10m: float = 8.0
    min_acres_20m: float = 20.0


CROP_PARAMS: dict[str, CropParams] = {
    "corn": CropParams("corn", 1, gdd_base_c=10.0, gdd_to_emergence=61.0,
                       prior_doy_mean=85, prior_doy_sd=14,          # ~Mar 26
                       emerg_doy_min=55, emerg_doy_max=150),
    "cotton": CropParams("cotton", 2, gdd_base_c=15.6, gdd_to_emergence=31.0,
                         prior_doy_mean=130, prior_doy_sd=14,       # ~May 10
                         emerg_doy_min=100, emerg_doy_max=185),
    "peanut": CropParams("peanut", 10, gdd_base_c=13.3, gdd_to_emergence=75.0,
                         prior_doy_mean=130, prior_doy_sd=12,       # ~May 10
                         emerg_doy_min=105, emerg_doy_max=180),
    "soybean": CropParams("soybean", 5, gdd_base_c=10.0, gdd_to_emergence=65.0,
                          prior_doy_mean=150, prior_doy_sd=20,      # ~May 30 (wide: double-crop)
                          emerg_doy_min=110, emerg_doy_max=215),
}

# Thermal lag (°C-days, same base as the crop) between true emergence and the
# first satellite-detectable green-up (seedlings are invisible at 10 m).
# Defaults were fitted on synthetic curves; overwrite with calibrate.py using
# known planting dates -> data/calibration.json takes precedence.
DETECT_LAG_GDD_DEFAULT = {"corn": 60.0, "cotton": 45.0, "peanut": 60.0, "soybean": 60.0}
CALIBRATION_FILE = DATA_DIR / "calibration.json"


def detect_lag_gdd(crop: str) -> float:
    import json
    if CALIBRATION_FILE.exists():
        try:
            cal = json.loads(CALIBRATION_FILE.read_text())
            if crop in cal.get("detect_lag_gdd", {}):
                return float(cal["detect_lag_gdd"][crop])
        except Exception:
            pass
    return DETECT_LAG_GDD_DEFAULT[crop]


# ----------------------------------------------------------------------------
# Field geometry handling
# ----------------------------------------------------------------------------
INWARD_BUFFER_M = 30.0       # shave edges: border trees, roads, ditches, mixed pixels (falls back to 20 / 10 m on small or narrow fields)
MIN_PIXELS_10M = 60          # minimum valid pixels for a field-date stat to count
EQUAL_AREA_CRS = "EPSG:5070" # CONUS Albers, for area/buffer math

# ----------------------------------------------------------------------------
# Imagery
# ----------------------------------------------------------------------------
STAC_URL = "https://earth-search.aws.element84.com/v1"
STAC_COLLECTION = "sentinel-2-c1-l2a"   # Collection-1 reprocessed; fallback "sentinel-2-l2a"
MAX_CLOUD_COVER = 60                     # scene-level filter; pixel-level SCL mask does the rest
# Earth Search asset keys -> Sentinel-2 band
BAND_ASSETS = {"blue": "blue", "green": "green", "red": "red",
               "rededge1": "rededge1", "nir": "nir", "nir08": "nir08", "scl": "scl"}
# SCL classes considered clear: 4 vegetation, 5 not-vegetated, 6 water, 7 unclassified
SCL_CLEAR = (4, 5, 6, 7)

VI_DEFS = {
    # name: (numerator bands, denominator bands) -> (a-b)/(a+b)
    "NDVI": ("nir", "red"),
    "GNDVI": ("nir", "green"),
    "NDRE": ("nir08", "rededge1"),   # 20 m native (B8A, B5)
}
VI_NATIVE_RES_M = {"NDVI": 10, "GNDVI": 10, "NDRE": 20}
# What the fast scene engine computes per field and scene (median, p10, p90), all from ONE read of the bands:
#   ("nd", a, b)   normalized difference (a - b) / (a + b)
#   ("evi2", nir, red)   2.5 (NIR - Red) / (NIR + 2.4 Red + 1)  — EVI without the blue band (Jiang et al. 2008)
VI_ENGINE = {
    "NDVI": ("nd", "nir", "red"),
    "GNDVI": ("nd", "nir", "green"),
    "NDRE": ("nd", "nir08", "rededge1"),
    "NDMI": ("nd", "nir08", "swir16"),      # canopy water (NIR vs SWIR 1.6 µm, 20 m)
    "EVI2": ("evi2", "nir", "red"),
}
VI_KIND = "vi5"     # cache name for this set (older 'vi3' caches lack NDMI / EVI2)

# ----------------------------------------------------------------------------
# Emergence detection
# ----------------------------------------------------------------------------
EMERG_VI = "NDVI"
EMERG_DELTA = 0.10            # rise above bare-soil baseline that counts as emergence
EMERG_CONFIRM_PEAK = 0.45     # series must later reach this to be a real crop canopy
BASELINE_QUANTILE = 0.15      # baseline = this quantile of pre-season obs
SMOOTH_LAMBDA = 50.0          # Whittaker smoother strength on daily grid

# ----------------------------------------------------------------------------
# Ranking
# ----------------------------------------------------------------------------
VIRTUAL_REF_QUANTILE = 0.98   # virtual reference = this quantile of comparable fields' VI (the top 2 %)
DEFAULT_BUFFER_KM = 15.0
PLANTING_WINDOW_DAYS = 10     # comparable: |estimated planting - my planting| <= this (days)
MIN_COHORT_SIZE = 8

# Weather: cluster field centroids on this grid (deg) to limit API calls
WX_GRID_DEG = 0.1
