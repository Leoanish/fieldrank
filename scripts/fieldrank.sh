#!/usr/bin/env bash
# FieldRank helper — finds (or creates) the right conda environment, installs what's missing, runs a step.
#
#   bash scripts/fieldrank.sh lanid        # download + clip the LANID irrigation layer (default)
#   bash scripts/fieldrank.sh soil         # build the soil-property table once (gNATSGO via Planetary Computer)
#   bash scripts/fieldrank.sh profiles     # pre-compute field profiles around Tifton (40 km)
#   bash scripts/fieldrank.sh profiles-ga  # pre-compute profiles for EVERY Georgia field (long; resumable) — fastest hosting
#   bash scripts/fieldrank.sh tiles        # split the field table into small map tiles (needed for hosting)
#   bash scripts/fieldrank.sh warm         # run the 4 example fields once so they are instant for visitors
#   bash scripts/fieldrank.sh publish      # copy the app to ~/FieldRank_repo and push it to GitHub (see DEPLOY.md)
#   bash scripts/fieldrank.sh app          # start the dashboard
#   bash scripts/fieldrank.sh check        # just show which environment is used and what is installed
set -euo pipefail

PROJ="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJ"
STEP="${1:-lanid}"
NEW_ENV="fieldrank"

say() { printf "\n\033[1;32m▶ %s\033[0m\n" "$*"; }
warn() { printf "\033[1;33m! %s\033[0m\n" "$*"; }

# ---------------------------------------------------------------- 1. find conda
if ! command -v conda >/dev/null 2>&1; then
  for c in "$HOME/anaconda3" "$HOME/miniconda3" "$HOME/miniforge3" "$HOME/mambaforge" "$HOME/opt/anaconda3" \
           "$HOME/opt/miniconda3" "/opt/anaconda3" "/opt/miniconda3" "/opt/homebrew/anaconda3" \
           "/opt/homebrew/Caskroom/miniconda/base" "/opt/homebrew/Caskroom/miniforge/base" "/usr/local/anaconda3"; do
    if [ -f "$c/etc/profile.d/conda.sh" ]; then source "$c/etc/profile.d/conda.sh"; break; fi
  done
fi
if ! command -v conda >/dev/null 2>&1; then
  warn "conda was not found. Install Miniforge (https://github.com/conda-forge/miniforge) and run this again."
  exit 1
fi
BASE="$(conda info --base)"
source "$BASE/etc/profile.d/conda.sh"

# ---------------------------------------------------------------- 2. pick the environment
# saved choice from an earlier run
ENV_PATH=""
if [ -f .fieldrank_env ] && [ -x "$(cat .fieldrank_env)/bin/python" ]; then ENV_PATH="$(cat .fieldrank_env)"; fi

if [ -z "$ENV_PATH" ]; then
  say "Looking for the environment that already runs the app (has streamlit + odc-stac)…"
  best=""; ok_streamlit=""
  for d in "$BASE" "$BASE"/envs/*; do
    [ -x "$d/bin/python" ] || continue
    if "$d/bin/python" -c "import streamlit, odc.stac, geopandas" >/dev/null 2>&1; then best="$d"; fi
    if [ -z "$ok_streamlit" ] && "$d/bin/python" -c "import streamlit" >/dev/null 2>&1; then ok_streamlit="$d"; fi
    printf "   %-60s %s\n" "$d" "$("$d/bin/python" -c 'import streamlit,odc.stac,geopandas; print("✓ app packages")' 2>/dev/null || echo "-")"
  done
  ENV_PATH="${best:-$ok_streamlit}"
fi

if [ -z "$ENV_PATH" ]; then
  say "No suitable environment found — creating '$NEW_ENV' (one time, a few minutes)…"
  conda create -y -n "$NEW_ENV" -c conda-forge python=3.11 geopandas rasterio pyogrio shapely pyarrow pip
  ENV_PATH="$BASE/envs/$NEW_ENV"
fi

conda activate "$ENV_PATH"
echo "$ENV_PATH" > .fieldrank_env
say "Using environment: $ENV_PATH"
echo "   python: $(command -v python)  ($(python --version 2>&1))"

# ---------------------------------------------------------------- 3. make sure the packages are there
missing=$(python - <<'EOF'
mods = {"rasterio": "rasterio", "geopandas": "geopandas", "pyarrow": "pyarrow", "requests": "requests",
        "odc.stac": "odc-stac", "pystac_client": "pystac-client", "planetary_computer": "planetary-computer",
        "streamlit": "streamlit", "streamlit_folium": "streamlit-folium", "folium": "folium", "altair": "altair", "scipy": "scipy", "adlfs": "adlfs"}
miss = []
for m, pkg in mods.items():
    try:
        __import__(m)
    except Exception:
        miss.append(pkg)
print(" ".join(miss))
EOF
)
if [ -n "$missing" ]; then
  say "Installing missing packages: $missing"
  case " $missing " in
    *" rasterio "*|*" geopandas "*) conda install -y -c conda-forge rasterio geopandas pyogrio ;;
  esac
  python -m pip install -q -r requirements.txt
else
  echo "   all packages present ✓"
fi

# ---------------------------------------------------------------- 4. run the step
case "$STEP" in
  lanid)
    say "Downloading + clipping LANID irrigation (≈ 950 MB into ~/FieldRank_downloads, Georgia layers into data/ancillary/lanid)…"
    python scripts/02_get_profile_layers.py lanid --delete-zips "${@:2}"
    say "Done. Start the app with:  bash scripts/fieldrank.sh app" ;;
  profiles)
    say "Pre-computing field profiles around Tifton (40 km)…"
    python scripts/02_get_profile_layers.py profiles --around -83.51 31.45 --km 40 "${@:2}" ;;
  soil)
    say "Building the national soil table from gNATSGO on Planetary Computer (one time, 1–3 min)…"
    python scripts/02_get_profile_layers.py soil "${@:2}" ;;
  profiles-ga)
    say "Pre-computing profiles for every Georgia field (resumable — stop with Ctrl-C and run again to continue)…"
    python scripts/02_get_profile_layers.py profiles --all "${@:2}" ;;
  fields)
    say "Rebuilding Georgia fields from the national CSB 2018–2025 (crop per year + double crops) — ~5–15 min…"
    CSB="${2:-data/raw/CSB1825.gdb}"
    python -m si_ranking.cli fields --csb "$CSB" ;;
  tiles)
    say "Splitting data/fields/ga_fields.parquet into map tiles…"
    python -m si_ranking.cli tiles ;;
  warm)
    say "Running the four example fields once (fills the caches that ship with the app)…"
    python scripts/warm_examples.py ;;
  publish)
    [ -f data/fields/tiles/index.parquet ] || python -m si_ranking.cli tiles
    bash scripts/publish_repo.sh "${@:2}" ;;
  app)
    say "Starting the dashboard…"
    # headless=false locally so the browser tab opens (the hosted server keeps headless=true from .streamlit/config.toml)
    python -m streamlit run app/dashboard.py --server.headless false "${@:2}" ;;
  check)
    python -c "import rasterio, geopandas, odc.stac, streamlit; print('rasterio', rasterio.__version__, '| geopandas', geopandas.__version__, '| streamlit', streamlit.__version__)" ;;
  *)
    warn "unknown step '$STEP' — use: fields | lanid | soil | profiles | profiles-ga | tiles | warm | publish | app | check"; exit 1 ;;
esac
