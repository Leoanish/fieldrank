#!/usr/bin/env bash
# Build a clean copy of FieldRank for GitHub (outside OneDrive — git and cloud-sync folders don't mix) and push it.
#
#   bash scripts/publish_repo.sh [destination] [github-repo-name]
#   default: ~/FieldRank_repo  →  github.com/<you>/fieldrank  (PRIVATE repo; the app's visibility is set on Streamlit)
set -euo pipefail
PROJ="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DEST="${1:-$HOME/FieldRank_repo}"
NAME="${2:-fieldrank}"
say() { printf "\n\033[1;32m▶ %s\033[0m\n" "$*"; }
warn() { printf "\033[1;33m! %s\033[0m\n" "$*"; }

[ -f "$PROJ/data/fields/tiles/index.parquet" ] || { warn "No field tiles yet — run:  bash scripts/fieldrank.sh tiles"; exit 1; }

say "Copying the app into $DEST"
mkdir -p "$DEST"
RS=(-a --delete --exclude ".git" --exclude "__pycache__" --exclude ".DS_Store" --exclude ".pytest_cache")
rsync "${RS[@]}" "$PROJ/si_ranking/" "$DEST/si_ranking/"
mkdir -p "$DEST/app" "$DEST/scripts" "$DEST/data/fields" "$DEST/data/ancillary"
rsync -a "$PROJ/app/dashboard.py" "$PROJ/app/crop_icons.py" "$DEST/app/"
rsync -a "$PROJ/scripts/fieldrank.sh" "$PROJ/scripts/02_get_profile_layers.py" "$PROJ/scripts/warm_examples.py" "$PROJ/scripts/publish_repo.sh" "$DEST/scripts/"
rsync "${RS[@]}" "$PROJ/tests/" "$DEST/tests/"
rsync "${RS[@]}" "$PROJ/.streamlit/" "$DEST/.streamlit/"
for f in requirements.txt requirements-dev.txt README.md DEPLOY.md .gitignore; do [ -f "$PROJ/$f" ] && cp "$PROJ/$f" "$DEST/"; done
rsync "${RS[@]}" "$PROJ/data/fields/tiles/" "$DEST/data/fields/tiles/"
if ls "$PROJ"/data/ancillary/lanid/*_GA.tif >/dev/null 2>&1; then
  mkdir -p "$DEST/data/ancillary/lanid"; rsync -a --delete "$PROJ"/data/ancillary/lanid/*_GA.tif "$DEST/data/ancillary/lanid/"
fi
[ -f "$PROJ/data/ancillary/field_profile.parquet" ] && cp "$PROJ/data/ancillary/field_profile.parquet" "$DEST/data/ancillary/"
TS_MB=$(du -sm "$PROJ/data/timeseries" 2>/dev/null | cut -f1 || echo 0)
if [ "${TS_MB:-0}" -le 150 ]; then
  # only the current engine's scene cache (older versions are retired by the app anyway)
  ENG=$(grep -o 'ENGINE_VERSION = "[a-z0-9]*"' "$PROJ/si_ranking/scenes.py" | cut -d'"' -f2)
  rsync "${RS[@]}" --exclude "scenes/*" --include "*" "$PROJ/data/timeseries/" "$DEST/data/timeseries/"
  if ls -d "$PROJ/data/timeseries/scenes/${ENG}_"* >/dev/null 2>&1; then
    mkdir -p "$DEST/data/timeseries/scenes"
    for d in "$PROJ/data/timeseries/scenes/${ENG}_"*; do rsync "${RS[@]}" "$d/" "$DEST/data/timeseries/scenes/$(basename "$d")/"; done
  fi
  find "$DEST/data/timeseries/scenes" -mindepth 1 -maxdepth 1 -type d ! -name "${ENG}_*" -exec rm -rf {} + 2>/dev/null || true
  echo "   imagery cache included (${TS_MB} MB, engine ${ENG})"
else
  warn "imagery cache is ${TS_MB} MB — not included (the server rebuilds it)"
fi

big=$(find "$DEST" -path "$DEST/.git" -prune -o -type f -size +95M -print)
[ -z "$big" ] || { warn "Files over GitHub's 100 MB limit: $big"; exit 1; }
echo "   repository size: $(du -sh --exclude=.git "$DEST" 2>/dev/null | cut -f1 || du -sh "$DEST" | cut -f1)"

say "Committing"
cd "$DEST"
[ -d .git ] || git init -q -b main
git add -A
if git diff --cached --quiet; then echo "   nothing changed since the last publish"; else git commit -q -m "FieldRank update $(date '+%Y-%m-%d %H:%M')"; echo "   committed"; fi

say "Pushing to GitHub"
if git remote get-url origin >/dev/null 2>&1; then
  git push -u origin main
elif command -v gh >/dev/null 2>&1 && gh auth status >/dev/null 2>&1; then
  gh repo create "$NAME" --private --source . --remote origin --push
else
  warn "GitHub CLI not set up. Either install it (brew install gh && gh auth login) and run this again, or:"
  echo "   1. create an EMPTY private repo named '$NAME' at https://github.com/new"
  echo "   2. cd \"$DEST\" && git remote add origin https://github.com/<your-user>/$NAME.git && git push -u origin main"
  exit 0
fi
say "Done. Next: deploy it at https://share.streamlit.io  (see DEPLOY.md — main file: app/dashboard.py)"
