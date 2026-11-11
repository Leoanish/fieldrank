# Putting FieldRank online (free, Georgia)

FieldRank runs on **Streamlit Community Cloud** straight from a GitHub repository. It is free, gives you a link like
`https://fieldrank.streamlit.app`, and redeploys by itself every time you publish an update.

What the hosted app ships with (≈ 24 MB, every file far below GitHub's 100 MB limit):

| | |
|---|---|
| `data/fields/tiles/` | all 78,985 Georgia CSB fields in 216 small tiles + an index — the app reads only the tiles around the chosen field |
| `data/ancillary/lanid/` | Georgia irrigation layers (LANID) |
| `data/ancillary/field_profile.parquet` | elevation / soil / irrigation per field (pre-computed; the more fields it covers, the faster searches are) |
| `data/timeseries/` | imagery cache from your local runs (examples become instant) |

Satellite images are always read live from Earth Search (AWS) and Planetary Computer — no keys or secrets needed.

## 1 · Prepare (on your Mac, in the project folder)

```bash
bash scripts/fieldrank.sh tiles          # already done for you — splits the field table into tiles
bash scripts/fieldrank.sh warm           # ~5 min — runs the 4 example fields once so visitors get them instantly
bash scripts/fieldrank.sh profiles-ga    # optional, long (≈1–2 h, resumable) — profiles for every GA field
```

`profiles-ga` is the biggest speed-up for real users: without it, the server works out elevation, soil and irrigation
for the fields of each new search area (≈ 20–60 s extra the first time an area is searched).

## 2 · Publish to GitHub

```bash
bash scripts/fieldrank.sh publish
```

This copies only what the app needs into `~/FieldRank_repo` (outside OneDrive — git and cloud-sync folders don't mix),
commits, and pushes to a **private** GitHub repo called `fieldrank`. It uses the GitHub CLI if you have it
(`brew install gh` then `gh auth login` once); otherwise it prints the two commands to run after creating an empty repo
at https://github.com/new. Run the same command again whenever you want to publish an update.

## 3 · Deploy on Streamlit Community Cloud (once)

1. Go to https://share.streamlit.io and sign in with GitHub (allow access to private repos when asked).
2. **Create app → Deploy a public app from GitHub** (wording varies) and pick:
   - Repository: `<you>/fieldrank` · Branch: `main` · Main file path: `app/dashboard.py`
   - App URL: e.g. `fieldrank`
   - **Advanced settings → Python 3.12**
3. **Deploy.** The first build takes ~5–10 minutes (geospatial packages); later updates take ~1 minute.

## 4 · Share it

Open the app → **Share** (top right):

- **Anyone with the link:** set *“This app is public and searchable”*, then email the link to whoever you like.
- **Only people you choose:** keep it private and **invite by email** — each person gets an email with a sign-in link
  (no password; they can use any email address). The free plan allows **one** private app (unlimited public ones).

The code can stay in a private GitHub repo either way.

## Good to know

- **Sleeps after 12 h without visitors**; the next visitor sees a “wake up” button (~30 s).
- **Memory ≈ 1 GB (can burst higher).** On the server FieldRank automatically reads 2 images at a time instead of 4
  and caps the search radius at 20 km, so several people can use it at once.
- **Georgia only** for now: field tiles cover GA; the irrigation layer and pre-computed profiles are GA.
  Going US-wide means statewide tiles for each state (same `tiles` step) and moving the bigger files to cloud storage.
- **Crop labels are 2025** (latest CSB). For the 2026 season a field's crop may have rotated — this is why the examples
  use the 2025 season.
- **Logs / restart:** in the app, *Manage app* (bottom right) → logs, reboot.
