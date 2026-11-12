# FieldRank — how is my field doing next to similar fields? (Georgia)

FieldRank compares a corn, cotton, peanut or soybean field in Georgia with nearby fields of the same crop that were
planted at about the same time, using free Sentinel-2 (and Landsat) satellite images. It answers one question in plain
words — **“Should I scout my field?”** — and shows the rank, a map and the evidence behind it.

**Live app: [fieldrank.streamlit.app](https://fieldrank.streamlit.app)** — free, no sign-in.

## How it works

1. **Pick a field** — click it on the map, upload a boundary (shapefile, KML/KMZ, GeoJSON, GeoPackage, CSV) or load one
   of the example fields. Field boundaries come from the USDA Crop Sequence Boundaries (108,186 Georgia fields that grew one of the four crops in any year 2018–2025).
2. **Find comparable fields** — same crop, within the search radius, planted within ± 10 days of your date. Each
   neighbour's green-up is found by fitting a smooth growth curve to its NDVI and EVI2, its planting date is worked back
   from that with local weather (degree-days) calibrated on your own field, and it is marked **well dated** (both indices
   agree, clear images around it — good to about a week) or **roughly dated** (blurred by clouds or a weak signal). Optional filters: same irrigation, similar elevation, similar soil texture.
3. **Rank my field** on any date — every comparable field is read on its clear image nearest that date, images of
   different days are calibrated to each other, and

   **SI (sufficiency index) = my field ÷ the reference**, where the reference is the 98th percentile of the comparable
   fields (never including mine). Each field's value is the **median** of its clear 10 m pixels at least 30 m inside its
   boundary, so edge trees, roads and ditches don't count.
4. **Which index** — the index most tied to yield for that crop and growth stage (★ Best: e.g. red-edge CIre for corn
   at tasseling and cotton at bloom, EVI2 for soybean at pod fill, WDRVI for peanut at pod set), plus a **◎ consensus**
   over all 8 indices (NDVI, GNDVI, NDRE, NDMI, EVI2, WDRVI, GCVI, CIre) weighted for the stage, as a second opinion.
5. **The answer** — 🔎 scout / 👀 keep an eye on it / ✅ on track, with the reasons, the trend over the last images,
   the yield-critical stage so far and how confident the call is.

Images are read live from Earth Search (AWS) and Microsoft Planetary Computer; nothing needs a key.

## Run it yourself

```bash
pip install -r requirements.txt          # Python 3.11–3.12
streamlit run app/dashboard.py
```

`bash scripts/fieldrank.sh` has helpers (`app`, `warm`, `publish`, `profiles-ga`, …). See `DEPLOY.md` for hosting on
Streamlit Community Cloud.

## Good to know

- **Georgia only** for now. **Seasons 2024, 2025 and 2026.** 2024 and 2025 compare against the fields that grew your crop *that* year
  (USDA crop labels, double crops such as soybean after wheat included); 2026 uses the 2025 labels until USDA publishes 2026, so some
  fields will have rotated.
- **Greenness is not yield.** Satellites show *that* a field is behind its neighbours, not *why*; the field visit tells
  you the cause. During the yield-critical stage the gap is the best early warning there is.
- Planting-date estimates are model-based (± about a week for well-dated fields, 1–2 weeks for roughly dated ones).
- Every field of your crop inside the search circle is compared (default 7 km; up to 20 km online). When several
  people search at once on the free hosting, the extra searches wait their turn.
- Decision support only — not a yield estimate and not agronomic advice.

## Data sources

USDA NASS Crop Sequence Boundaries · ESA Sentinel-2 L2A (Earth Search) · USGS Landsat 8/9 C2 L2 (Planetary Computer) ·
Copernicus DEM GLO-30 · USDA SSURGO/gNATSGO soils · LANID irrigation layers · Open-Meteo (ERA5-Land) daily weather.
