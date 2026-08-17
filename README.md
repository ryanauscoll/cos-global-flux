# COS Global Flux

Code and figures behind my COS Global Flux technical report - a cross-instrument
flux calibration comparison between HST/COS and STIS, FOS, GHRS, plus the legacy
missions FUSE and IUE. Also includes a COS self-consistency check using targets
with three or more repeat visits.

## Layout

- `src/pipeline/` - scripts that hit MAST directly (via astroquery), download
  the FITS files, and compute flux ratios / scatter stats. Slow on a cold
  cache since it's pulling real archive data.
- `src/plotting/` - scripts that just replot from the cached CSVs/npz files
  in `data/`. Fast, no MAST access needed.
- `data/` - the small catalogs and npz caches the plotting scripts read.
- `figures/` - the actual PNGs, one per figure in the report.

## Which script made which figure

- **Figures 1, 2, 3, 6** (3C273 / FAIRALL9 / HD271791 / GD153 diagnostic
  panels) - `ratio_comparison_plots_one_target.py`. Change the `TARGET`
  constant near the top and rerun for each target.
- **Figures 4 & 5** (HST vs. legacy instrument flux ratio histograms) -
  `plot_from_csv4_larger.py`.
- **Figure 7** (STIS/FUSE median flux ratio vs. wavelength) -
  `median_difference_as_wavelength_G140L_red_line_fixed_sn_larger.py`
  (`wavelength_comparison_analysis.py` is the general N-instrument version
  of the same idea).
- **Figures 8 & 9** (COS self-consistency scatter + histogram) -
  `plot_cos_vs_cos_from_npz.py`, reading `data/cache/cos_vs_cos_plot_arrays.npz`.

There's also `ratio_comparison_plots_one_target_one_panel.py`, a stripped-down
version of the Figure 1/2/3/6 script that only plots the flux spectra and
skips the ratio subplots below it - useful if you just want to eyeball
instrument coverage for a target without the full breakdown. Example output
is in `figures/supplementary_single_panel/`.

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Running it

Fast path (Figures 4, 5, 8, 9) - no MAST access needed, just replots from
what's already cached in `data/`:

```bash
cd src/plotting
python plot_from_csv4_larger.py
python plot_cos_vs_cos_from_npz.py
```

Everything else needs the full pipeline. It queries MAST and downloads FITS
files into a local `mastDownload/` (not tracked here - gets big fast), and
caches per-target arrays to `array_cache/*.pkl` so reruns against the same
target are quick:

```bash
cd src/pipeline
python ratio_comparison_plots_one_target.py   # set TARGET first
python median_difference_as_wavelength_G140L_red_line_fixed_sn_larger.py
```

## Notes carried over from the report

- COS G140L is masked above 1900 Å - known second-order contamination.
- The S/N ≥ 5 filter only applies to COS and STIS; FOS and GHRS don't have
  dedicated error columns in their archive file format, so their scatter
  may partly reflect noise-dominated spectra.
- This is a comparison pipeline only - it doesn't re-run `calcos`/`calstis`
  or touch the underlying calibration.
