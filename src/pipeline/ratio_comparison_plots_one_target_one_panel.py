import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pickle
from astroquery.mast import Observations
from astroquery.exceptions import ResolverError
from pathlib import Path
from astropy.io import fits
import sys
from scipy.stats import binned_statistic
from astropy.stats import mad_std

CSV_PATH = "instrument_flux_ratios_and_counts4.csv"

# Target to plot - change and re-run
TARGET = "GD153"

ARRAY_CACHE_DIR = Path("array_cache")
ARRAY_CACHE_DIR.mkdir(parents=True, exist_ok=True)
VERBOSE = True

TARGET_POINTS_PER_BIN = 40
MIN_WAVELENGTH   = 900
COS_G140L_CUTOFF = 1900

INSTRUMENT_COLORS = {
    "COS":   ("blue",    0.5, 4),
    "STIS":  ("green",   0.3, 1),
    "FOS":   ("cyan",    0.7, 5),
    "HRS/1": ("orange",  0.7, 2),
    "HRS/2": ("red",     0.7, 3),
    "FUSE":  ("purple",  0.7, 6),
    "IUE":   ("magenta", 0.7, 7),
}

plt.rcParams.update({
    "font.size":        32,
    "axes.titlesize":   44,
    "axes.labelsize":   44,
    "xtick.labelsize":  28,
    "ytick.labelsize":  28,
    "legend.fontsize":  30,
    "figure.titlesize": 40,
    "lines.markersize": 8,
})


# Array cache

def _cache_path(target):
    return ARRAY_CACHE_DIR / f"{target.replace(' ', '_')}.pkl"


def save_array_cache(target, payload):
    try:
        with open(_cache_path(target), "wb") as fh:
            pickle.dump(payload, fh, protocol=pickle.HIGHEST_PROTOCOL)
        if VERBOSE:
            print(f"[CACHE] Saved: {_cache_path(target)}")
    except Exception as e:
        if VERBOSE:
            print(f"[WARN] Could not write cache for {target}: {e}")


def load_array_cache(target):
    p = _cache_path(target)
    if not p.exists():
        return None
    try:
        with open(p, "rb") as fh:
            payload = pickle.load(fh)
        if VERBOSE:
            print(f"[CACHE] Loaded: {p}")
        return payload
    except Exception as e:
        if VERBOSE:
            print(f"[WARN] Cache load failed for {target}, will re-download: {e}")
        return None


# FITS helpers

def target_matches(header_objects, target_name):
    if not header_objects:
        return False
    target_clean = target_name.strip().upper()
    for obj in header_objects:
        if obj and target_clean in obj.strip().upper():
            return True
    return False


def query_products(target, instrument_or_collection, suffixes, is_collection=False):
    try:
        if is_collection:
            obs = Observations.query_criteria(
                target_name=target,
                obs_collection=instrument_or_collection,
                dataproduct_type="spectrum",
            )
        else:
            obs = Observations.query_criteria(
                objectname=target,
                instrument_name=instrument_or_collection,
                dataproduct_type="spectrum",
            )
    except ResolverError:
        if VERBOSE:
            print(f"[WARN] Could not resolve '{target}' to a sky position.")
        return None
    except Exception as e:
        if VERBOSE:
            print(f"[ERROR] {target}: {e}")
        return None

    if len(obs) == 0:
        return None

    prods = Observations.get_product_list(obs)
    prods = prods[prods["productType"] == "SCIENCE"]
    fnames = np.array(prods["productFilename"].astype(str))
    keep = np.zeros_like(fnames, dtype=bool)
    for suf in suffixes:
        keep |= np.char.endswith(fnames, suf)
    sel = prods[keep]
    if VERBOSE:
        info = instrument_or_collection if is_collection else f"instrument {instrument_or_collection}"
        print(f"- {target}, {info}: {len(sel)} matching SCIENCE files")
    return sel


def download_if_needed(table, label):
    if table is None or len(table) == 0:
        return None
    if VERBOSE:
        print(f"downloading {label}...")
    return Observations.download_products(table, mrp_only=False)


def load_ghrs_pair(c0_path, c1_path, target_name):
    try:
        with fits.open(c0_path) as w_fits, fits.open(c1_path) as f_fits:
            w = w_fits[0].data.astype(float)
            f = f_fits[0].data.astype(float)
            object_names = []
            for hdr in [w_fits[0].header, f_fits[0].header]:
                if 'OBJECT' in hdr:
                    object_names.append(hdr['OBJECT'])
                elif 'TARGNAME' in hdr:
                    object_names.append(hdr['TARGNAME'])
            if not target_matches(object_names, target_name):
                return np.array([]), np.array([]), []
            unique_objects = list(set(object_names)) if object_names else []
            if w.ndim == 1:
                wave_good = w >= MIN_WAVELENGTH
                good = (f > 0) & np.isfinite(w) & np.isfinite(f) & wave_good
                if np.any(good):
                    logf = np.log10(f[good])
                    final_good = (logf <= -5) & np.isfinite(logf)
                    if np.any(final_good):
                        return w[good][final_good], logf[final_good], unique_objects
                return np.array([]), np.array([]), unique_objects
            else:
                waves, fluxes = [], []
                for seg in range(w.shape[0]):
                    wl, fl = w[seg], f[seg]
                    wave_good = wl >= MIN_WAVELENGTH
                    g = (fl > 0) & np.isfinite(wl) & np.isfinite(fl) & wave_good
                    if np.any(g):
                        logf = np.log10(fl[g])
                        final_g = np.isfinite(logf)
                        if np.any(final_g):
                            waves.append(wl[g][final_g])
                            fluxes.append(logf[final_g])
                return waves, fluxes, unique_objects
    except Exception as e:
        if VERBOSE:
            print(f"[ERROR] GHRS/FOS pair {c0_path}: {e}")
        return np.array([]), np.array([]), []


def load_x1d(fname, target_name):
    try:
        with fits.open(fname) as hd:
            object_from_header = None
            if 'OBJECT' in hd[0].header:
                object_from_header = hd[0].header['OBJECT']
            elif 'TARGNAME' in hd[0].header:
                object_from_header = hd[0].header['TARGNAME']
            elif len(hd) > 1:
                if 'OBJECT' in hd[1].header:
                    object_from_header = hd[1].header['OBJECT']
                elif 'TARGNAME' in hd[1].header:
                    object_from_header = hd[1].header['TARGNAME']
            if not target_matches([object_from_header], target_name):
                return np.array([]), np.array([]), [], [], []
            w_all, f_all = [], []
            cenwaves, opt_elems, object_names = [], [], []
            cenwave_from_header  = None
            opt_elem_from_header = None
            if 'CENWAVE' in hd[0].header:
                cenwave_from_header = hd[0].header['CENWAVE']
            elif len(hd) > 1 and 'CENWAVE' in hd[1].header:
                cenwave_from_header = hd[1].header['CENWAVE']
            if 'OPT_ELEM' in hd[0].header:
                opt_elem_from_header = hd[0].header['OPT_ELEM']
            elif len(hd) > 1 and 'OPT_ELEM' in hd[1].header:
                opt_elem_from_header = hd[1].header['OPT_ELEM']
            for row in hd[1].data:
                wl, fl = row["WAVELENGTH"], row["FLUX"]
                row_opt_elem = None
                if hasattr(row, 'dtype') and 'OPT_ELEM' in row.dtype.names:
                    row_opt_elem = row['OPT_ELEM']
                elif opt_elem_from_header is not None:
                    row_opt_elem = opt_elem_from_header
                wave_good = wl >= MIN_WAVELENGTH
                if row_opt_elem == 'G140L':
                    wave_good = wave_good & (wl < COS_G140L_CUTOFF)
                g = (fl > 0) & np.isfinite(wl) & np.isfinite(fl) & wave_good
                if np.any(g):
                    logf = np.log10(fl[g])
                    final_g = (logf <= -5) & np.isfinite(logf)
                    if np.any(final_g):
                        w_all.append(wl[g][final_g])
                        f_all.append(logf[final_g])
                        row_cenwave = None
                        if hasattr(row, 'dtype') and 'CENWAVE' in row.dtype.names:
                            row_cenwave = row['CENWAVE']
                        elif cenwave_from_header is not None:
                            row_cenwave = cenwave_from_header
                        if row_cenwave is not None:
                            cenwaves.append(row_cenwave)
                        if row_opt_elem is not None:
                            opt_elems.append(row_opt_elem)
                        if object_from_header:
                            object_names.append(object_from_header)
            if w_all and f_all:
                return np.concatenate(w_all), np.concatenate(f_all), cenwaves, opt_elems, list(set(object_names))
            return np.array([]), np.array([]), [], [], []
    except Exception as e:
        if VERBOSE:
            print(f"[ERROR] X1D {fname}: {e}")
        return np.array([]), np.array([]), [], [], []


def load_collection_spectrum(fname, target_name, collection_type="FUSE", apply_wavelength_filter=True):
    try:
        with fits.open(fname) as hd:
            object_from_header = None
            if 'OBJECT' in hd[0].header:
                object_from_header = hd[0].header['OBJECT']
            elif 'TARGNAME' in hd[0].header:
                object_from_header = hd[0].header['TARGNAME']
            elif len(hd) > 1:
                if 'OBJECT' in hd[1].header:
                    object_from_header = hd[1].header['OBJECT']
                elif 'TARGNAME' in hd[1].header:
                    object_from_header = hd[1].header['TARGNAME']
            if not target_matches([object_from_header], target_name):
                return np.array([]), np.array([]), []
            data = None
            if len(hd) > 1 and hd[1].data is not None:
                data = hd[1].data
            elif hd[0].data is not None:
                data = hd[0].data
            if data is None:
                return np.array([]), np.array([]), []
            col_names = [col.name.upper() for col in data.columns] if hasattr(data, 'columns') else []
            wave_col = next((n for n in ['WAVELENGTH', 'WAVE', 'LAMBDA', 'WL'] if n in col_names), None)
            flux_col = next((n for n in ['FLUX', 'FLUX_TTAG', 'FLUX_HIST'] if n in col_names), None)
            if wave_col is None or flux_col is None:
                if VERBOSE:
                    print(f"[WARN] No wavelength/flux columns in {fname}. Available: {col_names}")
                return np.array([]), np.array([]), []
            wl = data[wave_col]
            fl = data[flux_col]
            wave_good = (wl >= MIN_WAVELENGTH) if apply_wavelength_filter else np.ones_like(wl, dtype=bool)
            g = (fl > 0) & np.isfinite(wl) & np.isfinite(fl) & wave_good
            if np.any(g):
                logf = np.log10(fl[g])
                final_g = (logf <= -5) & np.isfinite(logf)
                if np.any(final_g):
                    return wl[g][final_g], logf[final_g], ([object_from_header] if object_from_header else [])
            return np.array([]), np.array([]), []
    except Exception as e:
        if VERBOSE:
            print(f"[ERROR] {collection_type} {fname}: {e}")
        return np.array([]), np.array([]), []


# Binning

def compute_wavelength_bins(wavelengths, target_points_per_bin):
    if len(wavelengths) == 0:
        return np.array([])
    min_wl, max_wl = np.min(wavelengths), np.max(wavelengths)
    if min_wl >= max_wl:
        return np.array([])
    bin_width = (max_wl - min_wl) / (len(wavelengths) / target_points_per_bin)
    return np.arange(min_wl, max_wl + bin_width, bin_width)


def avg_bin_size(bin_edges):
    if len(bin_edges) < 2:
        return np.nan
    return float(np.mean(np.diff(bin_edges)))


def bin_fluxes(w, f, bins):
    if len(bins) < 2:
        return np.array([])
    mask = np.isfinite(w) & np.isfinite(f)
    if not np.any(mask):
        return np.array([])
    if np.any(np.diff(bins) <= 0):
        return np.array([])
    means, _, _ = binned_statistic(w[mask], f[mask], bins=bins, statistic='mean')
    return means


# Main plot

def plot_target(target):
    print(f"\n=== {target} ===")
    plots_dir = Path("ratio_comparison_plots_one_target_one_panel")
    plots_dir.mkdir(parents=True, exist_ok=True)
    out_path = plots_dir / f"{target.replace(' ', '_')}.png"
    if out_path.exists():
        if VERBOSE:
            print(f"[SKIPPED] Already exists: {out_path}")
        return

    cached = load_array_cache(target)
    if cached is not None:
        (cos_wave, cos_flux, cos_cenwaves, cos_opt_elems,
         stis_wave, stis_flux, stis_cenwaves, stis_opt_elems,
         fos_wave, fos_flux,
         hrs1_wave, hrs1_flux,
         hrs2_wave, hrs2_flux,
         fuse_wave, fuse_flux,
         iue_wave,  iue_flux,
         fos_objects, hrs1_objects, hrs2_objects,
         fuse_objects, iue_objects) = cached
    else:
        cos_tab  = query_products(target, "COS", ["_x1d.fits"])
        stis_tab = query_products(target, ["STIS", "STIS/FUV-MAMA", "STIS/NUV-MAMA", "STIS/CCD"], ["_x1d.fits"])
        fos_tab  = query_products(target, ["FOS/BL", "FOS/RD"], ["_c0f.fits", "_c1f.fits"])
        hrs1_tab = query_products(target, "HRS/1", ["_c0f.fits", "_c1f.fits"])
        hrs2_tab = query_products(target, "HRS/2", ["_c0f.fits", "_c1f.fits"])
        fuse_tab = query_products(target, "FUSE", ["_vo.fits", "_all.fits"], is_collection=True)
        iue_tab  = query_products(target, "IUE",  ["_vo.fits", "_all.fits"], is_collection=True)

        cos_dl  = download_if_needed(cos_tab,  "COS")
        stis_dl = download_if_needed(stis_tab, "STIS")
        fos_dl  = download_if_needed(fos_tab,  "FOS")
        hrs1_dl = download_if_needed(hrs1_tab, "HRS/1")
        hrs2_dl = download_if_needed(hrs2_tab, "HRS/2")
        fuse_dl = download_if_needed(fuse_tab, "FUSE")
        iue_dl  = download_if_needed(iue_tab,  "IUE")

        def extract_pairs(tab, dl):
            pairs = {}
            if dl is not None and tab is not None:
                for obsid, path in zip(list(tab["obs_id"]), list(dl["Local Path"])):
                    p = Path(path)
                    root = p.stem[:-4]
                    pairs.setdefault(root, [None, None])
                    if p.stem.endswith("_c0f"):
                        pairs[root][0] = path
                    elif p.stem.endswith("_c1f"):
                        pairs[root][1] = path
            return pairs

        ghrs1 = extract_pairs(hrs1_tab, hrs1_dl)
        ghrs2 = extract_pairs(hrs2_tab, hrs2_dl)
        fos   = extract_pairs(fos_tab,  fos_dl)

        def load_all_pairs(pairs, target_name):
            wave, flux, all_objects = [], [], []
            for root, (c0, c1) in pairs.items():
                if c0 and c1 and Path(c0).exists() and Path(c1).exists():
                    out = load_ghrs_pair(c0, c1, target_name)
                    if len(out) == 3:
                        if isinstance(out[0], list):
                            wave.extend(out[0]); flux.extend(out[1]); all_objects.extend(out[2])
                        else:
                            wave.append(out[0]); flux.append(out[1]); all_objects.extend(out[2])
            return wave, flux, list(set(all_objects))

        def load_all_files(files, target_name, file_type="X1D", apply_wavelength_filter=True):
            wave, flux, all_cenwaves, all_opt_elems, all_objects = [], [], [], [], []
            for f in files:
                if Path(f).exists():
                    if file_type == "X1D":
                        result = load_x1d(f, target_name)
                        if len(result) == 5:
                            w, fl, cenwaves, opt_elems, objects = result
                            if len(w) > 0:
                                wave.append(w); flux.append(fl)
                                all_cenwaves.extend(cenwaves); all_opt_elems.extend(opt_elems)
                                all_objects.extend(objects)
                    elif file_type in ["FUSE", "IUE"]:
                        result = load_collection_spectrum(f, target_name, file_type, apply_wavelength_filter)
                        if len(result) == 3:
                            w, fl, objects = result
                            if len(w) > 0:
                                wave.append(w); flux.append(fl); all_objects.extend(objects)
            return wave, flux, all_cenwaves, all_opt_elems, list(set(all_objects))

        cos_files  = [] if cos_dl  is None else list(cos_dl["Local Path"])
        stis_files = [] if stis_dl is None else list(stis_dl["Local Path"])
        fuse_files = [] if fuse_dl is None else list(fuse_dl["Local Path"])
        iue_files  = [] if iue_dl  is None else list(iue_dl["Local Path"])

        cos_wave,  cos_flux,  cos_cenwaves,  cos_opt_elems,  _ = load_all_files(cos_files,  target, "X1D")
        stis_wave, stis_flux, stis_cenwaves, stis_opt_elems, _ = load_all_files(stis_files, target, "X1D")
        fos_wave,  fos_flux,  fos_objects                      = load_all_pairs(fos,   target)
        hrs1_wave, hrs1_flux, hrs1_objects                     = load_all_pairs(ghrs1, target)
        hrs2_wave, hrs2_flux, hrs2_objects                     = load_all_pairs(ghrs2, target)
        fuse_wave, fuse_flux, _, _, fuse_objects = load_all_files(fuse_files, target, "FUSE", apply_wavelength_filter=False)
        iue_wave,  iue_flux,  _, _, iue_objects  = load_all_files(iue_files,  target, "IUE")

        save_array_cache(target, (
            cos_wave,  cos_flux,  cos_cenwaves,  cos_opt_elems,
            stis_wave, stis_flux, stis_cenwaves, stis_opt_elems,
            fos_wave,  fos_flux,
            hrs1_wave, hrs1_flux,
            hrs2_wave, hrs2_flux,
            fuse_wave, fuse_flux,
            iue_wave,  iue_flux,
            fos_objects, hrs1_objects, hrs2_objects,
            fuse_objects, iue_objects,
        ))

    if not cos_wave or not cos_flux:
        print("No COS spectrum found - skipping")
        return

    cos_all      = np.concatenate(cos_wave)
    cos_all_flux = np.concatenate(cos_flux)

    cos_display_mask = cos_all >= MIN_WAVELENGTH
    cos_display_wl   = cos_all[cos_display_mask]
    cos_display_flux = cos_all_flux[cos_display_mask]

    cos_bins        = compute_wavelength_bins(cos_display_wl, TARGET_POINTS_PER_BIN)
    binned_cos_disp = bin_fluxes(cos_display_wl, cos_display_flux, cos_bins)
    bin_centers_cos = (cos_bins[:-1] + cos_bins[1:]) / 2
    x_min = max(bin_centers_cos[0], MIN_WAVELENGTH)
    x_max = bin_centers_cos[-1]

    has_g140l = (any(oe == 'G140L' for oe in cos_opt_elems) or
                 any(oe == 'G140L' for oe in stis_opt_elems))

    spectra_data = {
        "STIS":  (stis_wave, stis_flux),
        "FOS":   (fos_wave,  fos_flux),
        "HRS/1": (hrs1_wave, hrs1_flux),
        "HRS/2": (hrs2_wave, hrs2_flux),
        "FUSE":  (fuse_wave, fuse_flux),
        "IUE":   (iue_wave,  iue_flux),
    }

    valid_spectra = {}
    for name, (w_list, f_list) in spectra_data.items():
        if not w_list or not f_list:
            continue
        try:
            w = np.concatenate(w_list)
            if len(w) == 0:
                continue
            min_wl_override = 0 if name == "FUSE" else MIN_WAVELENGTH
            overlap_min = max(min(w), min(cos_all), min_wl_override)
            overlap_max = min(max(w), max(cos_all))
            if overlap_max > overlap_min:
                valid_spectra[name] = (w_list, f_list)
        except ValueError:
            continue

    # NOTE: ratio sub-panels removed — we no longer require overlapping
    # instrument data to proceed, since there's nothing to compute a ratio
    # against anymore. We just plot whatever overlay spectra exist (if any).

    fig, ax = plt.subplots(figsize=(22, 12))

    c, a, z = INSTRUMENT_COLORS["COS"]
    ax.plot(bin_centers_cos, binned_cos_disp, marker='o', markersize=8,
            label="COS", color=c, alpha=a, zorder=z)

    for name, (w_list, f_list) in valid_spectra.items():
        w = np.concatenate(w_list)
        f = np.concatenate(f_list)
        min_wl_override = 0 if name == "FUSE" else MIN_WAVELENGTH
        ov_min = max(min(w), min(cos_all), min_wl_override)
        ov_max = min(max(w), max(cos_all))
        mask = (w >= ov_min) & (w <= ov_max)
        w_ov, f_ov = w[mask], f[mask]
        if len(w_ov) == 0:
            continue
        bins = compute_wavelength_bins(w_ov, TARGET_POINTS_PER_BIN)
        if len(bins) < 2:
            continue
        binned = bin_fluxes(w_ov, f_ov, bins)
        if len(binned) == 0:
            continue
        centers = (bins[:-1] + bins[1:]) / 2
        c, a, z = INSTRUMENT_COLORS.get(name, ("black", 0.7, 1))
        ax.plot(centers, binned, marker='o', markersize=8,
                label=name, color=c, alpha=a, zorder=z)

    ax.set_xlim(x_min, x_max)
    ax.set_xlabel("Wavelength (Å)")
    ax.set_ylabel("log₁₀(Flux)")
    title_suffix = f"  (G140L limited to <{COS_G140L_CUTOFF} Å)" if has_g140l else ""
    ax.set_title(f"{target} {title_suffix}")
    ax.legend(ncol=2)
    ax.grid(True)

    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()
    if VERBOSE:
        print(f"[SAVED] {out_path}")


def main():
    print(f"Target:      {TARGET}")
    print(f"Array cache: {ARRAY_CACHE_DIR.resolve()}")
    print()
    plot_target(TARGET)
    print("\nDone.")


if __name__ == "__main__":
    main()