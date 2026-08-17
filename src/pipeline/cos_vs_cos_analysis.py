import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from astroquery.mast import Observations
from astroquery.exceptions import ResolverError
from pathlib import Path
from astropy.io import fits
from astropy.time import Time
import sys
from scipy.stats import binned_statistic
from collections import defaultdict
from datetime import datetime

CACHE_DIR = Path(".")
VERBOSE = True
MIN_OBSERVATIONS_PER_TARGET = 3  # Need at least 3 obs per target to measure scatter
BIN_SIZE = 5      # 5 angstroms (display bins)
STD_BIN_SIZE = 1  # 1 angstrom bins for std/scatter calculation
MIN_WAVELENGTH = 950  # Angstroms
MIN_MEDIAN_SN = 5  # Minimum S/N to include spectrum
FLUX_THRESHOLD = 1e-14  # Linear flux threshold (erg/s/cm2/Ang) - only calc scatter above this

# Wavelength regions to exclude (geocoronal lines, airglow, artifacts)
BAD_WAVELENGTH_REGIONS = [
    (1210, 1220),  # Lyman-alpha geocoronal
    (1300, 1310),  # OI airglow
    (1550, 1560),  # CIV often saturated
    (2140, 2160),  # Artifact region
]

# Known variable sources to exclude
VARIABLE_SOURCE_PATTERNS = [
    'NGC-', 'MRK-', 'MRK', 'MARK', 'PKS', 'QSO', 'QUASAR',
    'RX-J', 'RXJ', 'PG', 'TON', 'TONS', 'HE', 'HS',
    'SDSS', '3C', 'IRAS', 'PDS', 'LBQS', 'PMNJ',
    'SN20', 'NOVA', 'V*',
]

def get_all_cos_observations():
    """Query ALL COS observations from MAST"""
    print("Querying ALL COS spectroscopic observations from MAST...")

    try:
        all_obs = Observations.query_criteria(
            instrument_name="COS",
            dataproduct_type="spectrum"
        )

        print(f"Found {len(all_obs)} total COS spectroscopic observations")

        print("Getting product list for all observations...")
        prods = Observations.get_product_list(all_obs)
        prods = prods[prods["productType"] == "SCIENCE"]

        fnames = np.array(prods["productFilename"].astype(str))
        keep = np.char.endswith(fnames, "_x1d.fits")
        x1d_prods = prods[keep]

        print(f"Found {len(x1d_prods)} x1d SCIENCE files across all COS observations")

        return all_obs, x1d_prods

    except Exception as e:
        print(f"Error querying MAST: {e}")
        import traceback
        traceback.print_exc()
        return None, None

def is_variable_source(target_name):
    """Check if target is a known variable source"""
    if not target_name:
        return False
    target_upper = target_name.upper()
    for pattern in VARIABLE_SOURCE_PATTERNS:
        if pattern in target_upper:
            return True
    return False

def calculate_spectrum_sn(wavelength, flux_linear, error=None):
    """Calculate median S/N for a spectrum (flux_linear is in linear space)"""
    if len(wavelength) == 0 or len(flux_linear) == 0:
        return 0.0

    if error is not None and len(error) > 0:
        good = (error > 0) & np.isfinite(error) & np.isfinite(flux_linear) & (flux_linear > 0)
        if np.sum(good) > 10:
            sn = flux_linear[good] / error[good]
            return np.median(sn[np.isfinite(sn)])

    good = (flux_linear > 0) & np.isfinite(flux_linear)
    if np.sum(good) > 10:
        noise = np.sqrt(flux_linear[good])
        noise[noise == 0] = np.median(noise[noise > 0])
        sn = flux_linear[good] / noise
        median_sn = np.median(sn[np.isfinite(sn) & (sn > 0)])
        return median_sn if median_sn > 0 else 0.0

    return 0.0

def load_cos_spectrum(fname):
    """Load COS spectrum - stores flux in LINEAR space for correct scatter calculation"""
    try:
        with fits.open(fname) as hd:
            w_all, f_all, e_all, dq_all = [], [], [], []

            obs_date = None
            cenwave = None
            exptime = None
            expstart_mjd = None
            target_name = None

            for hdu_idx in [0, 1]:
                if hdu_idx < len(hd):
                    header = hd[hdu_idx].header
                    if 'TARGNAME' in header:
                        target_name = str(header['TARGNAME']).strip()
                        break
                    elif 'OBJECT' in header:
                        target_name = str(header['OBJECT']).strip()
                        break

            for hdu_idx in [0, 1]:
                if hdu_idx < len(hd):
                    header = hd[hdu_idx].header
                    if 'EXPSTART' in header and header['EXPSTART'] not in [None, 0, '']:
                        try:
                            expstart_mjd = float(header['EXPSTART'])
                            t = Time(expstart_mjd, format='mjd')
                            obs_date = t.datetime.strftime('%Y-%m-%d')
                            break
                        except (ValueError, TypeError):
                            continue

            if not obs_date:
                for hdu_idx in [0, 1]:
                    if hdu_idx < len(hd):
                        header = hd[hdu_idx].header
                        if 'DATE-OBS' in header and header['DATE-OBS'] not in [None, '', 'None']:
                            raw_date = str(header['DATE-OBS']).strip()
                            obs_date = raw_date.split('T')[0] if 'T' in raw_date else raw_date
                            break

            for hdu_idx in [0, 1]:
                if hdu_idx < len(hd):
                    header = hd[hdu_idx].header
                    for keyword in ['CENWAVE', 'CENTRWV', 'CEN_WAVE']:
                        if keyword in header and header[keyword] not in [None, 0]:
                            cenwave = header[keyword]
                            break
                    if cenwave:
                        break

            for hdu_idx in [0, 1]:
                if hdu_idx < len(hd):
                    header = hd[hdu_idx].header
                    for keyword in ['EXPTIME', 'TEXPTIME', 'EXPOSURE']:
                        if keyword in header and header[keyword] not in [None, 0]:
                            exptime = header[keyword]
                            break
                    if exptime:
                        break

            for row in hd[1].data:
                wl, fl = row["WAVELENGTH"], row["FLUX"]

                try:
                    err = row["ERROR"]
                except (KeyError, ValueError):
                    err = None

                try:
                    dq_wgt = row["DQ_WGT"]
                except (KeyError, ValueError):
                    dq_wgt = None

                good_wave = np.isfinite(wl) & (wl > 0)
                good_flux = np.isfinite(fl) & (fl > 0)

                if dq_wgt is not None:
                    good_dq = dq_wgt > 0
                    combined_good = good_wave & good_flux & good_dq
                    dq_weights = dq_wgt[combined_good] if np.any(combined_good) else np.array([])
                else:
                    combined_good = good_wave & good_flux
                    dq_weights = np.ones(np.sum(combined_good)) if np.any(combined_good) else np.array([])

                if np.any(combined_good):
                    wl_clean = wl[combined_good]
                    fl_clean = fl[combined_good]

                    # Store LINEAR flux - apply sanity filter for physical reasonableness
                    flux_filter = (fl_clean > 0) & (fl_clean < 1e-5) & np.isfinite(fl_clean)

                    if np.any(flux_filter):
                        w_all.append(wl_clean[flux_filter])
                        f_all.append(fl_clean[flux_filter])          # LINEAR flux
                        dq_all.append(dq_weights[flux_filter])

                        if err is not None:
                            err_clean = err[combined_good]
                            e_all.append(err_clean[flux_filter])

            wavelength  = np.concatenate(w_all) if w_all else np.array([])
            flux_linear = np.concatenate(f_all) if f_all else np.array([])
            dq_weights  = np.concatenate(dq_all) if dq_all else np.array([])
            error       = np.concatenate(e_all) if e_all else None

            median_sn = calculate_spectrum_sn(wavelength, flux_linear, error)

            return {
                'wavelength':   wavelength,
                'flux':         flux_linear,   # LINEAR flux
                'error':        error,
                'dq_weights':   dq_weights,
                'obs_date':     obs_date,
                'expstart_mjd': expstart_mjd,
                'cenwave':      cenwave,
                'exptime':      exptime,
                'target_name':  target_name,
                'filename':     fname,
                'median_sn':    median_sn
            }
    except Exception as e:
        if VERBOSE:
            print(f"[ERROR] Failed to load COS spectrum: {fname} – {e}")
        return None

def bin_spectrum(w, f, bin_edges):
    """Bin spectrum data"""
    if len(w) == 0 or len(f) == 0:
        return np.array([])
    mask = np.isfinite(w) & np.isfinite(f)
    if not np.any(mask):
        return np.array([])
    bin_means, _, _ = binned_statistic(w[mask], f[mask], bins=bin_edges, statistic='mean')
    return bin_means

def mask_bad_wavelengths(bin_centers):
    """Create mask for good wavelength regions"""
    mask = np.ones(len(bin_centers), dtype=bool)
    for wmin, wmax in BAD_WAVELENGTH_REGIONS:
        mask &= ~((bin_centers >= wmin) & (bin_centers <= wmax))
    return mask

def calculate_per_target_consistency(target_spectra, wave_bins, bin_centers):
    """
    Calculate consistency for a single target using 1Å bins.
    All scatter calculations done in LINEAR flux space.
    """
    if len(target_spectra) < MIN_OBSERVATIONS_PER_TARGET:
        return None

    high_sn_spectra = [s for s in target_spectra if s['median_sn'] >= MIN_MEDIAN_SN]
    if len(high_sn_spectra) < MIN_OBSERVATIONS_PER_TARGET:
        return None

    # Build 1Å bins over the wavelength range of this target
    all_waves = np.concatenate([s['wavelength'] for s in high_sn_spectra])
    std_wave_bins = np.arange(
        np.floor(all_waves.min()),
        np.ceil(all_waves.max()) + STD_BIN_SIZE,
        STD_BIN_SIZE
    )
    std_bin_centers = (std_wave_bins[:-1] + std_wave_bins[1:]) / 2

    # Bin all spectra at 1Å resolution in LINEAR flux space
    binned_fluxes = []
    for spec in high_sn_spectra:
        binned_flux = bin_spectrum(spec['wavelength'], spec['flux'], std_wave_bins)
        if len(binned_flux) > 0:
            binned_fluxes.append(binned_flux)

    if len(binned_fluxes) < MIN_OBSERVATIONS_PER_TARGET:
        return None

    flux_array = np.array(binned_fluxes)   # shape: (n_obs, n_bins), LINEAR flux

    # Compute mean and std in LINEAR space
    mean_flux = np.nanmean(flux_array, axis=0)
    std_flux  = np.nanstd(flux_array, axis=0)

    # Relative scatter: only where mean flux exceeds linear threshold
    rel_scatter = np.full_like(mean_flux, np.nan)
    significant_flux = mean_flux > FLUX_THRESHOLD
    with np.errstate(divide='ignore', invalid='ignore'):
        rel_scatter[significant_flux] = (
            std_flux[significant_flux] / np.abs(mean_flux[significant_flux])
        )

    # Apply wavelength and bad-region masks
    wave_mask      = std_bin_centers >= MIN_WAVELENGTH
    good_wave_mask = mask_bad_wavelengths(std_bin_centers)
    combined_mask  = wave_mask & good_wave_mask

    filtered_rel_scatter = rel_scatter[combined_mask]
    finite_mask = np.isfinite(filtered_rel_scatter)
    median_scatter = (
        np.nanmedian(filtered_rel_scatter[finite_mask])
        if np.sum(finite_mask) > 10 else np.nan
    )

    return {
        'n_obs':          len(binned_fluxes),
        'mean_flux':      mean_flux,
        'std_flux':       std_flux,
        'rel_scatter':    rel_scatter,
        'median_scatter': median_scatter,
        'flux_array':     flux_array,
        'mean_sn':        np.mean([s['median_sn'] for s in high_sn_spectra]),
        'bin_centers':    std_bin_centers,
        'combined_mask':  combined_mask
    }

def build_per_target_arrays(all_target_results, bin_centers):
    """
    Build the minimal set of arrays needed to reproduce the per-target
    consistency figure: the display-grid wavelengths, the per-target
    scatter curves interpolated onto that grid, and the per-target
    median scatter values. Everything else in the figure (median curve,
    percentile bands, histogram stats) is derived from these three.
    """
    wave_mask      = bin_centers >= MIN_WAVELENGTH
    good_wave_mask = mask_bad_wavelengths(bin_centers)
    combined_mask  = wave_mask & good_wave_mask
    filtered_bin_centers = bin_centers[combined_mask]

    # Interpolate each target's 1Å scatter curve onto the 5Å display grid
    all_scatter_curves = []
    median_scatters    = []
    target_names       = []

    for target_name, result in all_target_results.items():
        if result is not None and 'rel_scatter' in result:
            src_centers = result['bin_centers'][result['combined_mask']]
            src_scatter = result['rel_scatter'][result['combined_mask']]
            interp_scatter = np.interp(
                filtered_bin_centers, src_centers, src_scatter,
                left=np.nan, right=np.nan
            )
            all_scatter_curves.append(interp_scatter)
            median_scatters.append(result['median_scatter'])
            target_names.append(target_name)

    if len(all_scatter_curves) == 0:
        return None

    all_scatter_array = np.array(all_scatter_curves)
    median_scatters    = np.array(median_scatters)
    target_names       = np.array(target_names)

    return {
        'filtered_bin_centers': filtered_bin_centers,
        'all_scatter_array':    all_scatter_array,
        'median_scatters':      median_scatters,
        'target_names':         target_names,
    }


def save_per_target_arrays(arrays, output_file="cos_vs_cos_plot_arrays.npz"):
    """Save the minimal plotting arrays for the COS-vs-COS figure to a .npz file."""
    np.savez(
        output_file,
        filtered_bin_centers=arrays['filtered_bin_centers'],
        all_scatter_array=arrays['all_scatter_array'],
        median_scatters=arrays['median_scatters'],
        target_names=arrays['target_names'],
        std_bin_size=STD_BIN_SIZE,
        min_median_sn=MIN_MEDIAN_SN,
    )
    print(f"Saved plotting arrays: {output_file}")


def plot_per_target_consistency(arrays,
                                 output_file="cos_per_target_consistency_scatterplot.png"):
    """Plot per-target consistency: scatter plot + median curve + histogram"""

    filtered_bin_centers = arrays['filtered_bin_centers']
    all_scatter_array    = arrays['all_scatter_array']
    median_scatters      = arrays['median_scatters']

    if len(all_scatter_array) == 0:
        print("No valid target results to plot!")
        return

    median_scatter_curve = np.nanmedian(all_scatter_array, axis=0)
    finite_medians_for_overall = median_scatters[np.isfinite(median_scatters)]
    overall_median = np.nanmedian(finite_medians_for_overall)
    finite_curve   = median_scatter_curve[np.isfinite(median_scatter_curve)]
    overall_max    = np.nanmax(finite_curve) if len(finite_curve) > 0 else 0.5

    fig, ax_scatter = plt.subplots(1, 1, figsize=(8, 6))

    # Scatter plot of all points (wavelength vs relative scatter), with the
    # median curve and overall-median line overlaid
    n_targets = all_scatter_array.shape[0]
    print(f"Plotting scatter for {n_targets} targets...")

    # Tile wavelengths to match flattened scatter values
    all_waves_flat   = np.tile(filtered_bin_centers, n_targets)
    all_scatter_flat = all_scatter_array.ravel()

    finite = np.isfinite(all_scatter_flat) & (all_scatter_flat >= 0)
    ax_scatter.scatter(
        all_waves_flat[finite], all_scatter_flat[finite],
        alpha=0.05, s=1, color='steelblue', rasterized=True
    )
    ax_scatter.plot(
        filtered_bin_centers, median_scatter_curve,
        'r-', linewidth=2.5, label=f'Median ({overall_median*100:.1f}%)', zorder=10
    )
    ax_scatter.axhline(
        overall_median, color='orange', linestyle='--', linewidth=1.8,
        label=f'Overall median: {overall_median:.3f}', zorder=9
    )
    ax_scatter.set_xlabel('Wavelength (Å)', fontsize=12)
    ax_scatter.set_ylabel('Relative Scatter (σ/|μ|)', fontsize=12)
    ax_scatter.set_title(
        f'COS vs COS: Per-Wavelength Scatter\n'
        f'{n_targets} targets, {STD_BIN_SIZE}Å bins, S/N≥{MIN_MEDIAN_SN}',
        fontsize=12, fontweight='bold'
    )
    ax_scatter.legend(fontsize=10, framealpha=0.9)
    ax_scatter.grid(True, alpha=0.3)
    ax_scatter.set_xlim(filtered_bin_centers[0], filtered_bin_centers[-1])
    ax_scatter.set_ylim(0, min(0.5, overall_max * 1.5))

    plt.tight_layout()
    print("Saving plot...")
    plt.savefig(output_file, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"Saved: {output_file}")


def plot_per_target_histogram(arrays,
                               output_file="cos_per_target_scatter_histogram.png"):
    """Standalone histogram of per-target median scatter values."""

    median_scatters = arrays['median_scatters']
    finite_medians = [m * 100 for m in median_scatters if np.isfinite(m)]

    if len(finite_medians) == 0:
        print("No valid median scatter values to plot histogram from!")
        return

    fig, ax_hist = plt.subplots(1, 1, figsize=(8, 6))

    n_bins = min(50, max(10, len(finite_medians) // 10))
    ax_hist.hist(
        finite_medians, bins=n_bins,
        color='steelblue', edgecolor='white', linewidth=0.5, alpha=0.85
    )

    med_val  = np.median(finite_medians)
    mean_val = np.mean(finite_medians)
    p16      = np.percentile(finite_medians, 16)
    p84      = np.percentile(finite_medians, 84)

    ax_hist.axvline(med_val,  color='orange', linestyle='--', linewidth=2,
                    label=f'Median: {med_val:.2f}%')
    ax_hist.axvline(mean_val, color='red',    linestyle=':',  linewidth=2,
                    label=f'Mean: {mean_val:.2f}%')
    ax_hist.axvspan(p16, p84, alpha=0.12, color='orange',
                    label=f'16–84th pct: [{p16:.1f}, {p84:.1f}]%')

    ax_hist.set_xlabel('Per-Target Median Scatter (%)', fontsize=12)
    ax_hist.set_ylabel('Number of Targets', fontsize=12)
    ax_hist.set_title(
        f'Distribution of COS vs COS Scatter\n'
        f'{len(finite_medians)} targets',
        fontsize=12, fontweight='bold'
    )
    ax_hist.legend(fontsize=10, framealpha=0.9)
    ax_hist.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    print("Saving histogram...")
    plt.savefig(output_file, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"Saved: {output_file}")

def save_per_target_summary(all_target_results,
                             output_file="cos_per_target_summary_sn5_bin1_fixed.csv"):
    """Save CSV with per-target consistency metrics"""
    rows = []
    for target_name, result in all_target_results.items():
        if result is not None and np.isfinite(result['median_scatter']):
            rows.append({
                'Target_Name':    target_name,
                'N_Observations': result['n_obs'],
                'Mean_SN':        result.get('mean_sn', 0),
                'Median_Scatter': result['median_scatter']
            })

    df = pd.DataFrame(rows).sort_values('Median_Scatter')
    df.to_csv(output_file, index=False)

    print(f"\nSaved per-target summary: {output_file}")
    print(f"  Total targets: {len(df)}")
    if len(df) > 0:
        print(f"  Best:   {df['Median_Scatter'].min():.4f} ({df['Median_Scatter'].min()*100:.2f}%)")
        print(f"  Worst:  {df['Median_Scatter'].max():.4f} ({df['Median_Scatter'].max()*100:.2f}%)")
        print(f"  Median: {df['Median_Scatter'].median():.4f} ({df['Median_Scatter'].median()*100:.2f}%)")

    return df

def main():
    force_rerun = '--force' in sys.argv
    npz_path = Path("cos_vs_cos_plot_arrays.npz")
    if npz_path.exists() and not force_rerun:
        print(f"Found existing {npz_path}, skipping MAST query/download/analysis.")
        print("Loading saved plot arrays and regenerating plots only...")
        print("(use --force to rerun the full analysis instead)\n")
        data = np.load(npz_path, allow_pickle=True)
        plot_arrays = {
            'filtered_bin_centers': data['filtered_bin_centers'],
            'all_scatter_array':    data['all_scatter_array'],
            'median_scatters':      data['median_scatters'],
            'target_names':         data['target_names'],
        }
        plot_per_target_consistency(
            plot_arrays,
            "cos_per_target_consistency_scatterplot.png"
        )
        plot_per_target_histogram(
            plot_arrays,
            "cos_per_target_scatter_histogram.png"
        )
        print(f"\nDone.")
        return

    print("=== COS PER-TARGET Consistency Analysis ===")
    print(f"Minimum observations per target: {MIN_OBSERVATIONS_PER_TARGET}")
    print(f"Minimum median S/N:              {MIN_MEDIAN_SN}")
    print(f"Display bin size:                {BIN_SIZE}Å")
    print(f"Std calculation bin size:        {STD_BIN_SIZE}Å")
    print(f"Flux threshold (linear):         {FLUX_THRESHOLD:.0e} erg/s/cm²/Å")
    print(f"Excluding {len(BAD_WAVELENGTH_REGIONS)} bad wavelength regions")
    print(f"Excluding {len(VARIABLE_SOURCE_PATTERNS)} variable source patterns")

    obs, x1d_prods = get_all_cos_observations()
    if obs is None or x1d_prods is None:
        print("Failed to query observations!")
        return

    print(f"\nDownloading {len(x1d_prods)} x1d files...")
    downloaded = Observations.download_products(x1d_prods, mrp_only=False)
    if downloaded is None:
        print("Download failed!")
        return

    print(f"Downloaded {len(downloaded)} files")

    print("\nLoading all COS spectra...")
    spectra = []
    excluded_variable = 0
    excluded_low_sn   = 0

    for i, file_path in enumerate(downloaded["Local Path"]):
        if i % 500 == 0:
            print(f"  Loaded {i}/{len(downloaded)} files...")
        if Path(file_path).exists():
            spectrum = load_cos_spectrum(file_path)
            if spectrum is not None and len(spectrum['wavelength']) > 0:
                if not spectrum['target_name']:
                    continue
                if is_variable_source(spectrum['target_name']):
                    excluded_variable += 1
                    continue
                if spectrum['median_sn'] < MIN_MEDIAN_SN:
                    excluded_low_sn += 1
                    continue
                spectra.append(spectrum)

    print(f"\nLoaded {len(spectra)} high-quality spectra")
    print(f"Excluded {excluded_variable} variable source spectra")
    print(f"Excluded {excluded_low_sn} low S/N spectra (S/N < {MIN_MEDIAN_SN})")

    print("\nGrouping by target...")
    target_groups = defaultdict(list)
    for spec in spectra:
        if spec['target_name']:
            target_groups[spec['target_name']].append(spec)

    print(f"Found {len(target_groups)} unique targets")
    valid_targets = {
        name: specs for name, specs in target_groups.items()
        if len(specs) >= MIN_OBSERVATIONS_PER_TARGET
    }
    print(f"Targets with ≥{MIN_OBSERVATIONS_PER_TARGET} observations: {len(valid_targets)}")

    if len(valid_targets) == 0:
        print("No targets found!")
        return

    all_waves = []
    for specs in valid_targets.values():
        for spec in specs:
            if len(spec['wavelength']) > 0:
                all_waves.extend([np.min(spec['wavelength']), np.max(spec['wavelength'])])

    min_wave = min(all_waves)
    max_wave = max(all_waves)
    print(f"\nWavelength range: {min_wave:.1f}–{max_wave:.1f}Å")

    wave_bins   = np.arange(np.floor(min_wave), np.ceil(max_wave) + BIN_SIZE, BIN_SIZE)
    bin_centers = (wave_bins[:-1] + wave_bins[1:]) / 2

    print(f"\nCalculating consistency ({STD_BIN_SIZE}Å bins, linear flux space)...")
    all_target_results = {}
    total_obs = 0

    for i, (target_name, target_specs) in enumerate(valid_targets.items()):
        if i % 50 == 0:
            print(f"  Processed {i}/{len(valid_targets)} targets...")
        result = calculate_per_target_consistency(target_specs, wave_bins, bin_centers)
        if result is not None and np.isfinite(result['median_scatter']):
            all_target_results[target_name] = result
            total_obs += result['n_obs']

    print(f"\nAnalyzed {len(all_target_results)} targets ({total_obs} total observations)")

    median_scatters = [
        r['median_scatter'] for r in all_target_results.values()
        if np.isfinite(r['median_scatter'])
    ]
    if len(median_scatters) == 0:
        print("No valid results!")
        return

    overall_median = np.median(median_scatters)
    overall_mean   = np.mean(median_scatters)
    overall_std    = np.std(median_scatters)

    df = save_per_target_summary(all_target_results, "cos_per_target_summary_scatterplot.csv")

    print("\nBuilding and saving plot arrays...")
    plot_arrays = build_per_target_arrays(all_target_results, bin_centers)
    if plot_arrays is None:
        print("No valid target results to build arrays from!")
        return
    save_per_target_arrays(plot_arrays, "cos_vs_cos_plot_arrays.npz")

    print("\nGenerating plots...")
    plot_per_target_consistency(
        plot_arrays,
        "cos_per_target_consistency_scatterplot.png"
    )
    plot_per_target_histogram(
        plot_arrays,
        "cos_per_target_scatter_histogram.png"
    )

    print(f"\n{'='*70}")
    print(f"COS PER-TARGET CONSISTENCY SUMMARY")
    print(f"{'='*70}")
    print(f"Targets analyzed:            {len(all_target_results)}")
    print(f"Total observations:          {total_obs}")
    print(f"Avg observations per target: {total_obs/len(all_target_results):.1f}")
    print(f"\nCalibration Consistency (linear flux):")
    print(f"  Median scatter: {overall_median:.4f} ({overall_median*100:.2f}%)")
    print(f"  Mean scatter:   {overall_mean:.4f} ({overall_mean*100:.2f}%)")
    print(f"  Std:            {overall_std:.4f}")
    print(f"  Best:   {df.iloc[0]['Target_Name']} = {df.iloc[0]['Median_Scatter']:.4f}")
    print(f"  Worst:  {df.iloc[-1]['Target_Name']} = {df.iloc[-1]['Median_Scatter']:.4f}")

    print(f"\nTop 10 most consistent:")
    for i in range(min(10, len(df))):
        print(f"  {i+1:2d}. {df.iloc[i]['Target_Name']:20s}: "
              f"{df.iloc[i]['Median_Scatter']*100:.2f}% – "
              f"{df.iloc[i]['N_Observations']} obs, S/N={df.iloc[i]['Mean_SN']:.1f}")

    print(f"\nTop 10 least consistent:")
    for i in range(min(10, len(df))):
        idx = len(df) - 1 - i
        print(f"  {i+1:2d}. {df.iloc[idx]['Target_Name']:20s}: "
              f"{df.iloc[idx]['Median_Scatter']*100:.2f}% – "
              f"{df.iloc[idx]['N_Observations']} obs, S/N={df.iloc[idx]['Mean_SN']:.1f}")

    print(f"\nOutput files:")
    print(f"  - cos_per_target_consistency_scatterplot.png")
    print(f"  - cos_per_target_scatter_histogram.png")
    print(f"  - cos_per_target_summary_sn5_bin1_fixed.csv")
    print(f"  - cos_vs_cos_plot_arrays.npz")
    print(f"{'='*70}\n")

if __name__ == "__main__":
    main()