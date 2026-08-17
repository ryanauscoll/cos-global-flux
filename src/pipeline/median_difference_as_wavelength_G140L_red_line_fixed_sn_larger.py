#!/usr/bin/env python3
"""
Median STIS/FUSE vs COS flux ratio as a function of wavelength (Figure 7).

Downloads spectra via astroquery (or reuses local FITS), bins by wavelength,
and requires S/N >= 5 for both COS and the comparison instrument. G140L data
above the 1900A second-order contamination cutoff is dropped; everything else
is kept.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from astroquery.mast import Observations
from astroquery.exceptions import ResolverError
from pathlib import Path
from astropy.io import fits
from scipy.stats import binned_statistic
from astropy.stats import mad_std

plt.rcParams.update({
    "font.size":        22,
    "axes.titlesize":   24,
    "axes.labelsize":   22,
    "xtick.labelsize":  18,
    "ytick.labelsize":  18,
    "legend.fontsize":  16,
    "figure.titlesize": 26,
})

# Configuration
BIN_SIZE = 10  # angstroms for wavelength binning
VERBOSE = True
COS_G140L_CUTOFF = 1900  # Wavelength cutoff for G140L grating only
SN_THRESHOLD = 5  # Minimum S/N for both COS and comparison instrument

# Exclusion lists
cos_excluded_targets = [
    '1ES1028+511', 'AV296', 'AZV18', 'BI237', 'BI253', 'CD-33-10685', 'CHX18N', 'GD50',
    'GJ1132', 'GJ1214', 'GJ163', 'GJ176', 'GJ436', 'GJ581', 'GJ581B', 'GJ649', 'GJ667C',
    'GJ674', 'GJ676A', 'GJ699', 'GJ832', 'GJ849', 'GJ876', 'HAT-P-11', 'HD103095',
    'HD117555', 'HD129333', 'HD150798', 'HD164058', 'HD39587', 'HD40307', 'HD72905',
    'HD82210', 'HD85512', 'HD97658', 'HE0238-1904', 'HE0435-5304', 'HE0439-5254',
    'HE2347-4342', 'HS0747+4259', 'HS1700+6416', 'IRAS08339+6517', 'IRAS13224-3809',
    'KIC11560431', 'LBQS1435-0134', 'LDS749B', 'M-87', 'MRK-110', 'MRK-231', 'MRK-279',
    'MRK-817', 'MRK335', 'NGC-7469', 'NGC6853', 'PDS456', 'PG1048+342', 'PG1049-005',
    'PG1206+459', 'PG1216+069', 'PG1411+442', 'PG1424+240', 'PG1522+101', 'PG1630+377',
    'PG1718+481', 'PG2112+059', 'PKS1302-102', 'PKS2005-489', 'PMNJ1103-2329',
    'PSRB0656+14', 'RXJ0438.6+1546', 'RXJ1556.1-3655', 'RXJ1852.3-3700', 'RXJ2154.1-4414',
    'SDSS1711+6052', 'SDSS2346-0016', 'SK-6826', 'SN2010AL', 'SN2010JL',
    'SSTC2DJ161243.8-381503', 'SSTC2DJ161344.1-373646', 'SZ10', 'SZ100', 'SZ104',
    'SZ110', 'SZ117', 'SZ129', 'SZ130', 'SZ19', 'SZ66', 'SZ69', 'SZ71', 'SZ72', 'SZ76',
    'SZ82', 'SZ97', 'SZ98', 'SZ99', 'TOI-1468', 'TOI-561', 'TONS210', 'TWA13A',
    'VIIZW403', 'WASP-29', 'WASP-43', 'WASP-69', 'WASP-80', 'WD1202-232', 'WD1337+705',
    'WD1544-377', 'WD1657+343', 'WD1743-521', 'WD1917-077'
]

stis_excluded_targets = [
    'AV234', 'AZV18', 'BI184', 'BI237', 'BI253', 'GJ1132', 'GJ1214', 'GJ163', 'GJ176',
    'GJ436', 'GJ581', 'GJ581B', 'GJ649', 'GJ667C', 'GJ674', 'GJ676A', 'GJ699', 'GJ832',
    'GJ849', 'GJ876', 'H1821+643', 'HAT-P-11', 'HD117555', 'HD129333', 'HD164058',
    'HD39587', 'HD40307', 'HD72905', 'HD82210', 'HD85512', 'HD97658', 'HE2347-4342',
    'HS0747+4259', 'HS1700+6416', 'IRAS08339+6517', 'LBQS1435-0134', 'MRK-509',
    'MRK-817', 'MRK335', 'MRK876', 'PG0804+761', 'PG1049-005', 'PG1116+215',
    'PG1206+459', 'PG1211+143', 'PG1216+069', 'PG1259+593', 'PG1424+240', 'PG1522+101',
    'PG1630+377', 'PG1718+481', 'PHL1811', 'PKS0405-123', 'PKS1302-102', 'PKS2155-304',
    'SDSS1711+6052', 'SDSS2346-0016', 'SK-68129', 'SK-68140', 'SN2010AL',
    'SSTC2DJ161243.8-381503', 'SSTC2DJ161344.1-373646', 'SZ10', 'SZ100', 'SZ104',
    'SZ110', 'SZ117', 'SZ130', 'SZ66', 'SZ69', 'SZ71', 'SZ76', 'SZ97', 'TOI-1468',
    'TOI-561', 'TONS210', 'TWA13A', 'VIIZW403', 'WASP-29', 'WASP-43', 'WASP-69',
    'WASP-80', 'WD1202-232'
]

fuse_excluded_targets = cos_excluded_targets.copy()


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
            print(f"[WARN] ResolverError for '{target}'")
        return None
    except Exception as e:
        if VERBOSE:
            print(f"[ERROR] Query error for '{target}': {e}")
        return None

    if len(obs) == 0:
        return None

    prods = Observations.get_product_list(obs)
    prods = prods[prods["productType"] == "SCIENCE"]
    fnames = np.array(prods["productFilename"].astype(str))
    keep = np.zeros_like(fnames, dtype=bool)
    for suf in suffixes:
        keep |= np.char.endswith(fnames, suf)

    return prods[keep] if np.any(keep) else None


def load_x1d(fname, target_name):
    try:
        with fits.open(fname) as hd:
            opt_elem_from_header = None
            if 'OPT_ELEM' in hd[0].header:
                opt_elem_from_header = hd[0].header['OPT_ELEM']
            elif len(hd) > 1 and 'OPT_ELEM' in hd[1].header:
                opt_elem_from_header = hd[1].header['OPT_ELEM']

            w_all, f_all, oe_all = [], [], []

            for row in hd[1].data:
                wl, fl = row["WAVELENGTH"], row["FLUX"]
                g = (fl > 0) & np.isfinite(wl) & np.isfinite(fl)
                if np.any(g):
                    logf = np.log10(fl[g])
                    final_g = (logf <= -5) & np.isfinite(logf)
                    if np.any(final_g):
                        n_pts = np.sum(final_g)
                        w_all.append(wl[g][final_g])
                        f_all.append(logf[final_g])
                        row_opt_elem = None
                        if hasattr(row, 'dtype') and 'OPT_ELEM' in row.dtype.names:
                            row_opt_elem = row['OPT_ELEM']
                        elif opt_elem_from_header is not None:
                            row_opt_elem = opt_elem_from_header
                        oe_label = row_opt_elem if row_opt_elem is not None else 'UNKNOWN'
                        oe_all.append(np.full(n_pts, oe_label, dtype='U20'))

            if w_all and f_all:
                return np.concatenate(w_all), np.concatenate(f_all), np.concatenate(oe_all)
            else:
                return np.array([]), np.array([]), np.array([], dtype='U20')
    except Exception as e:
        if VERBOSE:
            print(f"[ERROR] Failed to load {fname}: {e}")
        return np.array([]), np.array([]), np.array([], dtype='U20')


def load_fuse_spectrum(fname, target_name):
    try:
        with fits.open(fname) as hd:
            data = None
            if len(hd) > 1 and hd[1].data is not None:
                data = hd[1].data
            elif hd[0].data is not None:
                data = hd[0].data
            if data is None:
                return np.array([]), np.array([])

            col_names = [col.name.upper() for col in data.columns] if hasattr(data, 'columns') else []
            wave_col = next((n for n in ['WAVELENGTH', 'WAVE', 'LAMBDA', 'WL'] if n in col_names), None)
            flux_col = next((n for n in ['FLUX', 'FLUX_TTAG', 'FLUX_HIST'] if n in col_names), None)

            if wave_col is None or flux_col is None:
                if VERBOSE:
                    print(f"[WARN] Could not find wavelength/flux columns in {fname}")
                return np.array([]), np.array([])

            wl = data[wave_col]
            fl = data[flux_col]
            g = (fl > 0) & np.isfinite(wl) & np.isfinite(fl)
            if np.any(g):
                logf = np.log10(fl[g])
                final_g = (logf <= -5) & np.isfinite(logf)
                if np.any(final_g):
                    return wl[g][final_g], logf[final_g]
            return np.array([]), np.array([])
    except Exception as e:
        if VERBOSE:
            print(f"[ERROR] Failed to load FUSE file: {fname} – {e}")
        return np.array([]), np.array([])


def bin_spectrum(w, f, bin_edges):
    bin_means, _, _ = binned_statistic(w, f, bins=bin_edges, statistic='mean')
    return bin_means


def analyze_target_wavelength_bins(target, instrument='STIS'):
    if VERBOSE:
        print(f"Analyzing {target} ({instrument})...")

    cos_tab = query_products(target, "COS", ["_x1d.fits"])
    if instrument == 'STIS':
        inst_tab = query_products(target, ["STIS", "STIS/FUV-MAMA", "STIS/NUV-MAMA", "STIS/CCD"], ["_x1d.fits"])
    elif instrument == 'FUSE':
        inst_tab = query_products(target, "FUSE", ["_vo.fits", "_all.fits"], is_collection=True)
    else:
        return None

    if cos_tab is None or inst_tab is None:
        return None

    cos_dl  = Observations.download_products(cos_tab,  mrp_only=False)
    inst_dl = Observations.download_products(inst_tab, mrp_only=False)
    if cos_dl is None or inst_dl is None:
        return None

    cos_wave_all, cos_flux_all, cos_oe_all = [], [], []
    for path in cos_dl["Local Path"]:
        if Path(path).exists():
            w, f, oe = load_x1d(path, target)
            if len(w) > 0:
                cos_wave_all.append(w); cos_flux_all.append(f); cos_oe_all.append(oe)

    inst_wave_all, inst_flux_all, inst_oe_all = [], [], []
    for path in inst_dl["Local Path"]:
        if Path(path).exists():
            if instrument == 'STIS':
                w, f, oe = load_x1d(path, target)
                if len(w) > 0:
                    inst_wave_all.append(w); inst_flux_all.append(f); inst_oe_all.append(oe)
            elif instrument == 'FUSE':
                w, f = load_fuse_spectrum(path, target)
                if len(w) > 0:
                    inst_wave_all.append(w); inst_flux_all.append(f)
                    inst_oe_all.append(np.full(len(w), 'FUSE', dtype='U20'))

    if not cos_wave_all or not inst_wave_all:
        return None

    cos_wave  = np.concatenate(cos_wave_all)
    cos_flux  = np.concatenate(cos_flux_all)
    cos_oe    = np.concatenate(cos_oe_all)
    inst_wave = np.concatenate(inst_wave_all)
    inst_flux = np.concatenate(inst_flux_all)
    inst_oe   = np.concatenate(inst_oe_all)

    cos_has_g140l  = np.any(cos_oe == 'G140L')
    inst_has_g140l = np.any(inst_oe == 'G140L')
    has_g140l = cos_has_g140l or inst_has_g140l

    if has_g140l:
        cos_g140l_over  = (cos_oe == 'G140L')  & (cos_wave  >= COS_G140L_CUTOFF)
        inst_g140l_over = (inst_oe == 'G140L')  & (inst_wave >= COS_G140L_CUTOFF)
        n_cos_removed   = np.sum(cos_g140l_over)
        n_inst_removed  = np.sum(inst_g140l_over)
        cos_wave  = cos_wave[~cos_g140l_over];  cos_flux  = cos_flux[~cos_g140l_over]
        cos_oe    = cos_oe[~cos_g140l_over]
        inst_wave = inst_wave[~inst_g140l_over]; inst_flux = inst_flux[~inst_g140l_over]
        inst_oe   = inst_oe[~inst_g140l_over]
        if VERBOSE:
            sources = []
            if cos_has_g140l:  sources.append("COS")
            if inst_has_g140l: sources.append(instrument)
            print(f"  G140L detected in {', '.join(sources)}: "
                  f"removed {n_cos_removed} COS + {n_inst_removed} {instrument} "
                  f"G140L points >= {COS_G140L_CUTOFF}Å")

    if len(cos_wave) == 0 or len(inst_wave) == 0:
        return None

    min_wl = max(np.min(cos_wave), np.min(inst_wave))
    max_wl = min(np.max(cos_wave), np.max(inst_wave))
    if max_wl <= min_wl:
        return None

    overlap_bins = np.arange(np.floor(min_wl), np.ceil(max_wl) + BIN_SIZE, BIN_SIZE)
    binned_cos  = bin_spectrum(cos_wave,  cos_flux,  overlap_bins)
    binned_inst = bin_spectrum(inst_wave, inst_flux, overlap_bins)
    ratio = (10**binned_inst - 10**binned_cos) / 10**binned_cos
    bin_centers = (overlap_bins[:-1] + overlap_bins[1:]) / 2

    return bin_centers, ratio, has_g140l


def main():
    flux_df = pd.read_csv('instrument_flux_ratios_and_counts4.csv')
    sn_df   = pd.read_csv('all_instruments_sn_analysis_newest.csv')

    obs_cols = [c for c in flux_df.columns if 'Observations' in c]
    flux_df[obs_cols] = flux_df[obs_cols].replace(0, np.nan)
    flux_df = flux_df.groupby('Target_Name', as_index=False).first()
    flux_df[obs_cols] = flux_df[obs_cols].fillna(0)
    df = flux_df.merge(sn_df, on='Target_Name', how='left')

    print(f"Loaded {len(flux_df)} unique targets from flux ratios file")
    print(f"Loaded {len(sn_df)} targets from S/N file")
    print(f"Merged dataset has {len(df)} targets\n")

    # STIS
    print("=== Processing STIS data ===")
    stis_present = df[(df['COS_Observations'] > 0) & (df['STIS_Median_FluxRatio'].notna())].copy()
    n_start_stis = len(stis_present)
    stis_present = stis_present[~stis_present['Target_Name'].isin(cos_excluded_targets)]
    stis_present = stis_present[~stis_present['Target_Name'].isin(stis_excluded_targets)]
    n_after_excl_stis = len(stis_present)
    stis_present = stis_present[(stis_present['Median_COS_SN'].notna()) & (stis_present['Median_COS_SN'] >= SN_THRESHOLD)]
    n_after_cos_sn_stis = len(stis_present)
    stis_present = stis_present[(stis_present['Median_STIS_SN'].notna()) & (stis_present['Median_STIS_SN'] >= SN_THRESHOLD)]
    n_after_inst_sn_stis = len(stis_present)
    stis_present = stis_present[(stis_present['STIS_Median_FluxRatio'] >= -2) & (stis_present['STIS_Median_FluxRatio'] <= 2)]
    n_final_stis = len(stis_present)
    print(f"  Started: {n_start_stis} | After exclusions: {n_after_excl_stis} | After COS S/N: {n_after_cos_sn_stis} | After STIS S/N: {n_after_inst_sn_stis} | Final: {n_final_stis}\n")

    stis_wavelengths, stis_ratios = [], []
    successful_stis = 0; stis_g140l_count = 0
    for i, target in enumerate(stis_present['Target_Name'].values, 1):
        print(f"[{i}/{n_final_stis}] {target} (STIS)...")
        result = analyze_target_wavelength_bins(target, 'STIS')
        if result is not None:
            wavelengths, ratios, has_g140l = result
            if has_g140l: stis_g140l_count += 1
            valid_mask = np.isfinite(ratios) & (ratios >= -2) & (ratios <= 2)
            if np.any(valid_mask):
                stis_wavelengths.extend(wavelengths[valid_mask])
                stis_ratios.extend(ratios[valid_mask])
                successful_stis += 1
                print(f"  ✓ {np.sum(valid_mask)} valid bins")

    # FUSE
    print("\n=== Processing FUSE data ===")
    fuse_present = df[(df['COS_Observations'] > 0) & (df['FUSE_Median_FluxRatio'].notna())].copy()
    n_start_fuse = len(fuse_present)
    fuse_present = fuse_present[~fuse_present['Target_Name'].isin(cos_excluded_targets)]
    fuse_present = fuse_present[~fuse_present['Target_Name'].isin(fuse_excluded_targets)]
    n_after_excl_fuse = len(fuse_present)
    fuse_present = fuse_present[(fuse_present['Median_COS_SN'].notna()) & (fuse_present['Median_COS_SN'] >= SN_THRESHOLD)]
    n_after_cos_sn_fuse = len(fuse_present)
    fuse_present = fuse_present[(fuse_present['Median_FUSE_SN'].notna()) & (fuse_present['Median_FUSE_SN'] >= SN_THRESHOLD)]
    n_after_inst_sn_fuse = len(fuse_present)
    fuse_present = fuse_present[(fuse_present['FUSE_Median_FluxRatio'] >= -2) & (fuse_present['FUSE_Median_FluxRatio'] <= 2)]
    n_final_fuse = len(fuse_present)
    print(f"  Started: {n_start_fuse} | After exclusions: {n_after_excl_fuse} | After COS S/N: {n_after_cos_sn_fuse} | After FUSE S/N: {n_after_inst_sn_fuse} | Final: {n_final_fuse}\n")

    fuse_wavelengths, fuse_ratios = [], []
    successful_fuse = 0; fuse_g140l_count = 0
    for i, target in enumerate(fuse_present['Target_Name'].values, 1):
        print(f"[{i}/{n_final_fuse}] {target} (FUSE)...")
        result = analyze_target_wavelength_bins(target, 'FUSE')
        if result is not None:
            wavelengths, ratios, has_g140l = result
            if has_g140l: fuse_g140l_count += 1
            valid_mask = np.isfinite(ratios) & (ratios >= -2) & (ratios <= 2)
            if np.any(valid_mask):
                fuse_wavelengths.extend(wavelengths[valid_mask])
                fuse_ratios.extend(ratios[valid_mask])
                successful_fuse += 1
                print(f"  ✓ {np.sum(valid_mask)} valid bins")

    print(f"\nSuccessfully processed {successful_stis} STIS and {successful_fuse} FUSE targets")
    if len(stis_ratios) == 0 and len(fuse_ratios) == 0:
        print("No data to plot!")
        return

    stis_wavelengths = np.array(stis_wavelengths)
    stis_ratios      = np.array(stis_ratios)
    fuse_wavelengths = np.array(fuse_wavelengths)
    fuse_ratios      = np.array(fuse_ratios)

    wave_bins = np.arange(900, 3200, 50)

    stis_median_ratios,  bin_edges, _ = binned_statistic(stis_wavelengths, stis_ratios, bins=wave_bins, statistic='median')
    stis_mad_std_ratios, _,         _ = binned_statistic(stis_wavelengths, stis_ratios, bins=wave_bins, statistic=mad_std)
    stis_counts,         _,         _ = binned_statistic(stis_wavelengths, stis_ratios, bins=wave_bins, statistic='count')

    fuse_median_ratios,  _, _ = binned_statistic(fuse_wavelengths, fuse_ratios, bins=wave_bins, statistic='median')
    fuse_mad_std_ratios, _, _ = binned_statistic(fuse_wavelengths, fuse_ratios, bins=wave_bins, statistic=mad_std)
    fuse_counts,         _, _ = binned_statistic(fuse_wavelengths, fuse_ratios, bins=wave_bins, statistic='count')

    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    min_points  = 5
    stis_valid_bins = stis_counts >= min_points
    fuse_valid_bins = fuse_counts >= min_points

    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(18, 18))

    # Panel 1: flux ratios vs wavelength
    if np.any(stis_valid_bins):
        ax1.errorbar(bin_centers[stis_valid_bins], stis_median_ratios[stis_valid_bins],
                     yerr=stis_mad_std_ratios[stis_valid_bins],
                     fmt='o-', color='green', linewidth=2, markersize=6,
                     capsize=4, capthick=2, alpha=0.8, label='STIS Median ± MAD std')
    if np.any(fuse_valid_bins):
        ax1.errorbar(bin_centers[fuse_valid_bins], fuse_median_ratios[fuse_valid_bins],
                     yerr=fuse_mad_std_ratios[fuse_valid_bins],
                     fmt='s-', color='purple', linewidth=2, markersize=6,
                     capsize=4, capthick=2, alpha=0.8, label='FUSE Median ± MAD std')

    ax1.axhline(0, color='red', linestyle='--', linewidth=2, alpha=0.7, label='Zero line')
    ax1.axvline(1900, color='red', linestyle='-', linewidth=2, alpha=0.5, label='G140L cutoff (1900 Å)')
    ax1.set_xlabel('Wavelength (Å)')
    ax1.set_ylabel('Median Flux Ratio\n(Instrument - COS) / COS')

    title = (f'STIS and FUSE vs COS Median Flux Difference as a Function of Wavelength\n'
             f'COS S/N ≥ {SN_THRESHOLD}, Instrument S/N ≥ {SN_THRESHOLD} | '
             f'STIS: {successful_stis} targets, FUSE: {successful_fuse} targets')
    if stis_g140l_count > 0 or fuse_g140l_count > 0:
        title += f', G140L filtered for {stis_g140l_count + fuse_g140l_count} targets'
    ax1.set_title(title, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    ax1.set_ylim(-0.5, 0.5)

    # Panel 2: STIS coverage
    ax2.bar(bin_centers[stis_valid_bins], stis_counts[stis_valid_bins],
            width=40, color='lightgreen', edgecolor='darkgreen', alpha=0.7, label='STIS')
    ax2.set_xlabel('Wavelength (Å)')
    ax2.set_ylabel('Number of Data Points')
    ax2.set_title('STIS Data Coverage per Wavelength Bin', fontweight='bold')
    ax2.grid(True, alpha=0.3, axis='y')
    ax2.legend()

    # Panel 3: FUSE coverage
    ax3.bar(bin_centers[fuse_valid_bins], fuse_counts[fuse_valid_bins],
            width=40, color='plum', edgecolor='purple', alpha=0.7, label='FUSE')
    ax3.set_xlabel('Wavelength (Å)')
    ax3.set_ylabel('Number of Data Points')
    ax3.set_title('FUSE Data Coverage per Wavelength Bin', fontweight='bold')
    ax3.grid(True, alpha=0.3, axis='y')
    ax3.legend()

    plt.tight_layout()
    output_path = 'stis_fuse_cos_wavelength_comparison_g140l_red_line_fixed_sn_larger.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"\nPlot saved to: {output_path}")

    print("\n=== STIS Summary by wavelength range ===")
    for w_min, w_max in [(1150, 1450), (1450, 1750), (1750, 2050), (2050, 2500), (2500, 3000)]:
        mask = (stis_wavelengths >= w_min) & (stis_wavelengths < w_max)
        if np.any(mask):
            print(f"  {w_min}-{w_max} Å: median={np.median(stis_ratios[mask]):.4f}, "
                  f"MAD std={mad_std(stis_ratios[mask]):.4f}, N={np.sum(mask)}")

    print("\n=== FUSE Summary by wavelength range ===")
    for w_min, w_max in [(900, 1000), (1000, 1100), (1100, 1200)]:
        mask = (fuse_wavelengths >= w_min) & (fuse_wavelengths < w_max)
        if np.any(mask):
            print(f"  {w_min}-{w_max} Å: median={np.median(fuse_ratios[mask]):.4f}, "
                  f"MAD std={mad_std(fuse_ratios[mask]):.4f}, N={np.sum(mask)}")

    plt.show()


if __name__ == "__main__":
    main()