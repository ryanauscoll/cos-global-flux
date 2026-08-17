import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
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

def plot_cos_differences(flux_csv_path, sn_csv_path):
    try:
        flux_df = pd.read_csv(flux_csv_path)
        obs_cols = [c for c in flux_df.columns if 'Observations' in c]
        flux_df[obs_cols] = flux_df[obs_cols].replace(0, np.nan)
        flux_df = flux_df.groupby('Target_Name', as_index=False).first()
        flux_df[obs_cols] = flux_df[obs_cols].fillna(0)
        sn_df = pd.read_csv(sn_csv_path)

        df = flux_df.merge(sn_df, on='Target_Name', how='left')

        print(f"Loaded {len(flux_df)} targets from flux ratios file")
        print(f"Loaded {len(sn_df)} targets from S/N file")
        print(f"Merged dataset has {len(df)} targets\n")

        print("Combining HRS1 and HRS2 into GHRS...")
        df['GHRS_Median_FluxRatio'] = df[['HRS1_Median_FluxRatio', 'HRS2_Median_FluxRatio']].mean(axis=1)
        df['Median_GHRS_SN']        = df[['Median_HRS1_SN', 'Median_HRS2_SN']].mean(axis=1)
        df['GHRS_Observations']     = df['HRS1_Observations'].fillna(0) + df['HRS2_Observations'].fillna(0)

        hrs1_only = df[(df['HRS1_Observations'] > 0) & (df['HRS2_Observations'].isna() | (df['HRS2_Observations'] == 0))]
        hrs2_only = df[(df['HRS2_Observations'] > 0) & (df['HRS1_Observations'].isna() | (df['HRS1_Observations'] == 0))]
        hrs_both  = df[(df['HRS1_Observations'] > 0) & (df['HRS2_Observations'] > 0)]
        print(f"  HRS1 only: {len(hrs1_only)} targets")
        print(f"  HRS2 only: {len(hrs2_only)} targets")
        print(f"  Both HRS1 and HRS2: {len(hrs_both)} targets")
        print(f"  Total GHRS: {len(hrs1_only) + len(hrs2_only) + len(hrs_both)} targets\n")

        hst_instruments = [
            {'name': 'STIS', 'ratio_col': 'STIS_Median_FluxRatio', 'sn_col': 'Median_STIS_SN', 'use_sn_filter': True},
            {'name': 'FOS',  'ratio_col': 'FOS_Median_FluxRatio',  'sn_col': 'Median_FOS_SN',  'use_sn_filter': False},
            {'name': 'GHRS', 'ratio_col': 'GHRS_Median_FluxRatio', 'sn_col': 'Median_GHRS_SN', 'use_sn_filter': False},
        ]
        legacy_instruments = [
            {'name': 'FUSE', 'ratio_col': 'FUSE_Median_FluxRatio', 'sn_col': 'Median_FUSE_SN', 'use_sn_filter': True},
            {'name': 'IUE',  'ratio_col': 'IUE_Median_FluxRatio',  'sn_col': 'Median_IUE_SN',  'use_sn_filter': True},
        ]

        def process_instruments(instruments):
            differences_data    = {}
            original_counts     = {}
            excluded_cos_sn     = {}
            excluded_inst_sn    = {}
            excluded_out_of_range = {}

            for instrument in instruments:
                both_present = df[
                    (df['COS_Observations'] > 0) &
                    (df[instrument['ratio_col']].notna())
                ].copy()
                original_counts[instrument['name']] = len(both_present)

                before_cos_filter = len(both_present)
                both_present = both_present[
                    (both_present['Median_COS_SN'].notna()) &
                    (both_present['Median_COS_SN'] >= 5)
                ]
                excluded_cos_sn[instrument['name']] = before_cos_filter - len(both_present)

                before_inst_filter = len(both_present)
                if instrument.get('use_sn_filter', True):
                    both_present = both_present[
                        (both_present[instrument['sn_col']].notna()) &
                        (both_present[instrument['sn_col']] >= 5)
                    ]
                    excluded_inst_sn[instrument['name']] = before_inst_filter - len(both_present)
                else:
                    excluded_inst_sn[instrument['name']] = 0

                if len(both_present) > 0:
                    ratios_filtered  = both_present[instrument['ratio_col']]
                    ratios_in_range  = ratios_filtered[(ratios_filtered >= -2) & (ratios_filtered <= 2)]
                    excluded_out_of_range[instrument['name']] = len(ratios_filtered) - len(ratios_in_range)
                    differences_data[instrument['name']] = ratios_in_range.values

                    print(f"{instrument['name']}: {len(ratios_in_range)} targets")
                    print(f"  - Started with {original_counts[instrument['name']]} targets with data")
                    print(f"  - Excluded {excluded_cos_sn[instrument['name']]} for COS S/N < 5")
                    if instrument.get('use_sn_filter', True):
                        print(f"  - Excluded {excluded_inst_sn[instrument['name']]} for {instrument['name']} S/N < 5")
                    else:
                        print(f"  - Skipped {instrument['name']} S/N filter (no reliable S/N data)")
                    print(f"  - Excluded {excluded_out_of_range[instrument['name']]} for ratio outside [-2, 2]")
                    print(f"  - Final count: {len(ratios_in_range)}")
                else:
                    print(f"{instrument['name']}: No targets remaining after S/N filtering")

            return differences_data, original_counts, excluded_cos_sn, excluded_inst_sn, excluded_out_of_range

        print("HST INSTRUMENTS:")
        print("="*80)
        hst_data, hst_original, hst_cos_excluded, hst_inst_excluded, hst_oor = process_instruments(hst_instruments)

        print("\nLEGACY INSTRUMENTS:")
        print("="*80)
        legacy_data, legacy_original, legacy_cos_excluded, legacy_inst_excluded, legacy_oor = process_instruments(legacy_instruments)

        bin_edges = np.arange(-2, 2.05, 0.05)

        # HST plot
        if hst_data:
            fig, axes = plt.subplots(1, 3, figsize=(24, 7))
            colors = ['#1f77b4', '#ff7f0e', '#2ca02c']

            for i, (instrument_name, differences) in enumerate(hst_data.items()):
                ax = axes[i]
                ax.hist(differences, bins=bin_edges, color=colors[i], alpha=0.7,
                        edgecolor='black', linewidth=0.5)

                mean_diff   = np.mean(differences)
                median_diff = np.median(differences)
                std_diff    = np.std(differences)
                mad_std_diff = mad_std(differences)


                ax.axvline(x=mean_diff,   color='magenta', linestyle='-', linewidth=3,
                           label=f'Mean: {mean_diff:.3f}')
                ax.axvline(x=median_diff, color='black',   linestyle='-', linewidth=3,
                           label=f'Median: {median_diff:.3f}')

                ax.set_title(f'{instrument_name} vs COS\n({len(differences)} targets)',
                             fontweight='bold')
                ax.set_xlabel('Median Flux Ratio')
                ax.set_ylabel('Number of Targets')
                ax.set_xlim(-2, 2)
                ax.grid(axis='y', alpha=0.3)

                stats_text = (f'Mean: {mean_diff:.2f}\nMedian: {median_diff:.2f}\n'
                              f'Std: {std_diff:.2f}\nMAD std: {mad_std_diff:.2f}')
                ax.text(0.02, 0.98, stats_text, transform=ax.transAxes,
                        verticalalignment='top',
                        bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

            plt.tight_layout()
            plt.savefig('flux_ratio_histograms_HST_final_larger.png', dpi=300, bbox_inches='tight')
            print("\nHST plot saved as 'flux_ratio_histograms_HST_final_larger.png'")
            plt.show()

        # Legacy plot
        if legacy_data:
            fig, axes = plt.subplots(1, 2, figsize=(18, 7))
            colors = ['#9467bd', '#8c564b']

            for i, (instrument_name, differences) in enumerate(legacy_data.items()):
                ax = axes[i]
                ax.hist(differences, bins=bin_edges, color=colors[i], alpha=0.7,
                        edgecolor='black', linewidth=0.5)

                mean_diff   = np.mean(differences)
                median_diff = np.median(differences)
                std_diff    = np.std(differences)
                mad_std_diff = mad_std(differences)

                ax.axvline(x=mean_diff,   color='magenta', linestyle='-', linewidth=3,
                           label=f'Mean: {mean_diff:.3f}')
                ax.axvline(x=median_diff, color='black',   linestyle='-', linewidth=3,
                           label=f'Median: {median_diff:.3f}')

                ax.set_title(f'{instrument_name} vs COS\n({len(differences)} targets)',
                             fontweight='bold')
                ax.set_xlabel('Median Flux Ratio')
                ax.set_ylabel('Number of Targets')
                ax.set_xlim(-2, 2)
                ax.grid(axis='y', alpha=0.3)

                stats_text = (f'Mean: {mean_diff:.2f}\nMedian: {median_diff:.2f}\n'
                              f'Std: {std_diff:.2f}\nMAD std: {mad_std_diff:.2f}')
                ax.text(0.02, 0.98, stats_text, transform=ax.transAxes,
                        verticalalignment='top',
                        bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

            plt.tight_layout()
            plt.savefig('flux_ratio_histograms_Legacy_final_larger.png', dpi=300, bbox_inches='tight')
            print("Legacy plot saved as 'flux_ratio_histograms_Legacy_final_larger.png'")
            plt.show()

        print("\n" + "="*80)
        print("SUMMARY STATISTICS:")
        print("  - COS S/N >= 5 applied to ALL instruments")
        print("  - STIS/FUSE/IUE: Instrument S/N >= 5 filter applied")
        print("  - FOS/GHRS: No instrument S/N filter (unreliable S/N data)")
        print("  - GHRS combines HRS1 and HRS2 data")
        print("="*80)
        print("HST INSTRUMENTS:")
        for instrument_name, differences in hst_data.items():
            mean_diff   = np.mean(differences)
            median_diff = np.median(differences)
            std_diff    = np.std(differences)
            print(f"{instrument_name:>5}: Mean={mean_diff:8.4f}, Median={median_diff:8.4f}, Std={std_diff:8.4f} (N={len(differences)})")

        print("\nLEGACY INSTRUMENTS:")
        for instrument_name, differences in legacy_data.items():
            mean_diff   = np.mean(differences)
            median_diff = np.median(differences)
            std_diff    = np.std(differences)
            print(f"{instrument_name:>5}: Mean={mean_diff:8.4f}, Median={median_diff:8.4f}, Std={std_diff:8.4f} (N={len(differences)})")

    except Exception as e:
        print(f"Error processing file: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    flux_csv_path = "instrument_flux_ratios_and_counts4.csv"
    sn_csv_path   = "all_instruments_sn_analysis_newest.csv"
    plot_cos_differences(flux_csv_path, sn_csv_path)