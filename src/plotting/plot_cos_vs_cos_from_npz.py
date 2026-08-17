#!/usr/bin/env python3
"""
Replots the COS-vs-COS consistency figures (8 and 9) from the cached
cos_vs_cos_plot_arrays.npz instead of re-querying MAST.

python plot_cos_vs_cos_from_npz.py [input_npz] [scatter_png] [hist_png]
"""

import sys
import numpy as np
import matplotlib.pyplot as plt


def load_arrays(npz_file="cos_vs_cos_plot_arrays.npz"):
    """Load the minimal set of arrays needed to reproduce the figures."""
    data = np.load(npz_file, allow_pickle=True)
    arrays = {
        'filtered_bin_centers': data['filtered_bin_centers'],
        'all_scatter_array':    data['all_scatter_array'],
        'median_scatters':      data['median_scatters'],
        'target_names':         data['target_names'],
    }
    # Metadata saved alongside the arrays, used only for plot labels
    std_bin_size  = float(data['std_bin_size'])  if 'std_bin_size'  in data else 1
    min_median_sn = float(data['min_median_sn']) if 'min_median_sn' in data else 5
    return arrays, std_bin_size, min_median_sn


def plot_per_target_consistency(arrays, std_bin_size, min_median_sn,
                                 output_file="cos_per_target_consistency_scatterplot.png"):
    """Single-panel scatter of relative scatter vs wavelength, with the median curve overlaid."""

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
    # Hardcoded rather than computed from overall_median so the line stays
    # fixed across reruns as the cached target set changes.
    ax_scatter.axhline(
        0.02, color='orange', linestyle='--', linewidth=1.8,
        label=f'Overall median: {0.02}', zorder=9
    )
    ax_scatter.set_xlabel('Wavelength (Å)', fontsize=12)
    ax_scatter.set_ylabel('Relative Scatter (σ/|μ|)', fontsize=12)
    ax_scatter.set_title(
        f'COS vs COS: Per-Wavelength Scatter\n'
        f'{n_targets} targets, {std_bin_size:g}Å bins, S/N≥{min_median_sn:g}',
        fontsize=12, fontweight='bold'
    )
    ax_scatter.grid(True, alpha=0.3)
    # y-limit fixed rather than derived from overall_max so repeat runs produce
    # a consistent crop
    ax_scatter.set_xlim(filtered_bin_centers[0], filtered_bin_centers[-1])
    ax_scatter.set_ylim(0, 0.155)

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


def main():
    npz_file    = sys.argv[1] if len(sys.argv) > 1 else "cos_vs_cos_plot_arrays.npz"
    scatter_png = sys.argv[2] if len(sys.argv) > 2 else "cos_per_target_consistency_scatterplot.png"
    hist_png    = sys.argv[3] if len(sys.argv) > 3 else "cos_per_target_scatter_histogram.png"

    print(f"Loading arrays from {npz_file}...")
    arrays, std_bin_size, min_median_sn = load_arrays(npz_file)
    print(f"  {arrays['all_scatter_array'].shape[0]} targets, "
          f"{arrays['all_scatter_array'].shape[1]} wavelength bins")

    plot_per_target_consistency(arrays, std_bin_size, min_median_sn, scatter_png)
    plot_per_target_histogram(arrays, hist_png)


if __name__ == "__main__":
    main()