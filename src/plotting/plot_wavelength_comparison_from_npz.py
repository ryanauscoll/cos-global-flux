import sys
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

# Default per-instrument display styles. Any instrument found in the npz
# file but not listed here falls back to matplotlib's default color cycle.
DEFAULT_INSTRUMENT_STYLES = {
    'STIS': {'errorbar_fmt': 'o-', 'errorbar_color': 'green',  'bar_color': 'lightgreen', 'bar_edgecolor': 'darkgreen'},
    'FUSE': {'errorbar_fmt': 's-', 'errorbar_color': 'purple', 'bar_color': 'plum',       'bar_edgecolor': 'purple'},
}

# Wavelength ranges used for the per-instrument printed summary. Falls back
# to a single full-range bucket for any instrument not listed here.
SUMMARY_RANGES = {
    'STIS': [(1150, 1450), (1450, 1750), (1750, 2050), (2050, 2500), (2500, 3000)],
    'FUSE': [(900, 1000), (1000, 1100), (1100, 1200)],
}


def load_arrays(npz_file="wavelength_comparison_plot_arrays.npz"):
    """Load the minimal arrays needed to reproduce the comparison figure."""
    data = np.load(npz_file, allow_pickle=True)

    instrument_names = [str(x) for x in data['instrument_names']]

    instruments = {}
    for name in instrument_names:
        instruments[name] = {
            'median_ratios':  data[f'{name}_median_ratios'],
            'mad_std_ratios': data[f'{name}_mad_std_ratios'],
            'counts':         data[f'{name}_counts'],
            'valid_bins':     data[f'{name}_valid_bins'],
            'successful':     int(data[f'{name}_successful']),
            'g140l_count':    int(data[f'{name}_g140l_count']),
            'wavelengths':    data[f'{name}_wavelengths'],
            'ratios':         data[f'{name}_ratios'],
        }

    plot_arrays = {
        'bin_centers':  data['bin_centers'],
        'min_points':   int(data['min_points']),
        'sn_threshold': float(data['sn_threshold']),
        'g140l_cutoff': float(data['g140l_cutoff']),
        'instruments':  instruments,
    }
    return plot_arrays


def plot_wavelength_comparison(plot_arrays, instrument_styles=None,
                                output_file="stis_fuse_cos_wavelength_comparison.png"):
    """Two-panel figure: median flux ratio vs wavelength per instrument, plus coverage histogram."""
    if instrument_styles is None:
        instrument_styles = DEFAULT_INSTRUMENT_STYLES

    bin_centers = plot_arrays['bin_centers']
    instruments = plot_arrays['instruments']

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(18, 12))

    # Panel 1: flux ratios vs wavelength, one errorbar series per instrument
    title_parts = []
    total_g140l = 0
    for instrument, data in instruments.items():
        valid_bins = data['valid_bins']
        if not np.any(valid_bins):
            continue
        style = instrument_styles.get(instrument, {})
        ax1.errorbar(
            bin_centers[valid_bins], data['median_ratios'][valid_bins],
            yerr=data['mad_std_ratios'][valid_bins],
            fmt=style.get('errorbar_fmt', 'o-'),
            color=style.get('errorbar_color', None),
            linewidth=2, markersize=6, capsize=4, capthick=2, alpha=0.8,
            label=f'{instrument} Median ± MAD std'
        )
        title_parts.append(f'{instrument}: {data["successful"]} targets')
        total_g140l += data['g140l_count']

    ax1.axhline(0, color='red', linestyle='--', linewidth=2, alpha=0.7, label='Zero line')
    ax1.axvline(plot_arrays.get('g140l_cutoff', 1900), color='red', linestyle='-',
                linewidth=2, alpha=0.5, label='G140L cutoff (1900 Å)')
    ax1.set_xlabel('Wavelength (Å)')
    ax1.set_ylabel('Median Flux Ratio\n(Instrument - COS) / COS')

    sn_threshold = plot_arrays.get('sn_threshold', 5)
    title = (f'{" and ".join(instruments.keys())} vs COS Median Flux Difference as a Function of Wavelength\n')
    ax1.set_title(title, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    ax1.set_ylim(-0.5, 0.5)

    # Panel 2: combined coverage histogram, one bar series per instrument
    n_instruments = len(instruments)
    bar_width = 40 / max(n_instruments, 1)
    for idx, (instrument, data) in enumerate(instruments.items()):
        valid_bins = data['valid_bins']
        if not np.any(valid_bins):
            continue
        style = instrument_styles.get(instrument, {})
        offset = (idx - (n_instruments - 1) / 2) * bar_width
        ax2.bar(
            bin_centers[valid_bins] + offset, data['counts'][valid_bins],
            width=bar_width,
            color=style.get('bar_color', None),
            edgecolor=style.get('bar_edgecolor', None),
            alpha=0.7, label=instrument
        )

    ax2.set_xlabel('Wavelength (Å)')
    ax2.set_ylabel('Number of Data Points')
    ax2.set_title('Data Coverage per Wavelength Bin', fontweight='bold')
    ax2.grid(True, alpha=0.3, axis='y')
    ax2.legend()

    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"Plot saved to: {output_file}")
    return fig


def print_wavelength_range_summary(plot_arrays):
    """Print the per-instrument median/MAD-std summary over each instrument's summary ranges."""
    for instrument, data in plot_arrays['instruments'].items():
        wavelengths = data['wavelengths']
        ratios      = data['ratios']
        ranges = SUMMARY_RANGES.get(instrument, [(float(wavelengths.min()), float(wavelengths.max()))] if len(wavelengths) else [])
        print(f"\n=== {instrument} Summary by wavelength range ===")
        for w_min, w_max in ranges:
            mask = (wavelengths >= w_min) & (wavelengths < w_max)
            if np.any(mask):
                print(f"  {w_min}-{w_max} Å: median={np.median(ratios[mask]):.4f}, "
                      f"MAD std={mad_std(ratios[mask]):.4f}, N={np.sum(mask)}")


def main():
    npz_file   = sys.argv[1] if len(sys.argv) > 1 else "wavelength_comparison_plot_arrays.npz"
    output_png = sys.argv[2] if len(sys.argv) > 2 else "stis_fuse_cos_wavelength_comparison.png"

    print(f"Loading arrays from {npz_file}...")
    plot_arrays = load_arrays(npz_file)
    print(f"  Instruments found: {list(plot_arrays['instruments'].keys())}")

    plot_wavelength_comparison(plot_arrays, output_file=output_png)
    print_wavelength_range_summary(plot_arrays)


if __name__ == "__main__":
    main()