"""Plot estimated PSDs from saved PSD text files.

This tool loads one or more PSD text files (e.g. produced by
``hwilas_estimate_mojito_psd``) and produces a log-log plot of all channels
on a single axis.

Example::

    python tools/plotting/plot_estimate_mojito_psd.py \
        --psd-files LISA_A:output/A_mojito_SMOOTHED_PSD.txt \
                   LISA_E:output/E_mojito_SMOOTHED_PSD.txt \
                   LISA_T:output/T_mojito_SMOOTHED_PSD.txt \
        --output-file output/mojito_psd.png
"""

import argparse
import logging
import os

import numpy as np
from matplotlib import use

use('Agg')
from matplotlib import pyplot as plt

from hwilas import add_hwilas_common_options, init_logging
from hwilas.types import KeyValueAction

logger = logging.getLogger(__name__)


def load_arguments():
    parser = argparse.ArgumentParser(
        description="Plot estimated PSDs from saved PSD text files."
    )
    add_hwilas_common_options(parser)

    parser.add_argument(
        "--psd-files",
        required=True,
        action=KeyValueAction,
        nargs='+',
        help="PSD input files, one per channel, given as channel:filename pairs"
    )
    parser.add_argument(
        "--unsmoothed-psd-files", action=KeyValueAction, nargs='+',
        help="Optional raw PSDs as channel:filename pairs, overlaid as dashed curves",
    )
    parser.add_argument(
        "--output-file",
        required=True,
        help="Output plot filename"
    )
    parser.add_argument(
        "--f-min",
        type=float,
        default=1e-6,
        help="Minimum frequency for the x-axis (Hz). Default: 1e-6"
    )
    parser.add_argument(
        "--f-max",
        type=float,
        default=0.1,
        help="Maximum frequency for the x-axis (Hz). Default: 0.1"
    )

    return parser.parse_args()


def load_psd_data(path):
    """Load raw two-column PSD data from a text file."""
    file_data = np.loadtxt(path)
    if file_data.ndim != 2 or file_data.shape[1] < 2:
        raise ValueError(f"Expected a two-column PSD file, got shape {file_data.shape}")
    return file_data[:, 0], file_data[:, 1]


def main():
    args = load_arguments()
    init_logging(args.verbose)
    logger.info("Starting")

    channels = list(args.psd_files.keys())
    logger.debug("Channels: %s", channels)

    if args.unsmoothed_psd_files and set(args.unsmoothed_psd_files) != set(channels):
        raise ValueError("Provide an unsmoothed PSD for every plotted channel")

    os.makedirs(os.path.dirname(os.path.abspath(args.output_file)), exist_ok=True)

    fig, ax = plt.subplots(figsize=(10, 6))

    for index, channel in enumerate(channels):
        color = f"C{index % 10}"
        if args.unsmoothed_psd_files:
            raw_freqs, raw_psd = load_psd_data(args.unsmoothed_psd_files[channel])
            mask = (raw_freqs > 0) & (raw_psd > 0)
            ax.plot(raw_freqs[mask], raw_psd[mask], color=color, linestyle="--",
                    alpha=0.5, label=f"{channel} unsmoothed")
        psd_path = args.psd_files[channel]
        logger.info("Loading PSD for %s from %s", channel, psd_path)
        freqs, psd_data = load_psd_data(psd_path)
        mask = (freqs > 0) & (psd_data > 0)
        label = f"{channel} smoothed" if args.unsmoothed_psd_files else channel
        ax.plot(freqs[mask], psd_data[mask], color=color, label=label)

    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlim(args.f_min, args.f_max)
    ax.set_xlabel('Frequency (Hz)')
    ax.set_ylabel('PSD')
    ax.legend(loc='upper left')

    output_path = args.output_file
    fig.savefig(output_path, bbox_inches='tight')
    plt.close(fig)
    logger.info("Saved plot to %s", output_path)

    logger.info("Done!")


if __name__ == "__main__":
    main()