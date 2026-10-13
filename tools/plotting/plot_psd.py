"""Plot estimated PSDs from saved PSD text files.

This tool loads one or more PSD text files (e.g. produced by
``hwilas_estimate_psd``) and produces a log-log plot of all channels
on a single axis. Input keys are legend labels, so multiple estimates of
the same channel can be plotted together.

Example::

    python tools/plotting/plot_psd.py \
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
        help="PSD input files as label:filename pairs in plotting order (later curves draw on top); quote pairs when labels contain spaces"
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

    parser.add_argument(
        "--y-min", type=float, default=None,
        help="Minimum PSD value for the y-axis"
    )
    parser.add_argument(
        "--y-max", type=float, default=None,
        help="Maximum PSD value for the y-axis"
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

    os.makedirs(os.path.dirname(os.path.abspath(args.output_file)), exist_ok=True)

    fig, ax = plt.subplots(figsize=(10, 6))

    # Draw in command-line order so later curves appear above earlier ones.
    for label, psd_path in args.psd_files.items():
        logger.info("Loading PSD labelled %s from %s", label, psd_path)
        freqs, psd_data = load_psd_data(psd_path)
        mask = (freqs > 0) & (psd_data > 0)
        ax.plot(freqs[mask], psd_data[mask], label=label)

    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlim(args.f_min, args.f_max)
    ax.set_ylim(args.y_min, args.y_max)
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