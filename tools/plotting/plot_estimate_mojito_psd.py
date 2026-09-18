"""Plot estimated PSDs from saved PSD text files.

This tool loads one or more PSD text files (e.g. produced by
``hwilas_estimate_mojito_psd``) and produces a log-log plot of all channels
on a single axis.

Example::

    python tools/plotting/plot_estimate_mojito_psd.py \
        --psd-files LISA_A:output/A_mojito_SMOOTHED_PSD.txt \
                   LISA_E:output/E_mojito_SMOOTHED_PSD.txt \
                   LISA_T:output/T_mojito_SMOOTHED_PSD.txt \
        --output-directory output
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
        "--output-directory",
        required=True,
        help="Directory in which to save the plot"
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

    os.makedirs(args.output_directory, exist_ok=True)

    fig, ax = plt.subplots(figsize=(10, 6))

    for channel in channels:
        psd_path = args.psd_files[channel]
        logger.info("Loading PSD for %s from %s", channel, psd_path)
        freqs, psd_data = load_psd_data(psd_path)
        ax.plot(freqs, psd_data, label=channel)

    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlim(args.f_min, args.f_max)
    ax.set_xlabel('Frequency (Hz)')
    ax.set_ylabel('PSD')
    ax.legend(loc='upper left')

    output_path = os.path.join(args.output_directory, "mojito_psd.png")
    fig.savefig(output_path, bbox_inches='tight')
    plt.close(fig)
    logger.info("Saved plot to %s", output_path)

    logger.info("Done!")


if __name__ == "__main__":
    main()