"""Plot matched-filter SNR peaks from a trigger file.

This tool loads an HDF5 file of clustered SNR peaks (e.g. produced by
``hwilas_matched_filter_search``) and produces a scatter plot of the peaks
(SNR vs time in days). Only the clustered peaks are persisted by the search,
so this is a scatter rather than a full SNR timeseries.

Example::

    python tools/plotting/plot_matched_filter_snr.py \
        --trigger-file output/matched_filter/matched_filter_snr_peaks_LISA_A_0.hdf \
        --channel LISA_A \
        --output-directory plots/matched_filter
"""

import argparse
import logging
import os

import h5py
import numpy as np

from matplotlib import use

use('Agg')
from matplotlib import pyplot as plt

from hwilas import add_hwilas_common_options, init_logging

logger = logging.getLogger(__name__)


def load_arguments():
    parser = argparse.ArgumentParser(
        description="Plot clustered matched-filter SNR peaks."
    )
    add_hwilas_common_options(parser)

    parser.add_argument(
        "--trigger-file",
        required=True,
        help="HDF5 file of clustered SNR peaks (template_id, end_time, snr)"
    )
    parser.add_argument(
        "--channel",
        default=None,
        help="Channel label used only for the plot title"
    )
    parser.add_argument(
        "--output-directory",
        required=True,
        help="Directory in which to save the plot"
    )

    return parser.parse_args()


def load_peaks(trigger_file):
    """Load the clustered peaks from a matched-filter trigger file."""
    with h5py.File(trigger_file, 'r') as f:
        template_id = np.asarray(f["template_id"][:])
        end_time = np.asarray(f["end_time"][:], dtype=np.float64)
        snr = np.asarray(f["snr"][:], dtype=np.float64)
    return template_id, end_time, snr


def make_plot(template_id, end_time, snr, channel):
    """Build the SNR-peak scatter figure."""
    fig, ax = plt.subplots(figsize=(10, 5))

    if len(snr) > 0:
        ax.scatter(
            end_time / 86400.0,
            snr,
            c=template_id,
            cmap='jet',
            alpha=0.3,
        )
        im = ax.collections[0]
        cb = plt.colorbar(im, ax=ax)
        cb.set_label("Template ID")

    title = "Matched-filter SNR peaks"
    if channel is not None:
        title += f" ({channel})"
    ax.set_title(title)
    ax.set_xlabel("Time (days)")
    ax.set_ylabel("SNR")
    return fig


def main():
    args = load_arguments()
    init_logging(args.verbose)
    logger.info("Starting")

    template_id, end_time, snr = load_peaks(args.trigger_file)
    logger.info("Loaded %d peaks from %s", len(snr), args.trigger_file)

    os.makedirs(args.output_directory, exist_ok=True)

    fig = make_plot(template_id, end_time, snr, args.channel)
    output_path = os.path.join(
        args.output_directory, "matched_filter_snr_peaks.png"
    )
    fig.savefig(output_path, bbox_inches='tight')
    plt.close(fig)
    logger.info("Saved plot to %s", output_path)

    logger.info("Done!")


if __name__ == "__main__":
    main()
