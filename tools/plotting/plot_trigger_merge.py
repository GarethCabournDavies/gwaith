"""Plot merged triggers from a trigger-merge file.

This tool loads an HDF5 file of per-template clustered triggers (e.g. produced
by ``hwilas_trigger_merge``) and produces a scatter plot of the merged triggers
(SNR vs time in days), coloured by template ID.

Example::

    python tools/plotting/plot_trigger_merge.py \
        --trigger-file output/trigger_merge/trigger_merge_A.hdf \
        --channel LISA_A \
        --output-directory plots/trigger_merge
"""

import argparse
import logging

import h5py
import numpy as np

from matplotlib import use

use('Agg')
from matplotlib import pyplot as plt

from hwilas import add_hwilas_common_options, add_truth_arguments, init_logging, load_truth_file

logger = logging.getLogger(__name__)


def load_arguments():
    parser = argparse.ArgumentParser(
        description="Plot merged triggers from a trigger-merge file."
    )
    add_hwilas_common_options(parser)
    add_truth_arguments(parser)

    parser.add_argument(
        "--trigger-file",
        required=True,
        help="HDF5 file of merged triggers (template_id, end_time, snr)"
    )
    parser.add_argument(
        "--channel",
        default=None,
        help="Channel label used only for the plot title"
    )
    parser.add_argument(
        "--output",
        required=True,
        help="Filename in which to save the plot"
    )

    return parser.parse_args()


def load_triggers(trigger_file):
    """Load the merged triggers from a trigger-merge file."""
    with h5py.File(trigger_file, 'r') as f:
        template_id = np.asarray(f["template_id"][:])
        end_time = np.asarray(f["end_time"][:], dtype=np.float64)
        snr = np.asarray(f["snr"][:], dtype=np.float64)
    return template_id, end_time, snr


def make_plot(template_id, end_time, snr, channel, truth=None):
    """Build the merged-trigger scatter figure."""
    fig, ax = plt.subplots(figsize=(10, 5))

    if len(snr) > 0:
        ax.scatter(
            end_time / 86400.0,
            snr,
            c=template_id,
            cmap='jet',
            alpha=0.1,
        )
        im = ax.collections[0]
        cb = plt.colorbar(im, ax=ax)
        cb.set_label("Template ID")

    title = "Merged triggers"
    if channel is not None:
        title += f" ({channel})"
    ax.set_title(title)
    ax.set_xlabel("Time (days)")
    ax.set_ylabel("SNR")
    if truth is not None:
        _plot_truth_overlay(ax, truth)
    return fig


def _plot_truth_overlay(ax, truth):
    """Draw vertical lines at each injected-signal coalescence time on ``ax``.

    The x-axis is expected to be in days (as used by the trigger-merge plot), so
    the coalescence times (stored in seconds) are converted before plotting.
    """
    tc = truth.coalescence_times()
    if len(tc) == 0:
        return
    for time in tc:
        ax.axvline(time / 86400.0, color='r', linestyle='--', alpha=0.5,
                   zorder=0)
    handles = [plt.Line2D([0], [0], color='r', linestyle='--', label=f"Truth ({truth.source_group})")]
    ax.legend(handles=ax.get_legend_handles_labels()[0] + handles)


def main():
    args = load_arguments()
    init_logging(args.verbose)
    logger.info("Starting")

    template_id, end_time, snr = load_triggers(args.trigger_file)
    logger.info("Loaded %d triggers from %s", len(snr), args.trigger_file)

    truth = None
    if args.truth_file is not None:
        logger.info("Loading truth data from %s", args.truth_file)
        truth = load_truth_file(args.truth_file)
        logger.info(
            "Loaded %d %s truth source(s)", truth.n_sources, truth.source_group
        )

    fig = make_plot(template_id, end_time, snr, args.channel, truth=truth)

    fig.savefig(args.output, bbox_inches='tight')
    plt.close(fig)
    logger.info("Saved plot to %s", args.output)

    logger.info("Done!")


if __name__ == "__main__":
    main()
