"""Plot coincident triggers and the network-statistic histogram.

This tool loads an HDF5 file of coincident triggers (e.g. produced by
``hwilas_coincidence``) and reproduces the two plots that were previously
generated inline:

* ``coincident_triggers.png`` -- a scatter of per-channel and network
  coincidences, with the clustered (loudest-per-window) events marked as stars.
* ``snr_histogram.png`` -- a cumulative histogram of the clustered network
  statistic compared against the expected SNR of injected signals.

The coincidence file stores *all* raw coincidences, so this tool re-applies the
same greedy loudest-first clustering (using ``--coincidence-window``) to recover
the clustered events for plotting.

Example::

    python tools/plotting/plot_coincidence.py \
        --input-file output/coincidence/coincidence.hdf \
        --channels LISA_A LISA_E \
        --coincidence-window 3600 \
        --expected-snr-file optimal_snr/optimal_snr_fitting_factors.hdf \
        --output-directory plots/coincidence
"""

import argparse
import logging
import os

import h5py
import numpy as np

from matplotlib import use

use('Agg')
from matplotlib import pyplot as plt

from hwilas import add_hwilas_common_options, add_truth_arguments, init_logging, load_truth_file

logger = logging.getLogger(__name__)


def load_arguments():
    parser = argparse.ArgumentParser(
        description="Plot coincident triggers and the network-statistic histogram."
    )
    add_hwilas_common_options(parser)
    add_truth_arguments(parser)

    parser.add_argument(
        "--input-file",
        required=True,
        help="HDF5 file of coincident triggers (template_id, time_<ch>, snr_<ch>)"
    )
    parser.add_argument(
        "--channels",
        nargs='+',
        required=True,
        help="The two channels that were used for the coincidence search"
    )
    parser.add_argument(
        "--coincidence-window",
        type=float,
        default=3600.0,
        help="Coincidence/clustering window in seconds. Must match the value "
             "used by hwilas_coincidence to reproduce the clustered events."
    )
    parser.add_argument(
        "--expected-snr-file",
        default=None,
        help="Optional HDF5 file with mbhb_time and optimal_snr*fitting_factor "
             "for plotting expected signal SNRs"
    )
    parser.add_argument(
        "--output-directory",
        required=True,
        help="Directory in which to save the plots"
    )

    args = parser.parse_args()
    if len(args.channels) != 2:
        parser.error("--channels must specify exactly two channels")
    return args


def load_coincidences(input_file, channels):
    """Load coincident triggers for the given pair of channels."""
    coincs = {}
    with h5py.File(input_file, 'r') as f:
        coincs["template_id"] = np.asarray(f["template_id"][:])
        for channel in channels:
            coincs[f"time_{channel}"] = np.asarray(
                f[f"time_{channel}"][:], dtype=np.float64
            )
            coincs[f"snr_{channel}"] = np.asarray(
                f[f"snr_{channel}"][:], dtype=np.float64
            )
    return coincs


def cluster_coincidences(coincs, channels, coincidence_window):
    """Re-apply greedy loudest-first clustering to the raw coincidences."""
    ch0, ch1 = channels
    coinc_time = (coincs[f"time_{ch0}"] + coincs[f"time_{ch1}"]) / 2.0
    coinc_stat = np.sqrt(
        coincs[f"snr_{ch0}"] ** 2 + coincs[f"snr_{ch1}"] ** 2
    )

    n_coincs = len(coinc_time)
    indices = np.argsort(coinc_stat)[::-1]
    keep = np.ones(n_coincs, dtype=bool)
    for sorted_coinc_idx in range(n_coincs):
        coinc_idx = indices[sorted_coinc_idx]
        if not keep[coinc_idx]:
            continue
        within_window = np.abs(coinc_time - coinc_time[coinc_idx]) < coincidence_window
        keep[within_window] = False
        keep[coinc_idx] = True

    clustered = {key: value[keep] for key, value in coincs.items()}
    return clustered


def load_expected_snr(expected_snr_file):
    """Load expected MBHB times and SNRs, or empty arrays if not provided."""
    if expected_snr_file is None:
        return np.array([]), np.array([])
    with h5py.File(expected_snr_file, 'r') as f:
        mbhb_times = np.asarray(f["mbhb_time"][:], dtype=np.float64)
        expected_snrs = (
            np.asarray(f["optimal_snr"][:]) * np.asarray(f["fitting_factor"][:])
        )
    return mbhb_times, expected_snrs


def _plot_truth_overlay(ax, truth):
    """Draw vertical lines at each injected-signal coalescence time on ``ax``.

    The x-axis is expected to be in days (as used by the coincidence plot), so
    the coalescence times (stored in seconds) are converted before plotting.
    """
    tc = truth.coalescence_times()
    if len(tc) == 0:
        return
    ax.vlines(
        tc / 86400.0,
        ax.get_ylim()[0],
        ax.get_ylim()[1],
        color='r',
        linestyle='--',
        alpha=0.5,
        zorder=999,
    )
    handles = [plt.Line2D([0], [0], color='r', linestyle='--', label=f"Truth ({truth.source_group})")]
    ax.legend(handles=ax.get_legend_handles_labels()[0] + handles)


def plot_coincident_triggers(coincs, clustered, channels, mbhb_times, expected_snrs, output_path,
                             truth=None):
    """Build and save the coincident-triggers scatter plot."""
    ch0, ch1 = channels
    cnorm = plt.Normalize(
        float(coincs["template_id"].min()),
        float(coincs["template_id"].max()),
    )

    fig, ax = plt.subplots(figsize=(10, 4))

    ax.scatter(
        mbhb_times / 86400.0,
        expected_snrs,
        marker='x',
        c='red',
        label="MBHB expected SNR for this bank",
        zorder=100,
    )
    for mbhb_time in mbhb_times:
        ax.axvline(mbhb_time / 86400.0, c='tab:pink', linestyle=':')

    ax.scatter(
        coincs[f"time_{ch0}"] / 86400.0,
        coincs[f"snr_{ch0}"],
        marker='^',
        c=coincs["template_id"],
        norm=cnorm,
        cmap='jet',
        alpha=0.1,
    )
    ax.scatter(
        coincs[f"time_{ch1}"] / 86400.0,
        coincs[f"snr_{ch1}"],
        marker='v',
        c=coincs["template_id"],
        norm=cnorm,
        cmap='jet',
        alpha=0.1,
    )

    coinc_times_all = (
        coincs[f"time_{ch0}"] + coincs[f"time_{ch1}"]
    ) / 2.0
    coinc_stat_all = np.sqrt(
        coincs[f"snr_{ch0}"] ** 2 + coincs[f"snr_{ch1}"] ** 2
    )

    im = ax.scatter(
        coinc_times_all / 86400.0,
        coinc_stat_all,
        marker='o',
        c=coincs["template_id"],
        norm=cnorm,
        cmap='jet',
        alpha=0.1,
    )

    coinc_time_clustered = (
        clustered[f"time_{ch0}"] + clustered[f"time_{ch1}"]
    ) / 2.0
    coinc_stat_clustered = np.sqrt(
        clustered[f"snr_{ch0}"] ** 2 + clustered[f"snr_{ch1}"] ** 2
    )

    ax.scatter(
        coinc_time_clustered / 86400.0,
        coinc_stat_clustered,
        marker='*',
        edgecolors='k',
        c='y',
        zorder=1000,
        s=100,
    )

    ax.scatter([], [], label="Network", c='k', marker='o')
    ax.scatter([], [], label=ch0, c='k', marker='^')
    ax.scatter([], [], label=ch1, c='k', marker='v')
    ax.scatter([], [], label="Clustered", c='y', marker='*', edgecolors='k')
    ax.legend(loc='upper right')

    if truth is not None:
        _plot_truth_overlay(ax, truth)

    cb = plt.colorbar(im, ax=ax)
    cb.set_label("Template ID")

    ax.set_xlabel("Time (days)")
    ax.set_ylabel("SNR")

    fig.savefig(output_path, bbox_inches='tight')
    plt.close(fig)
    logger.info("Saved coincident-triggers plot to %s", output_path)


def plot_snr_histogram(clustered, channels, expected_snrs, output_path):
    """Build and save the cumulative network-statistic histogram."""
    ch0, ch1 = channels
    coinc_stat_clustered = np.sqrt(
        clustered[f"snr_{ch0}"] ** 2 + clustered[f"snr_{ch1}"] ** 2
    )

    fig, ax = plt.subplots()
    bins = np.logspace(
        np.log10(coinc_stat_clustered.min()),
        np.log10(coinc_stat_clustered.max()),
        100,
    )
    hist_snr, bins = np.histogram(coinc_stat_clustered, bins=bins)
    cumhist = np.cumsum(hist_snr[::-1])[::-1]
    ax.step(
        bins[:-1],
        cumhist,
        label="Clustered Coincs Statistic",
    )
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel("SNR")
    ax.set_ylabel("Count of louder events (per year)")

    sort_expected = np.argsort(expected_snrs)
    ax.step(
        expected_snrs[sort_expected],
        np.arange(len(expected_snrs))[::-1] + 1,
        label="Expected statistic for Signals",
    )
    ax.legend(loc='upper right')

    fig.savefig(output_path, bbox_inches='tight')
    plt.close(fig)
    logger.info("Saved SNR histogram to %s", output_path)


def main():
    args = load_arguments()
    init_logging(args.verbose)
    logger.info("Starting")

    channels = args.channels
    coincs = load_coincidences(args.input_file, channels)
    logger.info(
        "Loaded %d coincidences from %s", len(coincs["template_id"]), args.input_file
    )

    clustered = cluster_coincidences(coincs, channels, args.coincidence_window)
    logger.info("%d coincidences remain after clustering", len(clustered["template_id"]))

    mbhb_times, expected_snrs = load_expected_snr(args.expected_snr_file)

    truth = None
    if args.truth_file is not None:
        logger.info("Loading truth data from %s", args.truth_file)
        truth = load_truth_file(args.truth_file)
        logger.info(
            "Loaded %d %s truth source(s)", truth.n_sources, truth.source_group
        )

    os.makedirs(args.output_directory, exist_ok=True)

    plot_coincident_triggers(
        coincs,
        clustered,
        channels,
        mbhb_times,
        expected_snrs,
        os.path.join(args.output_directory, "coincident_triggers.png"),
        truth=truth,
    )
    plot_snr_histogram(
        clustered,
        channels,
        expected_snrs,
        os.path.join(args.output_directory, "snr_histogram.png"),
    )

    logger.info("Done!")


if __name__ == "__main__":
    main()
