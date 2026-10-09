"""Plot network-SNR coincidences and cumulative SNR counts.

Supply the clustered coincidence file with --input-file and the optional
pre-clustering file with --unclustered-input-file to show both populations.
Without the optional file, the input is clustered again using --cluster-window.
Truth coalescence times are drawn as full-height lines behind the data.
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
from hwilas.types import KeyValueAction

logger = logging.getLogger(__name__)


def load_arguments():
    parser = argparse.ArgumentParser(
        description="Plot network-SNR coincidences and per-channel foreground/background rates."
    )
    add_hwilas_common_options(parser)
    add_truth_arguments(parser)

    parser.add_argument(
        "--input-file",
        required=True,
        help="HDF5 file of coincident triggers (template_id, time_<ch>, snr_<ch>)"
    )
    parser.add_argument("--unclustered-input-file",
                        help="Optional pre-clustering coincidence file for the background points")
    parser.add_argument("--trigger-merge-files", nargs='+', action=KeyValueAction,
                        help="Merged triggers as channel:filename pairs, to plot foreground-vetoed background rates")
    parser.add_argument("--foreground-veto-window", type=float, default=86400.,
                        help="Half-width in seconds to veto around clustered coincidences; default 86400 (±1 day)")
    parser.add_argument("--data-file", help="Original data file for exposure when older triggers lack metadata")
    parser.add_argument("--end-exclusion",type=float,default=0.,
                        help="Minimum end exclusion for the exposure fallback")
    parser.add_argument("--edge-exclusion", type=float, default=0.,
                        help="Seconds excluded at each data boundary for the exposure fallback")
    parser.add_argument(
        "--channels",
        nargs='+',
        required=True,
        help="The two channels that were used for the coincidence search"
    )
    parser.add_argument(
        "--cluster-window", "--coincidence-window", dest="coincidence_window",
        type=float,
        default=86400.0,
        help="Event clustering window in seconds; default 86400 (1 day). Must match the value "
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
    if args.trigger_merge_files:
        if set(args.trigger_merge_files) != set(args.channels):
            parser.error("Provide a merged trigger file for each plotted channel")
    return args


def load_coincidences(input_file, channels):
    """Load coincident triggers for the given pair of channels."""
    coincs = {}
    with h5py.File(input_file, 'r') as f:
        coincs["template_id"] = np.asarray(f["template_id"][:])
        if "in_previous_window" in f:
            coincs["in_previous_window"] = f["in_previous_window"][:]
        for channel in channels:
            if f"trigger_id_{channel}" in f:
                coincs[f"trigger_id_{channel}"] = f[f"trigger_id_{channel}"][:]
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


def exclude_clustered_coincidences(coincs, clustered, channels):
    """Remove selected rows so the two plotted populations are disjoint."""
    fields = ['template_id', *(f'trigger_id_{c}' for c in channels)]
    if not all(field in coincs and field in clustered for field in fields):
        fields = ['template_id', *(f'time_{c}' for c in channels)]
    raw_keys = np.rec.fromarrays([coincs[field] for field in fields])
    selected_keys = np.rec.fromarrays([clustered[field] for field in fields])
    keep = ~np.isin(raw_keys, selected_keys)
    return {key:values[keep] for key, values in coincs.items()}


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
    for time in tc:
        ax.axvline(time / 86400.0, color='r', linestyle='--', alpha=0.5,
                   zorder=0)
    handles = [plt.Line2D([0], [0], color='r', linestyle='--', label=f"Truth ({truth.source_group})")]
    ax.legend(handles=ax.get_legend_handles_labels()[0] + handles)


def plot_coincident_triggers(coincs, clustered, channels, mbhb_times, expected_snrs, output_path,
                             truth=None):
    """Build and save the coincident-triggers scatter plot."""
    ch0, ch1 = channels
    template_ids = np.concatenate([coincs["template_id"], clustered["template_id"]])
    cnorm = plt.Normalize(
        float(template_ids.min()) if len(template_ids) else 0.,
        float(template_ids.max()) if len(template_ids) else 1.,
    )

    fig, ax = plt.subplots(figsize=(10, 4))

    if len(expected_snrs):
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

    all_times = (coincs[f"time_{ch0}"]+coincs[f"time_{ch1}"])/2
    all_snr = np.hypot(coincs[f"snr_{ch0}"],coincs[f"snr_{ch1}"])
    im = ax.scatter(all_times/86400., all_snr, c=coincs['template_id'],
                    norm=cnorm, cmap='jet', alpha=.2, s=15, marker='o',
                    label='Unclustered coincidences', zorder=2)
    clustered_times = (clustered[f"time_{ch0}"]+clustered[f"time_{ch1}"])/2
    clustered_snr = np.hypot(clustered[f"snr_{ch0}"],clustered[f"snr_{ch1}"])
    ax.scatter(clustered_times/86400., clustered_snr, c=clustered['template_id'],
               norm=cnorm, cmap='jet', edgecolors='k', s=85, marker='*',
               label='Clustered coincidences', zorder=3)

    flags = clustered.get('in_previous_window')
    if flags is not None and np.any(flags):
        ax.scatter(clustered_times[flags]/86400., clustered_snr[flags], facecolors='none',
                   edgecolors='darkorange', s=140, linewidths=1.5, zorder=4,
                   label='Inside first-round optimisation window')
    ax.legend(loc='upper right')

    if truth is not None:
        _plot_truth_overlay(ax, truth)

    cb = plt.colorbar(im, ax=ax)
    cb.set_label("Template ID")

    ax.set_xlabel("Time (days)")
    ax.set_ylabel("Network SNR")

    fig.savefig(output_path, bbox_inches='tight')
    plt.close(fig)
    logger.info("Saved coincident-triggers plot to %s", output_path)


def merge_intervals(intervals):
    """Union half-open intervals so overlapping vetoes count once."""
    merged = []
    for start, end in sorted(intervals, key=lambda interval: (interval[0], interval[1])):
        if end <= start:
            continue
        if merged and start <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], end)
        else:
            merged.append([float(start), float(end)])
    return np.asarray(merged, dtype=float).reshape(-1, 2)


def in_intervals(times, intervals):
    """Select times in a sorted union of half-open intervals."""
    times = np.asarray(times)
    if not len(intervals):
        return np.zeros(len(times), dtype=bool)
    index = np.searchsorted(intervals[:, 0], times, side='right')-1
    return (index >= 0) & (times < intervals[np.maximum(index, 0), 1])


def load_background_snrs(clustered, trigger_files, channels, half_width,
                         data_file=None, edge_exclusion=0., coincs=None, end_exclusion=0.):
    """Veto foreground windows and measure the remaining searched exposure."""
    if half_width < 0 or edge_exclusion < 0:
        raise ValueError('Veto width and edge exclusion must be nonnegative')
    merged = {}
    searched = None
    fallback = None
    if data_file:
        with h5py.File(data_file, 'r') as f:
            start = float(f.attrs.get('analysis_start_time', f.attrs['epoch']))
            end = float(f.attrs.get('analysis_end_time',
                        f.attrs['epoch']+len(f[channels[0]])*f.attrs['delta_t']
                        -f.attrs.get('padding_duration', 0.)))
        fallback = merge_intervals([[start+edge_exclusion, end-end_exclusion-edge_exclusion]])
    for channel in channels:
        with h5py.File(trigger_files[channel], 'r') as f:
            merged[channel] = (f['end_time'][:], f['snr'][:])
            intervals = f.attrs.get('analysis_intervals', fallback)
        if intervals is None:
            raise ValueError('Searched exposure is missing; provide --data-file and --edge-exclusion for older trigger files')
        intervals = merge_intervals(intervals)
        if not len(intervals):
            raise ValueError('No searched time remains')
        if searched is not None and (searched.shape != intervals.shape or not np.allclose(searched, intervals, rtol=0, atol=1e-8)):
            raise ValueError('A/E merged files must describe the same searched intervals')
        searched = intervals
    times = (clustered[f'time_{channels[0]}']+clustered[f'time_{channels[1]}'])/2
    veto = merge_intervals([[max(time-half_width, start), min(time+half_width, end)]
                           for time in times for start, end in searched])
    total_exposure = float(np.sum(searched[:, 1]-searched[:, 0]))
    background_exposure = total_exposure-float(np.sum(veto[:, 1]-veto[:, 0]))
    if background_exposure <= 0:
        raise ValueError('Foreground veto leaves no background observing time')
    background = {}
    for channel, (trigger_times, snrs) in merged.items():
        keep = in_intervals(trigger_times, searched) & ~in_intervals(trigger_times, veto)
        if coincs is not None:
            ids = np.unique(coincs[f'trigger_id_{channel}'])
            if np.any(ids < 0) or np.any(ids >= len(keep)):
                raise ValueError(f'Coincidence IDs do not match the {channel} merged file')
            keep[ids] = False
        background[channel] = snrs[keep]
        logger.info('%s: %d background triggers over %.6f days',
                    channel, keep.sum(), background_exposure/86400.)
    logger.info('Searched %.6f days; vetoed %.6f days around clustered coincidences',
                total_exposure/86400., (total_exposure-background_exposure)/86400.)
    return background, total_exposure, background_exposure


def plot_snr_histogram(clustered, channels, expected_snrs, output_path,
                       background_snrs=None,
                       total_exposure=None, background_exposure=None):
    """Compare clustered foreground and time-vetoed background by channel."""
    fig, ax = plt.subplots()

    def cumulative(snrs, label, exposure, **style):
        snrs = np.asarray(snrs)
        snrs = np.sort(snrs[np.isfinite(snrs) & (snrs > 0)])
        if len(snrs):
            values, counts = np.unique(snrs, return_counts=True)
            rates = np.cumsum(counts[::-1])[::-1]
            if exposure is not None:
                if exposure <= 0:
                    raise ValueError("Exposure must be positive")
                rates = rates/(exposure/86400.)
            ax.step(values, rates, where='pre',
                    label=label, **style)

    for index, channel in enumerate(channels):
        color = f'C{index}'
        cumulative(clustered[f'snr_{channel}'], f'{channel} foreground',
                   total_exposure, color=color)
        if background_snrs is not None and channel in background_snrs:
            cumulative(background_snrs[channel], f'{channel} background',
                       background_exposure, color=color, linestyle='--')
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel("Channel SNR")
    ax.set_ylabel("Triggers per day with SNR ≥ threshold" if total_exposure is not None
                  else "Count with SNR ≥ threshold")
    ax.legend(loc='upper right')
    fig.savefig(output_path, bbox_inches='tight')
    plt.close(fig)
    logger.info("Saved SNR histogram to %s", output_path)


def main():
    args = load_arguments()
    init_logging(args.verbose)
    logger.info("Starting")

    channels = args.channels
    coincs = load_coincidences(args.unclustered_input_file or args.input_file, channels)
    logger.info(
        "Loaded %d coincidences from %s", len(coincs["template_id"]), args.input_file
    )

    clustered = (load_coincidences(args.input_file, channels) if args.unclustered_input_file
                 else cluster_coincidences(coincs, channels, args.coincidence_window))
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

    unclustered = exclude_clustered_coincidences(coincs, clustered, channels)
    logger.info("%d nonselected coincidences remain for plotting", len(unclustered["template_id"]))
    plot_coincident_triggers(
        unclustered,
        clustered,
        channels,
        mbhb_times,
        expected_snrs,
        os.path.join(args.output_directory, "coincident_triggers.png"),
        truth=truth,
    )
    background_snrs, total_exposure, background_exposure = None, None, None
    if args.trigger_merge_files:
        background_snrs, total_exposure, background_exposure = load_background_snrs(
            clustered, args.trigger_merge_files, channels, args.foreground_veto_window,
            args.data_file, args.edge_exclusion, coincs=coincs, end_exclusion=args.end_exclusion)
    plot_snr_histogram(
        clustered,
        channels,
        expected_snrs,
        os.path.join(args.output_directory, "snr_histogram.png"),
        background_snrs=background_snrs,
        total_exposure=total_exposure, background_exposure=background_exposure,
    )

    logger.info("Done!")


if __name__ == "__main__":
    main()
