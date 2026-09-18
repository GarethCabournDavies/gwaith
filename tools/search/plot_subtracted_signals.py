"""
Plot the original data, the subtracted data and the subtracted signal.

This script is intended to be run after `gwaith_subtract_signals`. It reads:

* The original AET data file (e.g. ``output/round_0/data_processing/sangria_aet.hdf``)
* The optimised signal timeseries written by ``hwilas_optimise_snr_peaks``
  (e.g. ``output/round_0/optimise/optimise.hdf``)
* The subtracted data file written by ``gwaith_subtract_signals``
  (e.g. ``output/round_0/subtract/sangria_data_subtracted.hdf``)
* The optimised parameter file written by ``hwilas_optimise_snr_peaks``
  (e.g. ``output/round_0/optimise/optimise_parameters.hdf``)

It produces:

* One full-length plot of the original data, signal and residual for all
  channels on a single axis, with the x-axis limited to the data limits. The
  original data is drawn as a faded background, with the signal (dashed) and
  the residual drawn on top.
* One zoomed-in plot per event reusing the same figure, with only the x-axis
  limited to ``event_time - window_before`` through ``event_time +
  window_after``, where the windows are configurable (default 0.2 days each).
"""

import argparse
import logging
import os

import h5py

from matplotlib import use


use('Agg')
from matplotlib import pyplot as plt

from gwaith.types import TimeSeries

from hwilas import add_hwilas_common_options, init_logging, load_truth_file

logger = logging.getLogger(__name__)

CHANNEL_COLORS = {
    'LISA_A': 'tab:blue',
    'LISA_E': 'tab:orange',
    'LISA_T': 'tab:green',
}


def parse_event_window_value(value):
    """argparse type converter for ``--event-window``.

    Accepts either a single float (e.g. ``0.1``) or a comma-separated pair of
    floats (e.g. ``0.15,0.05``) and returns the corresponding list of floats.
    """
    values = [float(v) for v in value.split(',')]
    if not 1 <= len(values) <= 2:
        raise argparse.ArgumentTypeError(
            f"Expected one or two comma-separated float values, got '{value}'"
        )
    return values


def parse_event_window(window):
    """Parse the ``--event-window`` argument into a (before, after) tuple.

    A single value gives a symmetric window; two values are interpreted as
    ``(window_before, window_after)`` in days. Both values must be positive.
    """
    if len(window) == 1:
        before = after = window[0]
    elif len(window) == 2:
        before, after = window
    else:
        raise ValueError(
            f"Expected one or two values for --event-window, got {len(window)}: "
            f"{window}"
        )

    if before <= 0 or after <= 0:
        raise ValueError(
            f"--event-window values must be positive, got ({before}, {after})"
        )

    return float(before), float(after)


def load_arguments():
    parser = argparse.ArgumentParser(
        description="Plot the original data, subtracted data and subtracted signal."
    )
    add_hwilas_common_options(parser)

    parser.add_argument(
        "--data-file",
        required=True,
        help="HDF5 file containing the original AET data "
             "(e.g. output/round_0/data_processing/sangria_aet.hdf)"
    )
    parser.add_argument(
        "--truth-file",
        default=None,
        help="Standardised truth HDF5 file (see tools/data_processing/extract_truth_*) "
             "from which to read the summed AET truth waveform for overlay."
    )
    parser.add_argument(
        "--signal-file",
        required=True,
        help="HDF5 file containing the optimised signal timeseries written by "
             "hwilas_optimise_snr_peaks "
             "(e.g. output/round_0/optimise/optimise.hdf)"
    )
    parser.add_argument(
        "--subtracted-file",
        required=True,
        help="HDF5 file containing the subtracted data written by "
             "gwaith_subtract_signals "
             "(e.g. output/round_0/subtract/sangria_data_subtracted.hdf)"
    )
    parser.add_argument(
        "--parameter-file",
        required=True,
        help="HDF5 file containing the optimised event parameters written by "
             "hwilas_optimise_snr_peaks "
             "(e.g. output/round_0/optimise/optimise_parameters.hdf)"
    )
    parser.add_argument(
        "--output-directory",
        required=True,
        help="Directory in which to save the plots"
    )
    parser.add_argument(
        "--channels",
        nargs='+',
        default=['LISA_A', 'LISA_E', 'LISA_T'],
        help="Channels to plot. Default: LISA_A LISA_E LISA_T"
    )
    parser.add_argument(
        "--event-window",
        type=parse_event_window_value,
        default=[0.2],
        help="Window in days around each event time to include in the zoomed "
             "plot (i.e. event_time - window_before to event_time + "
             "window_after). Give either one value for a symmetric window or "
             "two comma-separated values as before,after (e.g. 0.15,0.05). "
             "Default: 0.2 days"
    )

    return parser.parse_args()


def load_timeseries(hdf_file, channel):
    """Load a TimeSeries from an HDF5 file for the given channel."""
    with h5py.File(hdf_file, 'r') as f:
        epoch = float(f.attrs['epoch'])
        delta_t = float(f.attrs['delta_t'])
        data = f[channel][:]
    return TimeSeries(data=data, epoch=epoch, delta_t=delta_t)


def make_figure(
    ts_original,
    ts_subtracted,
    ts_signal,
    ts_truth,
    channels,
):
    """Build a full-length figure of the original data, signal and residual.

    All channels are drawn on a single axis, with the x-axis limited to the
    data limits. The original data is drawn as a faded background (alpha 0.2);
    the signal (dashed) and the residual (the subtracted data) are drawn on
    top. Each channel keeps its own colour.

    Returns the figure and its single axis so that event views can be produced
    by re-limiting the x-axis of the same figure.
    """
    fig, ax = plt.subplots(figsize=(14, 5))

    # Determine the overall x limits from the data so all channels share the
    # same time range.
    t_min = min(
        ts_original[ch].sample_times[0] for ch in channels
    ) / 86400.0
    t_max = max(
        ts_original[ch].sample_times[-1] for ch in channels
    ) / 86400.0

    for channel in channels:
        color = CHANNEL_COLORS.get(channel, 'black')
        t_days = ts_original[channel].sample_times / 86400.0

        # Original data as a faded background.
        # ax.plot(
        #     t_days, ts_original[channel].data,
        #     label=f"Data ({channel})",
        #     color=color,
        #     alpha=0.5,
        # )
        if ts_truth is not None:
            ax.plot(
                t_days, ts_truth[channel].data,
                label=f"Signal Truth ({channel})",
                color=color,
                alpha=1.0, 
                linestyle='--',
            )
        # # The subtracted signal, drawn on top of the background.
        ax.plot(
            t_days, ts_signal[channel].data,
            label=f"signal ({channel})",
            color=color,
            alpha=1.0, 
            linestyle=':',
        )
        # The residual (subtracted data), drawn on top of the signal.
        # ax.plot(
        #     t_days, ts_subtracted[channel].data,
        #     label=f"residual ({channel})",
        #     color=color,
        #     alpha=0.2,
        # )

    ax.set_xlim(t_min, t_max)
    ax.set_xlabel("Time (days)")
    ax.set_ylabel("Amplitude")
    ax.legend(loc='upper left')
    return fig, ax


def plot_full(
    ts_original,
    ts_subtracted,
    ts_signal,
    ts_truth,
    channels,
    output_path,
):
    """Create and save a full-length plot of the original data, signal and
    residual for all channels on a single axis."""
    fig, ax = make_figure(ts_original, ts_subtracted, ts_signal, ts_truth, channels)
    ax.set_title(f"Full data: {', '.join(channels)}")
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def save_event_view(
    fig,
    ax,
    event_time,
    window_before_seconds,
    window_after_seconds,
    channels,
    output_path,
):
    """Save a zoomed-in view of an already-built figure around a single event.

    The same full-length figure is reused; only the x-axis limits are changed
    to ``event_time - window_before`` through ``event_time + window_after`` and
    the title is updated before saving.
    """
    ax.set_xlim(
        (event_time - window_before_seconds) / 86400.0,
        (event_time + window_after_seconds) / 86400.0,
    )
    ax.set_title(f"Event at {event_time / 86400.0:.3f} days: {', '.join(channels)}")
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)


def main():
    args = load_arguments()
    init_logging(args.verbose)

    os.makedirs(args.output_directory, exist_ok=True)

    logger.info("Loading original data from %s", args.data_file)
    ts_original = {
        ch: load_timeseries(args.data_file, ch) for ch in args.channels
    }

    logger.info("Loading signal timeseries from %s", args.signal_file)
    ts_signal = {
        ch: load_timeseries(args.signal_file, ch) for ch in args.channels
    }

    logger.info("Loading subtracted data from %s", args.subtracted_file)
    ts_subtracted = {
        ch: load_timeseries(args.subtracted_file, ch) for ch in args.channels
    }

    if args.truth_file is not None:
        logger.info("Loading truth data from %s", args.truth_file)
        truth = load_truth_file(args.truth_file)
        ts_truth = {
            ch: TimeSeries(
                truth.timeseries[ch],
                delta_t=truth.delta_t,
                epoch=truth.epoch,
            )
            for ch in args.channels
            if truth.timeseries is not None and ch in truth.timeseries
        }
        if not ts_truth:
            logger.warning(
                "Truth file %s has no AET timeseries for the requested channels; "
                "skipping truth overlay.", args.truth_file,
            )
    else:
        ts_truth = None

    # Load event times from the parameter file
    logger.info("Loading event parameters from %s", args.parameter_file)
    with h5py.File(args.parameter_file, 'r') as f:
        event_times = f['tc'][:]

    logger.info("Found %d events to plot", len(event_times))

    # Check that all three timeseries have the same length and epoch for each
    # channel.
    for channel in args.channels:
        if not (
            len(ts_original[channel]) == len(ts_subtracted[channel]) == len(ts_signal[channel])
        ):
            logger.warning(
                "Timeseries lengths differ for channel %s: original=%d, subtracted=%d, signal=%d",
                channel,
                len(ts_original[channel]),
                len(ts_subtracted[channel]),
                len(ts_signal[channel]),
            )

    # Build the full-length figure once with all channels on one axis.
    fig, ax = make_figure(
        ts_original,
        ts_subtracted,
        ts_signal,
        ts_truth,
        args.channels,
    )

    # Save the full-length plot.
    full_plot_path = os.path.join(args.output_directory, "full.png")
    logger.info("Saving combined full plot to %s", full_plot_path)
    ax.set_title(f"Full data: {', '.join(args.channels)}")
    fig.tight_layout()
    fig.savefig(full_plot_path, dpi=150)

    # Save one zoomed view per event by re-limiting the x-axis of the same
    # figure.
    window_before_days, window_after_days = parse_event_window(args.event_window)
    window_before_seconds = window_before_days * 86400.0
    window_after_seconds = window_after_days * 86400.0
    logger.info(
        "Event zoom windows: %.3f days before and %.3f days after each event",
        window_before_days,
        window_after_days,
    )
    for idx, event_time in enumerate(event_times):
        event_plot_path = os.path.join(
            args.output_directory,
            f"event_{idx}.png",
        )
        logger.info(
            "Saving event %d plot (t=%.3f days) to %s",
            idx, float(event_time) / 86400.0, event_plot_path,
        )
        save_event_view(
            fig,
            ax,
            float(event_time),
            window_before_seconds,
            window_after_seconds,
            args.channels,
            event_plot_path,
        )

    plt.close(fig)
    logger.info("Done!")


if __name__ == "__main__":
    main()