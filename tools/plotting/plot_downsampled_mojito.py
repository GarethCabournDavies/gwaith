"""Plot downsampled + bandpassed Mojito data.

This tool loads an HDF5 file containing downsampled and bandpassed Mojito
timeseries (e.g. produced by ``hwilas_downsample_mojito``) and produces:

* One full-length plot of the data for all channels on a single axis, with the
  x-axis in days relative to the epoch.
* Optionally, one zoomed-in plot per window specified via ``--zoom-windows``,
  where each window is given as ``start_days,end_days`` (relative to epoch).

Example::

    python tools/plotting/plot_downsampled_mojito.py \
        --data-file output/mojito_downsampled.hdf \
        --output-directory plots/mojito \
        --channels LISA_A LISA_E LISA_T \
        --zoom-windows 0.5,1.5 2.0,3.0
"""

import argparse
import logging
import os

import h5py
import numpy as np

from matplotlib import use

use('Agg')
from matplotlib import pyplot as plt

from gwaith.types import TimeSeries

from hwilas import add_hwilas_common_options, add_truth_arguments, init_logging, load_truth_file

logger = logging.getLogger(__name__)

CHANNEL_COLORS = {
    'LISA_A': 'tab:blue',
    'LISA_E': 'tab:orange',
    'LISA_T': 'tab:green',
}


def parse_zoom_window(value):
    """argparse type converter for ``--zoom-windows``.

    Accepts a comma-separated pair of floats (e.g. ``0.5,1.5``) representing
    the start and end of a zoom window in days relative to the epoch.
    """
    values = [float(v) for v in value.split(',')]
    if len(values) != 2:
        raise argparse.ArgumentTypeError(
            f"Expected two comma-separated float values (start,end), got '{value}'"
        )
    if values[0] >= values[1]:
        raise argparse.ArgumentTypeError(
            f"Start must be less than end, got ({values[0]}, {values[1]})"
        )
    return tuple(values)


def load_arguments():
    parser = argparse.ArgumentParser(
        description="Plot downsampled + bandpassed Mojito data."
    )
    add_hwilas_common_options(parser)
    add_truth_arguments(parser)

    parser.add_argument(
        "--data-file",
        required=True,
        help="HDF5 file containing the downsampled/bandpassed timeseries "
             "(e.g. output/mojito_downsampled.hdf)"
    )
    output = parser.add_mutually_exclusive_group(required=True)
    output.add_argument("--output-file", help="Full data plot filename")
    output.add_argument(
        "--output-directory",
        help="Directory in which to save the plots"
    )
    parser.add_argument(
        "--channels",
        nargs='+',
        default=['LISA_A', 'LISA_E', 'LISA_T'],
        help="Channels to plot. Default: LISA_A LISA_E LISA_T"
    )
    parser.add_argument(
        "--zoom-windows",
        type=parse_zoom_window,
        nargs='+',
        default=None,
        help="Zoom windows in days relative to epoch, each given as "
             "start_days,end_days (e.g. 0.5,1.5). A separate plot is saved "
             "for each window."
    )
    parser.add_argument(
        "--decimate",
        type=int,
        default=10,
        help="Decimation factor for plotting (plot every Nth sample). "
            "Default: 10. Use 1 for no decimation."
    )
    

    parser.add_argument("--end-exclusion",type=float,default=0.,
                        help="Seconds to omit at the end of the plotted data")
    parser.add_argument("--removed-parameter-files", nargs='+',
                        help="Fitted parameter HDF5 files whose signals have already been subtracted")
    args = parser.parse_args()
    if args.decimate < 1:
        parser.error("Decimation must be positive")
    return args


def load_timeseries(hdf_file, channel, decimate=1, end_exclusion=0.):
    """Load a TimeSeries from an HDF5 file for the given channel."""
    with h5py.File(hdf_file, 'r') as f:
        epoch = float(f.attrs['epoch'])
        delta_t = float(f.attrs['delta_t'])
        stop = len(f[channel])-int(np.ceil(end_exclusion/delta_t))
        if stop<=0:
            raise ValueError("End exclusion removes all plotted data")
        data = f[channel][:stop:decimate]
    return TimeSeries(data=data, epoch=epoch, delta_t=delta_t*decimate)


def _plot_truth_overlay(ax, truth, epoch=0.):
    """Draw full-height truth markers behind data on an epoch-relative axis."""
    for i, time in enumerate(truth.coalescence_times()):
        ax.axvline((time-epoch)/86400., color='r', linestyle='--', alpha=.5,
                   zorder=0, label=f'Truth ({truth.source_group})' if i==0 else '_nolegend_')


def load_removed_times(paths):
    """Use the fitted coalescence times of models already removed from the data."""
    times = []
    for path in paths or []:
        with h5py.File(path, 'r') as f:
            times.extend(f['tc'][:])
    return np.asarray(times)


def make_full_plot(data, channels, truth=None, removed_times=None):
    """Build a full-length figure of the downsampled/bandpassed data.

    All channels are drawn on a single axis with the x-axis in days relative
    to the epoch.

    Returns the figure and its single axis so that zoom views can be produced
    by re-limiting the x-axis of the same figure.
    """
    fig, ax = plt.subplots(figsize=(14, 5))

    for channel in channels:
        color = CHANNEL_COLORS.get(channel, 'black')
        t_days = (data[channel].sample_times - data[channel].epoch) / 86400.0
        ax.plot(
            t_days, data[channel].data,
            label=channel,
            color=color,
        )

    t_min = min(
        (data[ch].sample_times[0] - data[ch].epoch) / 86400.0 for ch in channels
    )
    t_max = max(
        (data[ch].sample_times[-1] - data[ch].epoch) / 86400.0 for ch in channels
    )

    ax.set_xlim(t_min, t_max)
    ax.set_xlabel("Time (days)")
    ax.set_ylabel("Amplitude")
    if truth is not None:
        _plot_truth_overlay(ax, truth, epoch=float(data[channels[0]].epoch))
    for i, time in enumerate(removed_times if removed_times is not None else []):
        ax.axvline((time-float(data[channels[0]].epoch))/86400., color='k',
                   linestyle=':', alpha=.7, zorder=0,
                   label='Removed signals' if i==0 else '_nolegend_')
    ax.legend(loc='upper left')
    return fig, ax


def plot_full(data, channels, output_path, truth=None, removed_times=None):
    """Create and save the full-length plot."""
    fig, _ = make_full_plot(data, channels, truth=truth, removed_times=removed_times)
    fig.savefig(output_path, bbox_inches='tight')
    plt.close(fig)
    logger.info("Saved full plot to %s", output_path)


def plot_zoom(data, channels, window, output_path, truth=None, removed_times=None):
    """Create and save a zoomed-in plot for the given (start, end) window in days."""
    start_day, end_day = window
    fig, _ = make_full_plot(data, channels, truth=truth, removed_times=removed_times)
    fig.axes[0].set_xlim(start_day, end_day)
    fig.axes[0].set_title(f"Zoom: {start_day:.2f} - {end_day:.2f} days")
    fig.savefig(output_path, bbox_inches='tight')
    plt.close(fig)
    logger.info("Saved zoom plot (%.2f-%.2f days) to %s", start_day, end_day, output_path)


def main():
    args = load_arguments()
    init_logging(args.verbose)
    logger.info("Starting")

    channels = args.channels
    logger.debug("Channels: %s", channels)

    data = {}
    for channel in channels:
        logger.debug("Loading %s from %s", channel, args.data_file)
        data[channel] = load_timeseries(
            args.data_file,
            channel,
            decimate=args.decimate, end_exclusion=args.end_exclusion
        )

    truth = None
    if args.truth_file is not None:
        logger.info("Loading truth data from %s", args.truth_file)
        truth = load_truth_file(args.truth_file)
        logger.info(
            "Loaded %d %s truth source(s)", truth.n_sources, truth.source_group
        )

    output_directory = args.output_directory or os.path.dirname(os.path.abspath(args.output_file))
    os.makedirs(output_directory, exist_ok=True)
    removed_times = load_removed_times(args.removed_parameter_files)

    # Full-length plot
    full_plot_path = args.output_file or os.path.join(output_directory, "full.png")
    plot_full(data, channels, full_plot_path, truth=truth, removed_times=removed_times)

    # Zoom plots
    if args.zoom_windows:
        for i, window in enumerate(args.zoom_windows):
            zoom_path = os.path.join(
                output_directory, f"zoom_{i:02d}.png"
            )
            plot_zoom(data, channels, window, zoom_path, truth=truth, removed_times=removed_times)

    logger.info("Done!")


if __name__ == "__main__":
    main()