"""Plot processed Mojito or Sangria timeseries.

This tool loads processed Mojito or Sangria timeseries from an HDF5 file
with root-level channel datasets and ``epoch`` / ``delta_t`` attributes,
and produces:

* One full-length plot of the data for all channels on a single axis, with the
  x-axis in days relative to the epoch.
* Optional whitened and overwhitened plots via ``--output-whitened-plot``
  and ``--output-overwhitened-plot``, using ``--psd-file`` or ``--psd-files``.
  These omit two days at each end after filtering by default, configurable
  with ``--edge-exclusion``; their time axis remains relative to the file epoch.
* Optionally, one zoomed-in plot per window specified via ``--zoom-windows``,
  where each window is given as ``start_days,end_days`` (relative to epoch).

Example::

    python tools/plotting/plot_processed_data.py \
        --data-file output/processed_data.hdf \
        --output-directory plots/processed_data \
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
from hwilas.psd.read import from_txt as psd_from_txt
from hwilas.types import KeyValueAction

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
        description="Plot processed Mojito or Sangria timeseries."
    )
    add_hwilas_common_options(parser)
    add_truth_arguments(parser)

    parser.add_argument(
        "--data-file",
        required=True,
        help="Processed Mojito or Sangria HDF5 file with root-level channel "
             "datasets and epoch/delta_t attributes"
    )
    output = parser.add_mutually_exclusive_group(required=True)
    output.add_argument("--output-plot", help="Full data plot filename")
    output.add_argument(
        "--output-directory",
        help="Directory in which to save the plots"
    )
    parser.add_argument("--output-whitened-plot", "--output_whitened_plot",
                        help="Data divided by PSD plot filename (requires a PSD)")
    parser.add_argument("--output-overwhitened-plot", "--output_overwhitened_plot",
                        help="Data divided by PSD squared plot filename (requires a PSD)")
    parser.add_argument("--psd-file", help="Two-column frequency/PSD text file for all channels")
    parser.add_argument("--psd-files", nargs="+", action=KeyValueAction,
                        help="Channel-specific PSDs as channel:filename pairs")
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
    parser.add_argument("--edge-exclusion", type=float, default=172800.,
                        help="Seconds to trim from each end after whitening or overwhitening; "
                             "default: 172800 (2 days). Use 0 to keep all samples.")
    parser.add_argument("--removed-parameter-files", nargs='+',
                        help="Fitted parameter HDF5 files whose signals have already been subtracted")
    args = parser.parse_args()
    if args.decimate < 1:
        parser.error("Decimation must be positive")
    if not np.isfinite(args.edge_exclusion) or args.edge_exclusion < 0:
        parser.error("Edge exclusion must be finite and nonnegative")
    if args.output_whitened_plot or args.output_overwhitened_plot:
        if not args.psd_file and not args.psd_files:
            parser.error("Whitened plots require --psd-file or --psd-files")
        if args.psd_files and any(ch not in args.psd_files for ch in args.channels):
            parser.error("--psd-files must supply every selected channel")
        psd_paths = ([args.psd_files[ch] for ch in args.channels]
                     if args.psd_files else [args.psd_file])
        for path in psd_paths:
            if not os.path.isfile(path):
                parser.error(f"PSD file does not exist: {path}")
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


def whiten_timeseries(ts, psd_path, power=.5):
    """Use PSD**0.5 for whitening, or PSD**1 for inverse-covariance weighting."""
    spectrum = np.fft.rfft(ts.data)
    delta_f = 1. / (len(ts.data) * ts.delta_t)
    psd = psd_from_txt(psd_path, len(spectrum), delta_f, delta_f,
                       is_asd_file=False).data
    filtered = np.zeros_like(spectrum)
    usable = min(len(psd), len(spectrum))
    valid = np.flatnonzero(np.isfinite(psd[:usable]) & (psd[:usable] > 0))
    valid = valid[valid > 0]
    logger.debug("Applying PSD power %s to %d Fourier bins", power, len(valid))
    filtered[valid] = spectrum[valid] / psd[valid] ** power
    return TimeSeries(np.fft.irfft(filtered, n=len(ts.data)),
                      epoch=ts.epoch, delta_t=ts.delta_t)


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


def make_full_plot(data, channels, truth=None, removed_times=None, reference_epoch=None):
    """Build a full-length figure of the processed timeseries.

    All channels are drawn on a single axis with the x-axis in days relative
    to the epoch.

    Returns the figure and its single axis so that zoom views can be produced
    by re-limiting the x-axis of the same figure.
    """
    fig, ax = plt.subplots(figsize=(14, 5))
    epoch = float(data[channels[0]].epoch) if reference_epoch is None else reference_epoch

    for channel in channels:
        color = CHANNEL_COLORS.get(channel, 'black')
        t_days = (data[channel].sample_times - epoch) / 86400.0
        ax.plot(
            t_days, data[channel].data,
            label=channel,
            color=color,
        )

    t_min = min(
        (data[ch].sample_times[0] - epoch) / 86400.0 for ch in channels
    )
    t_max = max(
        (data[ch].sample_times[-1] - epoch) / 86400.0 for ch in channels
    )

    ax.set_xlim(t_min, t_max)
    ax.set_xlabel("Time (days)")
    ax.set_ylabel("Amplitude")
    if truth is not None:
        _plot_truth_overlay(ax, truth, epoch=epoch)
    for i, time in enumerate(removed_times if removed_times is not None else []):
        ax.axvline((time-epoch)/86400., color='k',
                   linestyle=':', alpha=.7, zorder=0,
                   label='Removed signals' if i==0 else '_nolegend_')
    ax.legend(loc='upper left')
    return fig, ax


def plot_full(data, channels, output_path, truth=None, removed_times=None, reference_epoch=None):
    """Create and save the full-length plot."""
    fig, _ = make_full_plot(data, channels, truth=truth, removed_times=removed_times,
                            reference_epoch=reference_epoch)
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
    filtered_data = {power: {} for power, path in
                     [(.5, args.output_whitened_plot), (1., args.output_overwhitened_plot)]
                     if path}
    for channel in channels:
        logger.debug("Loading %s from %s", channel, args.data_file)
        ts = load_timeseries(
            args.data_file,
            channel,
            decimate=1 if filtered_data else args.decimate,
            end_exclusion=args.end_exclusion
        )

        data[channel] = TimeSeries(ts.data[::args.decimate], epoch=ts.epoch,
                                   delta_t=ts.delta_t*args.decimate) if filtered_data else ts
        if filtered_data:
            edge_samples = int(np.ceil(args.edge_exclusion / ts.delta_t))
            if 2 * edge_samples >= len(ts.data):
                raise ValueError("Edge exclusion removes all whitened plotted data; "
                                 "reduce --edge-exclusion")
        for power, views in filtered_data.items():
            psd_path = args.psd_files[channel] if args.psd_files else args.psd_file
            view = whiten_timeseries(ts, psd_path, power)
            stop = len(view.data) - edge_samples
            logger.debug("Trimming %d samples from each filtered edge of %s", edge_samples, channel)
            views[channel] = TimeSeries(view.data[edge_samples:stop:args.decimate],
                                        epoch=view.epoch + edge_samples * view.delta_t,
                                        delta_t=view.delta_t*args.decimate)

    truth = None
    if args.truth_file is not None:
        logger.info("Loading truth data from %s", args.truth_file)
        truth = load_truth_file(args.truth_file)
        logger.info(
            "Loaded %d %s truth source(s)", truth.n_sources, truth.source_group
        )

    output_directory = args.output_directory or os.path.dirname(os.path.abspath(args.output_plot))
    os.makedirs(output_directory, exist_ok=True)
    removed_times = load_removed_times(args.removed_parameter_files)

    # Full-length plot
    full_plot_path = args.output_plot or os.path.join(output_directory, "full.png")
    plot_full(data, channels, full_plot_path, truth=truth, removed_times=removed_times)

    for power, path in [(.5, args.output_whitened_plot),
                        (1., args.output_overwhitened_plot)]:
        if path:
            os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
            plot_full(filtered_data[power], channels, path, truth=truth,
                      removed_times=removed_times, reference_epoch=float(data[channels[0]].epoch))

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
