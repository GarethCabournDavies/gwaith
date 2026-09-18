"""Plot padded / inpainted data from a ``hwilas_split_pad_data`` output file.

This tool reproduces the diagnostic plots that were previously generated inline
by ``hwilas_split_pad_data`` when run with ``--testing-plots``:

* one combined figure showing the raw (padded + inpainted) time series, the
    whitened data, and the overwhitened data for the selected channels.
    Inpainted / padded regions should appear flat in the overwhitened panel,
    which is how we verify that the inpainting has worked.

Because the whitened / overwhitened panels are derived from the same frequency
data and PSD used during processing, this tool requires the same ``--psd-file``
(and matching ``--channels`` / analysis window) that were passed to
``hwilas_split_pad_data``. The padded section is read from the ``padding_duration``
attribute written into the output file; gaps are read from the ``gap_starts`` /
``gap_ends`` attributes when present.

Example::

    python tools/plotting/plot_split_pad_data.py \
        --data-file output/data_processing/padded.hdf \
        --psd-file psd_estimation/output/A_sangria_hm_SMOOTHED_PSD.txt \
        --channels LISA_A LISA_E \
        --analysis-start-time 0 \
        --analysis-end-time 10542720 \
        --output-file plots/data_processing/padded_summary.png
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

logger = logging.getLogger(__name__)


def load_arguments():
    parser = argparse.ArgumentParser(
        description="Plot padded / inpainted data from a hwilas_split_pad_data output file."
    )
    add_hwilas_common_options(parser)
    add_truth_arguments(parser)

    parser.add_argument(
        "--data-file",
        required=True,
        help="HDF5 file of padded + inpainted data (output of hwilas_split_pad_data)"
    )
    parser.add_argument(
        "--psd-file",
        required=True,
        help="PSD file used for whitening / overwhitening (same as passed to "
             "hwilas_split_pad_data)"
    )
    parser.add_argument(
        "--channels",
        nargs='+',
        default=None,
        help="Channels to plot. Default: all channels in the data file."
    )
    parser.add_argument(
        "--analysis-start-time",
        type=float,
        required=False,
        help="Start of the analysis window (seconds from epoch). If not given, "
             "the start of the data file is used. Used only to set the plot x-range."
    )
    parser.add_argument(
        "--analysis-end-time",
        type=float,
        required=False,
        help="End of the analysis window (seconds from epoch). If not given, the "
             "padded section is assumed to start at the end of the data file. "
             "Used only to set the plot x-range."
    )
    parser.add_argument(
        "--output-file",
        required=True,
        help="Path to the output image file"
    )

    return parser.parse_args()


def load_data(data_file, channels):
    """Load channel data and file attributes from a padded-data HDF5 file.

    Returns ``(data, attrs)`` where ``data`` maps each requested channel to a
    :class:`gwaith.types.TimeSeries` and ``attrs`` holds the relevant file
    attributes (padding duration and any gaps).
    """
    with h5py.File(data_file, 'r') as f:
        epoch = float(f.attrs["epoch"])
        delta_t = float(f.attrs["delta_t"])

        if channels is None:
            channel_names = list(f.keys())
        else:
            missing = [c for c in channels if c not in f]
            if missing:
                raise ValueError(
                    f"Channel(s) {missing} not found in data file. "
                    f"Available channels: {list(f.keys())}"
                )
            channel_names = list(channels)

        data = {}
        for channel in channel_names:
            data[channel] = TimeSeries(
                np.asarray(f[channel], dtype=np.float64),
                epoch=epoch,
                delta_t=delta_t,
            )

        attrs = {
            "padding_duration": float(f.attrs.get("padding_duration", 0.0)),
            "gap_starts": list(f.attrs["gap_starts"]) if "gap_starts" in f.attrs else None,
            "gap_ends": list(f.attrs["gap_ends"]) if "gap_ends" in f.attrs else None,
        }
    return data, attrs


def _gaps_from_attrs(attrs):
    """Return a list of ``(start, end)`` gap tuples from the file attributes."""
    if attrs["gap_starts"] is None or attrs["gap_ends"] is None:
        return None
    return list(zip(attrs["gap_starts"], attrs["gap_ends"]))


def _shade_padding_and_gaps(ax, analysis_end_time, padding_duration, gaps, *, label_padding=False):
    """Shade the padded and gap regions on a plot axis."""
    ax.axvspan(
        analysis_end_time / 86400.0,
        (analysis_end_time + padding_duration) / 86400.0,
        color='gray',
        alpha=0.2,
        label='Padded section' if label_padding else None,
    )
    if gaps is not None:
        for start, end in gaps:
            ax.axvspan(start / 86400.0, end / 86400.0, color='r', alpha=0.2)
        if label_padding:
            ax.axvspan(np.nan, np.nan, color='r', alpha=0.2, label='Gap(s)')


def _make_frequency_domain_views(ts, inpainting_psd):
    """Return whitened and overwhitened time series for one input channel."""
    fs = ts.to_frequencyseries()
    data = np.asarray(fs.data, dtype=np.complex128)
    psd = np.asarray(inpainting_psd.data, dtype=np.float64)

    if len(data) != len(psd):
        raise ValueError(
            "Frequency data and PSD length mismatch: "
            f"{len(data)} vs {len(psd)}"
        )

    valid = np.isfinite(psd) & (psd > 0)
    valid[0] = False

    whitened = np.zeros_like(data)
    whitened[valid] = data[valid] / psd[valid]

    overwhitened = np.zeros_like(data)
    overwhitened[valid] = data[valid] / (psd[valid] ** 2)

    frequency_series_type = type(fs)
    whitened_ts = frequency_series_type(whitened, delta_f=fs.delta_f).to_timeseries()
    overwhitened_ts = frequency_series_type(overwhitened, delta_f=fs.delta_f).to_timeseries()

    # Preserve the original absolute epoch; to_timeseries() starts at 0.
    whitened_ts = TimeSeries(np.asarray(whitened_ts.data, dtype=np.float64), epoch=ts.epoch, delta_t=ts.delta_t)
    overwhitened_ts = TimeSeries(np.asarray(overwhitened_ts.data, dtype=np.float64), epoch=ts.epoch, delta_t=ts.delta_t)
    return whitened_ts, overwhitened_ts


def _plot_truth_overlay(axs, truth, analysis_start_time=None, analysis_end_time=None):
    """Draw truth coalescence markers on each panel behind the plotted data.

    The split/pad diagnostics use absolute time on the x-axis (seconds from the
    source epoch, displayed in days). Truth coalescence times are also absolute,
    so we filter by the requested window and plot directly without re-basing.
    """
    if not isinstance(axs, (tuple, list)):
        axs = (axs,)

    tc = truth.coalescence_times()
    if len(tc) == 0:
        return

    if analysis_start_time is not None:
        tc = tc[tc >= analysis_start_time]
    if analysis_end_time is not None:
        tc = tc[tc <= analysis_end_time]

    if len(tc) == 0:
        return

    for ax in axs:
        for tc_i in tc:
            ax.axvline(
                tc_i / 86400.0,
                color='r',
                linestyle='--',
                alpha=0.35,
                zorder=0,
            )

    handles = [plt.Line2D([0], [0], color='r', linestyle='--', label=f"Truth ({truth.source_group})")]
    axs[0].legend(handles=axs[0].get_legend_handles_labels()[0] + handles)


def plot_data_summary(data, channels, analysis_start_time, analysis_end_time,
                      padding_duration, gaps, inpainting_psd, output_path, truth=None):
    """Plot the raw, whitened and overwhitened data for the selected channels."""
    fig, axes = plt.subplots(3, 1, figsize=(16, 12), sharex=True)

    ax_raw, ax_whitened, ax_overwhitened = axes

    for channel in channels:
        ts = data[channel]
        whitened_data, overwhitened_data = _make_frequency_domain_views(ts, inpainting_psd)

        ax_raw.plot(
            ts.sample_times / 86400.0,
            ts.data,
            label=f"{channel}",
        )
        ax_whitened.plot(
            whitened_data.sample_times / 86400.0,
            whitened_data.data,
            label=f"{channel}",
        )
        ax_overwhitened.plot(
            overwhitened_data.sample_times / 86400.0,
            overwhitened_data.data,
            label=f"{channel}",
        )

    _shade_padding_and_gaps(
        ax_raw,
        analysis_end_time,
        padding_duration,
        gaps,
        label_padding=True,
    )
    if truth is not None:
        _plot_truth_overlay(
            (ax_raw, ax_whitened, ax_overwhitened),
            truth,
            analysis_start_time=analysis_start_time,
            analysis_end_time=analysis_end_time + padding_duration,
        )

    for ax in (ax_raw, ax_whitened, ax_overwhitened):
        _shade_padding_and_gaps(ax, analysis_end_time, padding_duration, gaps)

    ax_raw.set_ylabel('Amplitude')
    ax_whitened.set_ylabel('Whitened data')
    ax_overwhitened.set_ylabel('Overwhitened data')
    ax_overwhitened.set_xlabel('Time (days)')

    ax_raw.legend(loc='upper right')
    ax_whitened.legend(loc='upper right')
    ax_overwhitened.legend(loc='upper right')

    for ax in axes:
        ax.set_xlim(
            analysis_start_time / 86400.0,
            (analysis_end_time + padding_duration) / 86400.0,
        )

    fig.suptitle('Split / pad summary')
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    logger.info("Saved summary plot to %s", output_path)


def main():
    args = load_arguments()

    # Some libraries may add a default StreamHandler to the root logger upon
    # import; clear any pre-existing handlers before initialising our own.
    logging.getLogger().handlers.clear()
    init_logging(args.verbose)

    logger.info("Starting")

    data, attrs = load_data(args.data_file, args.channels)
    channels = list(data.keys())
    logger.info(
        "Loaded %d channels from %s: %s", len(channels), args.data_file, channels
    )

    # Resolve the analysis window. If not given explicitly, fall back to the data
    # range (padded section assumed to start at the end of the file).
    ref_channel = channels[0]
    data_start_time = data[ref_channel].epoch
    data_tlen = len(data[ref_channel]) * data[ref_channel].delta_t
    data_end_time = data_start_time + data_tlen

    if (args.analysis_start_time is None) != (args.analysis_end_time is None):
        raise ValueError(
            "Provide both --analysis-start-time and --analysis-end-time, or neither."
        )

    if args.analysis_start_time is not None and args.analysis_end_time is not None:
        analysis_start_time = max(data_start_time, args.analysis_start_time)
        analysis_end_time = min(data_end_time, args.analysis_end_time)
    else:
        analysis_start_time = data_start_time
        analysis_end_time = data_end_time

    padding_duration = attrs["padding_duration"]
    gaps = _gaps_from_attrs(attrs)

    truth = None
    if args.truth_file is not None:
        logger.info("Loading truth data from %s", args.truth_file)
        truth_time_min = args.truth_time_min
        truth_time_max = args.truth_time_max
        if truth_time_min is None:
            truth_time_min = analysis_start_time
        if truth_time_max is None:
            truth_time_max = analysis_end_time + padding_duration

        truth = load_truth_file(
            args.truth_file,
            time_min=truth_time_min,
            time_max=truth_time_max,
        )
        logger.info(
            "Loaded %d %s truth source(s) for the active window [%s, %s]",
            truth.n_sources,
            truth.source_group,
            truth_time_min,
            truth_time_max,
        )
        if truth.n_sources:
            logger.info(
                "Truth times in active window: %s",
                np.array2string(
                    np.asarray(truth.coalescence_times(), dtype=np.float64),
                    precision=6,
                    separator=', ',
                ),
            )

    # Build the PSD at the length of the (padded) data, matching how
    # hwilas_split_pad_data constructs its inpainting PSD.
    padded_tlen = len(data[ref_channel]) * data[ref_channel].delta_t
    padded_delta_f = 1. / padded_tlen
    padded_flen = int(len(data[ref_channel])) // 2 + 1
    inpainting_psd = psd_from_txt(
        args.psd_file,
        padded_flen,
        padded_delta_f,
        padded_delta_f,
        is_asd_file=False,
    )

    output_dir = os.path.dirname(args.output_file)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    plot_data_summary(
        data,
        channels,
        analysis_start_time,
        analysis_end_time,
        padding_duration,
        gaps,
        inpainting_psd,
        args.output_file,
        truth=truth,
    )

    logger.info("Done!")


if __name__ == "__main__":
    main()
