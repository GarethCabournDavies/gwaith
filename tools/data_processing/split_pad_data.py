"""
Common post-processing for data files: split to analysis window, pad, and inpaint.

This tool works on any HDF5 file containing time-series channels
(e.g., output from hwilas_process_sangria_data or hwilas_process_mojito_data).
Channels are specified via --channels and are not assumed to be any particular set.
"""

import argparse
import logging

import h5py
import numpy as np

from gwaith.types import TimeSeries, FrequencySeries

from hwilas import add_hwilas_common_options, init_logging
from hwilas.types import KeyValueAction
from hwilas.data_processing.inpainting import gate_and_paint
from hwilas.psd.read import from_txt as psd_from_txt

logger = logging.getLogger(__name__)


def load_data(data_file):
    """Load all channel data from an HDF5 file.

    Returns a dict mapping channel name to gwaith TimeSeries.
    All datasets in the file are loaded (no filtering by prefix).
    """
    with h5py.File(data_file, 'r') as f:
        epoch = float(f.attrs["epoch"])
        delta_t = float(f.attrs["delta_t"])
        data = {}
        for key in f.keys():
            data[key] = TimeSeries(
                np.asarray(f[key], dtype=np.float64),
                epoch=epoch,
                delta_t=delta_t,
            )
    return data


def main():
    parser = argparse.ArgumentParser(
        description="Split data to an analysis window, pad, and inpaint gaps."
    )
    add_hwilas_common_options(parser)
    parser.add_argument(
        "--data-file",
        required=True,
        help="Path to the input HDF5 file containing time-series channels"
    )
    parser.add_argument(
        "--output-file",
        required=True,
        help="Path to the output HDF5 file"
    )
    parser.add_argument(
        "--padding-duration",
        type=float,
        default=30 * 86400,
        help="Padding duration per selected end (seconds). Default: 30 days."
    )
    parser.add_argument(
        "--analysis-start-time",
        type=float,
        required=False,
        help="The start time of the analysis window within the data (seconds from epoch). "
             "If not specified, the full data range is used."
    )
    parser.add_argument(
        "--analysis-end-time",
        type=float,
        required=False,
        help="The end time of the analysis window within the data (seconds from epoch). "
             "Inpainted padding will be added after this time."
    )
    parser.add_argument(
        '--gaps',
        type=lambda s: tuple(map(float, s.split(','))),
        nargs='+',
        default=None,
        help='List of start,end gaps in seconds (e.g., --gaps 10,20 35,50)'
    )
    parser.add_argument(
        "--channels",
        nargs='+',
        default=None,
        help="Channels that will be padded / inpainted. Default: all channels in the file. "
             "These will be the only channels added to the output file."
    )
    parser.add_argument(
        "--psd-file",
        help="File to define the PSD"
    )

    parser.add_argument("--padding-side", choices=["start", "end", "both"], default="end",
                        help="Where to add padding; with both, duration is added at EACH end")
    parser.add_argument("--psd-files", nargs="+", action=KeyValueAction,
                        help="Channel-specific PSD files as channel:filename pairs")
    args = parser.parse_args()
    if args.padding_duration < 0:
        parser.error("padding-duration must be nonnegative")
    if args.psd_file is None and args.psd_files is None:
        parser.error("Provide --psd-file or --psd-files")

    init_logging(args.verbose)

    logger.info("Loading data from %s", args.data_file)

    data = load_data(args.data_file)

    # If no channels specified, use all channels in the file
    if args.channels is None:
        args.channels = list(data.keys())
        logger.info("No --channels specified, using all channels: %s", args.channels)

    # Verify requested channels exist
    for channel in args.channels:
        if channel not in data:
            raise ValueError(
                f"Channel {channel} not found in input file. "
                f"Available channels: {list(data.keys())}"
            )

    data_start_time = data[args.channels[0]].epoch
    delta_t = data[args.channels[0]].delta_t
    data_tlen = len(data[args.channels[0]]) * delta_t
    data_end_time = data_start_time + data_tlen

    if (args.analysis_start_time is None) != (args.analysis_end_time is None):
        raise ValueError(
            "Provide both --analysis-start-time and --analysis-end-time, or neither."
        )

    if args.analysis_start_time is not None and args.analysis_end_time is not None:
        analysis_start_time = max(data_start_time, args.analysis_start_time)
        analysis_end_time = min(data_end_time, args.analysis_end_time)
        analysis_duration = analysis_end_time - analysis_start_time
    else:
        # If no analysis window is specified, use the full data range
        analysis_start_time = data_start_time
        analysis_end_time = data_end_time
        analysis_duration = data_tlen

    if analysis_duration <= 0:
        raise ValueError("Analysis interval must be nonempty")

    if data_tlen > analysis_duration:
        # Cut the data down to just the analysis start / end period
        analysis_start_idx = int((analysis_start_time - data_start_time) / delta_t)
        analysis_end_idx = int((analysis_end_time - data_start_time) / delta_t)

        # Rebuild TimeSeries explicitly so the chunk epoch is pinned to the
        # analysis-window start for all non-first chunks.
        data = {
            channel: TimeSeries(
                np.asarray(data[channel][analysis_start_idx:analysis_end_idx], dtype=np.float64).copy(),
                epoch=analysis_start_time,
                delta_t=delta_t,
            )
            for channel in args.channels
        }

    logger.info(
        "Analysis window: %.0f to %.0f",
        analysis_start_time,
        analysis_end_time
    )

    logger.info("Padding whole analysis interval at %s", args.padding_side)
    original_dlen = len(data[args.channels[0]])
    if any(len(data[c]) != original_dlen for c in args.channels):
        raise ValueError("Selected channels must have equal lengths")
    padding_idxlen = round(args.padding_duration / delta_t)
    if not np.isclose(padding_idxlen * delta_t, args.padding_duration):
        raise ValueError("Padding duration must be a whole number of samples")
    left = padding_idxlen if args.padding_side in ("start", "both") else 0
    right = padding_idxlen if args.padding_side in ("end", "both") else 0
    output_epoch = float(data[args.channels[0]].epoch) - left * delta_t
    for channel in args.channels:
        data[channel] = TimeSeries(np.pad(data[channel].data, (left, right)),
                                   epoch=output_epoch, delta_t=delta_t)
    padded_dlen = original_dlen + left + right
    padded_flen = padded_dlen // 2 + 1
    padded_delta_f = 1. / (padded_dlen * delta_t)
    gap_indices_start, gap_indices_end = [], []
    if left:
        gap_indices_start.append(0)
        gap_indices_end.append(left)
    if right:
        gap_indices_start.append(left + original_dlen)
        gap_indices_end.append(padded_dlen)

    if args.gaps is not None:
        for start, end in args.gaps:
            start_idx = int((start - data[args.channels[0]].epoch) / delta_t)
            end_idx = int((end - data[args.channels[0]].epoch) / delta_t)
            gap_indices_start.append(start_idx)
            gap_indices_end.append(end_idx)

    for channel in args.channels:
        logger.info(f"Applying inpainting to padded + gated sections in channel {channel}")
        psd_path = args.psd_files.get(channel) if args.psd_files else args.psd_file
        if psd_path is None:
            raise ValueError(f"Missing PSD file for {channel}")
        logger.debug("Reading %s PSD from %s", channel, psd_path)
        inpainting_psd = psd_from_txt(psd_path, padded_flen, padded_delta_f,
                                     padded_delta_f, is_asd_file=False)
        data[channel] = gate_and_paint(
            data[channel],
            gap_indices_start,
            gap_indices_end,
            inpainting_psd,
        )

    logger.info("Saving data to %s", args.output_file)

    with h5py.File(args.output_file, 'w') as f:
        with h5py.File(args.data_file, "r") as source:
            for key, value in source.attrs.items():
                f.attrs[key] = value
        f.attrs["epoch"] = float(data[args.channels[0]].epoch)
        f.attrs["delta_t"] = float(delta_t)
        for channel in args.channels:
            f.create_dataset(channel, data=data[channel].data)
        gap_starts, gap_ends = [], []
        if args.gaps is not None:
            for start, end in args.gaps:
                gap_starts.append(start)
                gap_ends.append(end)
            f.attrs["gap_starts"] = gap_starts
            f.attrs["gap_ends"] = gap_ends
        f.attrs["padding_duration"] = args.padding_duration
        f.attrs["padding_start_duration"] = left * delta_t
        f.attrs["padding_end_duration"] = right * delta_t
        f.attrs["analysis_start_time"] = float(analysis_start_time)
        f.attrs["analysis_end_time"] = float(analysis_start_time) + original_dlen * delta_t


    logger.info("Done")


if __name__ == "__main__":
    main()