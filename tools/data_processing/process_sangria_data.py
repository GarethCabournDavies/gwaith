"""
Add padding to either end of the data and inpaint into them and any gaps
"""

import argparse
import copy
import logging

import h5py
import numpy as np
from matplotlib import pyplot as plt
from matplotlib import use

use('Agg')

from hwilas import add_hwilas_common_options, init_logging
from hwilas.data_processing.inpainting import gate_and_paint
from hwilas.data_processing.sangria import load_sangria_timeseries
from hwilas.psd.read import from_txt as psd_from_txt

logger = logging.getLogger(__name__)

def main():
    parser = argparse.ArgumentParser()
    add_hwilas_common_options(parser)
    parser.add_argument(
        "--data-file",
        required=True,
    )
    parser.add_argument(
        "--output-file",
        required=True,
    )
    parser.add_argument(
        "--padding-duration",
        type=float,
        default=30 * 86400,
        help="Duration of padded section to add to the end of the data"
    )
    parser.add_argument(
        "--analysis-start-time",
        type=float,
        required=False,
        help="The start time of the analysis window within the data. "
        "Tapering (if --taper-window is given) will be applied after this point."
    )
    parser.add_argument(
        "--analysis-end-time",
        type=float,
        required=False,
        help="The end time of the analysis window within the data. "
             "Inpainted padding will be added after this time."
    )
    parser.add_argument(
        "--testing-plots",
        help="Directory to store plots used for testing"
    )
    # Add argument for introducing gaps to data
    def gap_pair(s):
        try:
            # Splits "10,20" into (10.0, 20.0)
            start, end = map(float, s.split(','))
            return (start, end)
        except ValueError:
            raise argparse.ArgumentTypeError("Gaps must be start,end (e.g., 10,20)")

    parser.add_argument(
        '--gaps', 
        type=gap_pair, 
        nargs='+', 
        default=None,
        help='List of start,end gaps in seconds (e.g., --gaps 10,20 35,50)'
    )
    parser.add_argument(
        "--channels",
        nargs='+',
        default=['LISA_A', 'LISA_E'],
        help="Channels that will be padded / inpainted. Default LISA_A and LISA_E. "
        "These will be the only channels added to the output file."
    )
    parser.add_argument(
        "--psd-file",
        required=True,
        help="File to define the PSD"
    )
    parser.add_argument(
        "--delta-t",
        type=float,
        default=4,
        help="Sample rate to resample the data to. Must be an integer multiple of the delta_t in the file"
    )

    args = parser.parse_args()

    init_logging(args.verbose)

    logger.info("Loading data from %s", args.data_file)

    data = load_sangria_timeseries(
        args.data_file,
    )

    dtype = data[args.channels[0]].dtype
    data_start_time = data[args.channels[0]].epoch
    delta_t = data[args.channels[0]].delta_t
    data_tlen = len(data[args.channels[0]]) * delta_t
    data_end_time = data_start_time + data_tlen
    # data_flen = data_tlen // 2 + 1
    # data_delta_f = 1. / data_tlen

    if (args.analysis_start_time is None) != (args.analysis_end_time is None):
        raise ValueError("Provide both --analysis-start-time and --analysis-end-time, or neither.")

    if args.analysis_start_time is not None and args.analysis_end_time is not None:
        analysis_start_time = max(data_start_time, args.analysis_start_time)
        analysis_end_time = min(data_end_time, args.analysis_end_time)
        analysis_duration = analysis_end_time - analysis_start_time
    else:
        # If no analysis window is specified, use the full data range
        analysis_start_time = data_start_time
        analysis_end_time = data_end_time
        analysis_duration = data_tlen

    if data_tlen > analysis_duration:
        # Cut the data down to just the analysis start / end period
        analysis_start_idx = int((analysis_start_time - data_start_time) / delta_t)
        analysis_end_idx = int((analysis_end_time - data_start_time) / delta_t)

        # The copy() is so that it owns its own data
        data = {
            channel: data[channel][analysis_start_idx:analysis_end_idx].copy()
            for channel in args.channels
        }
        
    logger.info(
        "Analysis window: %.0f to %.0f",
        analysis_start_time,
        analysis_end_time
    )

    logger.info("Padding Data")
    # The total duration of the output padded data file
    padded_tlen = analysis_duration + args.padding_duration
    original_dlen = len(data[args.channels[0]])
    padding_idxlen = int(args.padding_duration / delta_t)
    padded_dlen = int(padded_tlen / delta_t)

    for channel in args.channels:
        data[channel].resize(padded_dlen)

    logger.info("Loading PSD")

    # Determine the PSD for the inpainting regions.
    # The inpainting function `gate_and_paint` expects a PSD that matches the length
    # of the data it's operating on.

    # We use a PSD that is derived from the full length of the padded data
    # to get the correct delta_f for the PSD.
    padded_flen = padded_dlen // 2 + 1
    padded_delta_f = 1. / padded_tlen

    inpainting_psd = psd_from_txt(
        args.psd_file,
        padded_flen,
        padded_delta_f,
        padded_delta_f,
        is_asd_file=False,
    )

    # Define the "gaps" (padding) at the start / end of the data
    gap_indices_start = [original_dlen]
    gap_indices_end = [padded_dlen]

    if args.gaps is not None:
        for start, end in args.gaps:
            start_idx = int((start - data[channel].epoch) / delta_t)
            end_idx = int((end - data[channel].epoch) / delta_t)
            gap_indices_start.append(start_idx)
            gap_indices_end.append(end_idx)

    for channel in args.channels:
        logger.info(f"Applying inpainting to padded + gated sections in channel {channel}")
        data[channel] = gate_and_paint(
            data[channel],
            gap_indices_start,
            gap_indices_end,
            inpainting_psd,
        )

    logger.info("Saving data to %s", args.output_file)

    with h5py.File(args.output_file, 'w') as f:
        f.attrs["epoch"] = float(data[args.channels[0]].epoch)
        f.attrs["delta_t"] = data[channel].delta_t
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

    if args.testing_plots is not None:
        logger.info("Generating plots")
        fig, ax = plt.subplots()
        for channel in args.channels:
            ax.plot(
                data[channel].sample_times / 86400,
                data[channel],
                label=f"Padded and inpainted data {channel}"
            )

        # Add shaded background for padded section
        ax.axvspan(
            analysis_end_time / 86400,
            (analysis_end_time + args.padding_duration) / 86400,
            color='gray',
            alpha=0.2,
            label='Padded section at end of data'
        )
        if args.gaps is not None:
            for start, end in args.gaps:
                ax.axvspan(
                    start / 86400,
                    end / 86400,
                    color='r',
                    alpha=0.2,
                )
            ax.axvspan(np.nan, np.nan, color='r', alpha=0.2, label="Gap(s)")

        ax.set_xlabel('Time (days)')
        ax.set_ylabel('Amplitude')
        ax.legend(loc='upper right')

        logger.info("Saving plots")

        ax.set_xlim(
            (analysis_start_time) / 86400,
            (analysis_end_time + args.padding_duration) / 86400
        )
        fig.savefig(f'{args.testing_plots}/processed_data.png')
        plt.close(fig)

    logger.info("Done")

if __name__ == "__main__":
    main()