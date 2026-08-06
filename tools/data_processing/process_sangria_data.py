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

from gwaith.types import TimeSeries

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
    )
    parser.add_argument(
        "--analysis-start-time",
        type=float,
        required=False,
        help="The start time of the analysis window within the data. "
             "Padding will be added before and after this window."
    )
    parser.add_argument(
        "--analysis-end-time",
        type=float,
        required=False,
        help="The end time of the analysis window within the data. "
             "Padding will be added before and after this window."
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
        analysis_start_time = args.analysis_start_time
        analysis_end_time = args.analysis_end_time
    else:
        # If no analysis window is specified, use the full data range
        analysis_start_time = data_start_time
        analysis_end_time = data_end_time

    logger.info(f"Analysis window: {analysis_start_time:.0f} to {analysis_end_time:.0f}")

    logger.info("Padding Data")
    # The total duration of the output padded data file
    padded_tlen = (analysis_end_time - analysis_start_time) + 2 * args.padding_duration
    padded_dlen = int(padded_tlen / delta_t)

    # The epoch of the new padded data TimeSeries
    padded_data_epoch = analysis_start_time - args.padding_duration
    padded_data_end_time = analysis_end_time + args.padding_duration

    padded_data = {
        channel:TimeSeries(
            np.zeros(padded_dlen, dtype=dtype),
            delta_t=delta_t,
            epoch=padded_data_epoch
        )
        for channel in args.channels
    }

    # Calculate the indices within the padded_data where the original data should be inserted
    # This is the start of the original data relative to the padded_data_epoch
    original_data_insert_start_idx = int(max(0, (data_start_time - padded_data_epoch) / delta_t))
    # This is the end of the original data relative to the padded_data_epoch
    original_data_insert_end_idx = int(min(padded_dlen, (data_end_time - padded_data_epoch) / delta_t))

    # Calculate the indices within the original data to extract
    original_data_extract_start_idx = int(max(0, (padded_data_epoch - data_start_time) / delta_t))
    original_data_extract_end_idx = int(min(len(data[args.channels[0]]), (padded_data_end_time - data_start_time) / delta_t))

    for channel in args.channels:
        # Copy the relevant portion of the original data into the padded data
        padded_data[channel].data[original_data_insert_start_idx:original_data_insert_end_idx] = \
            data[channel].data[original_data_extract_start_idx:original_data_extract_end_idx]

        # Zero out any "future" data (beyond the original data's end time)
        if padded_data_end_time > data_end_time:
            future_start_idx = int(max(0, (data_end_time - padded_data_epoch) / delta_t))
            padded_data[channel].data[future_start_idx:] = 0.0

    if args.gaps is not None:
        logger.info("Applying Gaps")
        for start, end in args.gaps:
            logger.info("Applying gap from %.1f to %.1f seconds", start, end)
            if start > data_end_time or end < data_start_time:
                logger.info("Gap is outside of data range, skipping")
                continue
            if start < data_start_time:
                logger.info("Gap starts before data, setting to data start")
                start = data_start_time
            if end > data_end_time:
                logger.info("Gap ends after data, setting to data end")
                end = data_end_time
            start_idx = int((start - padded_data[channel].epoch) / delta_t)
            end_idx = int((end - padded_data[channel].epoch) / delta_t)
            logger.info("%d indices to be zeroed out", end_idx - start_idx)
            padded_data[channel][start_idx:end_idx] = 0

    # Define the padding regions for inpainting
    start_padding_idx = 0
    end_padding_idx = int(args.padding_duration / delta_t)
    

    logger.info("Loading PSD")
    # padded_psd = psd_from_txt(
    #     args.psd_file,
    #     padded_flen,
    #     padded_delta_f,
    #     padded_delta_f,
    #     is_asd_file=False,
    # )

    # The padded data should be kept rather than overwritten
    inpainted_data = padded_data # We will modify this in place

    # Determine the PSD for the inpainting regions.
    # The inpainting function `gate_and_paint` expects a PSD that matches the length
    # of the data it's operating on.
    # For the padding regions, we need a PSD that covers at least the padding duration.
    # Let's use a PSD derived from a segment of length 3 * padding_duration
    # to ensure enough frequency resolution for inpainting the padding regions.
    inpainting_segment_duration = 3 * args.padding_duration
    inpainting_segment_dlen = int(inpainting_segment_duration / delta_t)
    if inpainting_segment_dlen % 2 == 0:
        inpainting_segment_dlen += 1 # Ensure odd length for FFT
    inpainting_segment_tlen = inpainting_segment_dlen * delta_t
    inpainting_segment_flen = inpainting_segment_dlen // 2 + 1
    inpainting_segment_delta_f = 1. / inpainting_segment_tlen

    ends_psd = psd_from_txt( # Renaming `ends_psd` to `inpainting_psd` for clarity
        args.psd_file,
        inpainting_segment_flen,
        inpainting_segment_delta_f,
        inpainting_segment_delta_f,
        is_asd_file=False,
    )

    gap_indices_start = [start_padding_idx, padded_dlen - end_padding_idx]
    gap_indices_end = [end_padding_idx, padded_dlen - start_padding_idx]

    if args.gaps is not None:
        for start, end in args.gaps:
            start_idx = int((start - padded_data[channel].epoch) / delta_t)
            end_idx = int((end - inpainted_data[channel].epoch) / delta_t)
            gap_indices_start.append(start_idx)
            gap_indices_end.append(end_idx)


    for channel in args.channels:

        # Apply inpainting to the initial padding region
        logger.info(f"Applying inpainting to initial padding in channel {channel}")
        inpainted_data[channel] = gate_and_paint(
            inpainted_data[channel],
            gap_indices_start,
            gap_indices_end,
            ends_psd,
        )


    logger.info("Saving data to %s", args.output_file)

    with h5py.File(args.output_file, 'w') as f:
        f.attrs["epoch"] = float(inpainted_data[args.channels[0]].epoch)
        f.attrs["delta_t"] = inpainted_data[channel].delta_t
        for channel in args.channels:
            f.create_dataset(channel, data=inpainted_data[channel].data)
        gap_starts, gap_ends = [], []
        if args.gaps is not None:
            for start, end in args.gaps:
                gap_starts.append(start)
                gap_ends.append(end)
            f.attrs["gap_starts"] = gap_starts
            f.attrs["gap_ends"] = gap_ends
        f.attrs["padding_duration"] = args.padding_duration

    # Plotting logic needs to be updated to reflect the new time ranges
    # and the fact that original data might not span the whole plot.
    if args.testing_plots is not None:
        logger.info("Generating plots")
        fig, ax = plt.subplots()
        for channel in args.channels:
            ax.plot(
                data[channel].sample_times / 86400,
                data[channel],
                label=f"Original data {channel}"
            )
            ax.plot(
                padded_data[channel].sample_times / 86400,
                padded_data[channel],
                label=f"Padded data {channel}"
            )
            ax.plot(
                inpainted_data[channel].sample_times / 86400,
                inpainted_data[channel],
                label=f"Inpainted data {channel}"
            )

        # Add shaded background for padded sections
        # The padding is now relative to the analysis window
        ax.axvspan(
            (padded_data_epoch - padded_data_epoch) / 86400, # 0
            (analysis_start_time - padded_data_epoch) / 86400,
            color='gray',
            alpha=0.2,
            label='Padded sections'
        )
        ax.axvspan(
            (analysis_end_time - padded_data_epoch) / 86400,
            (padded_data_end_time - padded_data_epoch) / 86400,
            color='gray',
            alpha=0.2,
        )
        # Indicate the original data range
        ax.axvspan(
            (data_start_time - padded_data_epoch) / 86400, (data_end_time - padded_data_epoch) / 86400,
            color='blue', alpha=0.1, label='Original Data Range'
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

        ax.set_xlabel('Time from epoch (days)')
        ax.set_ylabel('Amplitude')
        ax.legend(loc='upper right')

        logger.info("Saving plots")
        # Output as a series of figures but with different x limits to the axis
        # Start pad:
        ax.set_xlim(padded_data_epoch / 86400,
                    (padded_data_epoch + 3 * args.padding_duration) / 86400)
        fig.savefig(f'{args.testing_plots}/processed_data_start.png')
        ax.set_xlim(
            (padded_data_end_time - 3 * args.padding_duration) / 86400,
            padded_data_end_time / 86400,
        )
        fig.savefig(f'{args.testing_plots}/processed_data_end.png')

        if args.gaps is not None:
            # TODO: Add specific plots around gaps if desired
            pass
        # Full data
        ax.set_xlim(
            float(padded_data[channel].epoch) / 86400,
            float(padded_data[channel].epoch + padded_tlen) / 86400
        )
        fig.savefig(f'{args.testing_plots}/processed_data.png')
        plt.close(fig)

    logger.info("Done")

if __name__ == "__main__":
    main()