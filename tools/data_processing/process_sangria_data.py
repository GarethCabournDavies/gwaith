"""
Add padding to either end of the data and inpaint into them and any gaps
"""

import argparse
import h5py
import logging
import numpy as np
import copy
from matplotlib import pyplot as plt
from matplotlib import use
use('Agg')

import pycbc

from hwilas.search_utils.utils import (
    load_sangria_timeseries,
    apply_inpainting
)

logger = logging.getLogger(__name__)

def main():
    parser = argparse.ArgumentParser()
    pycbc.add_common_pycbc_options(parser)
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

    pycbc.init_logging(args.verbose)

    logger.info("Loading data from %s", args.data_file)

    data = load_sangria_timeseries(
        args.data_file,
    )

    dtype = data[args.channels[0]].dtype
    data_start_time = data[args.channels[0]]._epoch
    delta_t = data[args.channels[0]].delta_t
    data_tlen = len(data[args.channels[0]]) * delta_t
    data_end_time = data_start_time + data_tlen
    data_flen = data_tlen // 2 + 1
    data_delta_f = 1. / data_tlen

    logger.info("Padding Data")
    padding_samples = int(args.padding_duration / delta_t)
    padded_tlen = data_tlen + 2 * args.padding_duration
    padded_dlen = int(padded_tlen / delta_t)
    padded_flen = padded_dlen // 2 + 1
    padded_delta_f = 1. / padded_tlen

    padded_data = {
        channel:pycbc.types.TimeSeries(
            np.zeros(padded_dlen, dtype=dtype),
            delta_t=delta_t,
            epoch=data_start_time - args.padding_duration
        )
        for channel in args.channels
    }
    for channel in args.channels:
        padded_data[channel].data[padding_samples:-padding_samples] = data[channel].data

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
            start_idx = int((start - padded_data[channel]._epoch) / delta_t)
            end_idx = int((end - padded_data[channel]._epoch) / delta_t)
            logger.info("%d indices to be zeroed out", end_idx - start_idx)
            padded_data[channel][start_idx:end_idx] = 0

    logger.info("Loading PSD")
    padded_psd = pycbc.psd.from_txt(
        args.psd_file,
        padded_flen,
        padded_delta_f,
        padded_delta_f,
        is_asd_file=False,
    )

    # The padded data should be kept rather than overwritten
    inpainted_data = copy.deepcopy(padded_data)

    # For the inpainting at the ends, we are going to use 2x the padding duration
    # of the data, along with the pads
    start_pad_plus_data = {
        channel:inpainted_data[channel][0:padding_samples*3]
        for channel in args.channels
    }
    end_pad_plus_data = {
        channel:inpainted_data[channel][padded_dlen-padding_samples*3:]
        for channel in args.channels
    }
    ends_dlen = len(end_pad_plus_data[args.channels[0]])
    ends_tlen = ends_dlen * delta_t

    starts_dlen = len(start_pad_plus_data[args.channels[0]])
    assert ends_dlen == starts_dlen


    if starts_dlen % 2 == 0:
        for channel in args.channels:
            start_pad_plus_data[channel].resize(starts_dlen + 1)
            end_pad_plus_data[channel].resize(starts_dlen + 1)

    ends_dlen = len(end_pad_plus_data[args.channels[0]])
    ends_tlen = ends_dlen * delta_t

    starts_dlen = len(start_pad_plus_data[args.channels[0]])
    assert ends_dlen == starts_dlen
    ends_flen = ends_dlen // 2 + 1
    ends_delta_f = 1. / ends_tlen

    ends_psd = pycbc.psd.from_txt(
        args.psd_file,
        ends_flen,
        ends_delta_f,
        ends_delta_f,
        is_asd_file=False,
    )


    for channel in args.channels:

        logger.info("Applying inpainting to pad at the start of the data in channel %s", channel)
        inpainted_start = apply_inpainting(
            start_pad_plus_data[channel],
            ends_psd,
            0,
            padding_samples,
        )

        inpainted_data[channel].data[0:padding_samples] = inpainted_start.data[0:padding_samples]

        logger.info("Applying inpainting to pad at the end of the data in channel %s", channel)
        inpainted_end = apply_inpainting(
            end_pad_plus_data[channel],
            ends_psd,
            padding_samples * 2,
            padding_samples * 3,
        )
        inpainted_data[channel].data[padded_dlen-padding_samples:padded_dlen] = inpainted_end.data[ends_dlen - padding_samples:ends_dlen]


    if args.gaps is not None:
        for start, end in args.gaps:
            gap_section_start_idx = int((start - inpainted_data[channel]._epoch) / delta_t)
            gap_section_end_idx = int((end - inpainted_data[channel]._epoch) / delta_t)
            gap_section = padded_data[channel][gap_section_start_idx - 2 * padding_samples:gap_section_end_idx + 2 * padding_samples]

    logger.info("Saving data to %s", args.output_file)

    with h5py.File(args.output_file, 'w') as f:
        f.attrs["epoch"] = float(inpainted_data[channel]._epoch)
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

        # Add shaded background for padded sections and gaps
        ax.axvspan(
            -args.padding_duration / 86400,
            0,
            color='gray',
            alpha=0.2,
            label='Padded sections'
        )
        ax.axvspan(
            data_tlen / 86400,
            data_tlen / 86400 + args.padding_duration / 86400,
            color='gray',
            alpha=0.2,
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
        ax.set_xlim(
            float(padded_data[channel]._epoch) / 86400,
            (float(padded_data[channel]._epoch) + args.padding_duration * 3) / 86400)
        fig.savefig(f'{args.testing_plots}/processed_data_start.png')
        ax.set_xlim(
            float(padded_data[channel]._epoch + padded_tlen - 3 * args.padding_duration) / 86400,
            float(padded_data[channel]._epoch + padded_tlen) / 86400,
        )
        fig.savefig(f'{args.testing_plots}/processed_data_end.png')

        if args.gaps is not None:
            pass
        # Full data
        ax.set_xlim(
            float(padded_data[channel]._epoch) / 86400,
            float(padded_data[channel]._epoch + padded_tlen) / 86400
        )
        fig.savefig(f'{args.testing_plots}/processed_data.png')
        plt.close(fig)

    logger.info("Done")

if __name__ == "__main__":
    main()