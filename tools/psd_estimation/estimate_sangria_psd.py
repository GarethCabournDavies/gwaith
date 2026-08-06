import argparse
import logging

import numpy as np
from matplotlib import pyplot as plt
from matplotlib import use

use('Agg')

import ldc.io.hdf5 as hdfio

from gwaith.types import TimeSeries
from gwaith.io import KeyValueOptionAction

from hwilas import add_hwilas_common_options, init_logging
from hwilas.psd.interpolate import interpolate
from hwilas.psd_estimation.smooth import add_smoothing_options, smooth_psd

logger = logging.getLogger(__name__)

def load_arguments():
    parser = argparse.ArgumentParser()
    add_hwilas_common_options(parser)
    parser.add_argument(
        "--data-file",
        required=True,
    )
    parser.add_argument(
        "--segment-duration",
        default=1576800,
        type=float,
        help="Segment duration to use in Welch's method for PSD calculation"
    )
    parser.add_argument(
        "--output-file",
        required=True,
        action=KeyValueOptionAction,
        help="Output files for each channel, given as channel:filename pairs"
    )
    parser.add_argument(
        "--testing-plots",
        help="Directory to store plots used for testing"
    )
    add_smoothing_options(parser)

    return parser.parse_args()


def main():
    args = load_arguments()

    init_logging(args.verbose)

    logger.info("Loading Sangria dataset")

    tdi_ts, tdi_descr = hdfio.load_array(args.data_file, name="obs/tdi")
    
    delta_t = tdi_descr['dt']
    epoch = tdi_descr['t0']

    X = tdi_ts['X']
    Y = tdi_ts['Y']
    Z = tdi_ts['Z']

    # Convert to AET timeseries
    arrays = {
        'LISA_A': (Z - X)/np.sqrt(2),
        'LISA_E': (X - 2*Y + Z)/np.sqrt(6),
        'LISA_T': (X + Y + Z)/np.sqrt(3)
    }

    channels = list(args.output_file.keys())

    logger.info("Converting to TimeSeries")
    timeseries = {
        channel: TimeSeries(
            arrays[channel],
            delta_t=delta_t,
            epoch=epoch
        )
        for channel in channels
    }

    tlen = len(timeseries[channels[0]])
    delta_f = 1. / tlen

    logger.info("Converting to PSD")
    full_psds = {
        channel: timeseries[channel].psd(args.segment_duration)
        for channel in channels
    
    }

    logger.info("Interpolating PSDs")
    psds = {
        channel: interpolate(full_psds[channel], delta_f)
        for channel in channels
    }

    channel_colours = {
        "LISA_A": "tab:blue",
        "LISA_E": "tab:orange",
        "LISA_T": "tab:green"
    }
    smoothed_colours = {
        "LISA_A": "tab:red",
        "LISA_E": "tab:pink",
        "LISA_T": "tab:olive"
    }


    logger.info("Plotting PSDs")
    fig, ax = plt.subplots()
    for channel in channels:
        ax.plot(
            psds[channel].sample_frequencies,
            psds[channel],
            c=channel_colours[channel],
            label=channel
        )

    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlim(1e-6, 0.1)
    ax.set_ylim(1e-43, 1e-37)
    ax.set_xlabel('Frequency (Hz)')
    ax.set_ylabel('PSD (Hz$^{-1}$)')
    ax.legend()


    if args.smooth_regions is not None:
        logger.info("Smoothing PSDs")

        psds = smooth_psd(psds, channels, args.smooth_regions)

        logger.debug("Indicating smoothed regions")
        for region_start, region_end in args.smooth_regions:
            ax.axvspan(
                region_start,
                region_end,
                color='gray',
                alpha=0.2,
            )

        ax.axvspan(np.nan, np.nan, color='gray', alpha=0.2, label="Smoothed region(s)")

        logger.debug("Plotting smoothed PSDs")
        for channel in channels:
            for region_start, region_end in args.smooth_regions:
                this_region = np.logical_and(
                    psds[channel].sample_frequencies >= region_start,
                    psds[channel].sample_frequencies <= region_end
                )
                ax.plot(
                    psds[channel].sample_frequencies[this_region],
                    psds[channel][this_region],
                    c=smoothed_colours[channel],
                    linestyle=':',
                    alpha=0.5
                )
            ax.plot(np.nan, np.nan, smoothed_colours[channel], label=f"Smoothed {channel}", linestyle=':', alpha=0.5)

    ax.legend(loc='upper left')
    fig.savefig(f'{args.testing_plots}/sangria_psd.png')

    logger.info("Saving PSDs")
    for channel in channels:
        psds[channel].save(args.output_file[channel])

    logger.info("Done!")

if __name__ == "__main__":
    main()