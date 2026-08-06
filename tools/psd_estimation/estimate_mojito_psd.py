"""Estimate the PSD from a mojito data file."""

import argparse
import logging

from matplotlib import pyplot as plt
from matplotlib import use

use('Agg')


from hwilas import add_hwilas_common_options, init_logging
from hwilas.data_processing.mojito import (
    load_downsample_mojito,
)
from hwilas.types import KeyValueAction

logger = logging.getLogger(__name__)

def load_arguments():
    parser = argparse.ArgumentParser()
    add_hwilas_common_options(parser)
    parser.add_argument(
        "--data-file",
        required=True,
    )
    parser.add_argument(
        "--delta-t",
        type=float,
        help="If given, loadf the Mojito data file with this dt rather than the file content - must be higher than the data dt"
    )
    parser.add_argument(
        "--psd-files",
        required=True,
        action=KeyValueAction,
        nargs='+',
        help="PSD output files, one per channel, given as channel:filename pairs"
    )
    parser.add_argument(
        "--downsample-delta-t",
        type=float,
        default=4,
        help="Sample rate to resample the data to. Must be an integer multiple of the delta_t in the file"
    )
    parser.add_argument(
        "--processor-options",
        action=KeyValueAction,
        nargs='+',
    )
    parser.add_argument(
        "--output-plots",
        default=None,
    )
    parser.add_argument(
        "--segment-duration",
        default=1576800,
        type=float,
        help="Segment duration, in seconds, to use in Welch's method for PSD "
        "calculation. Default 1576800 (18.25 days)"
    )
    parser.add_argument(
        '--smoothing-frequencies',
        type=float,
        nargs='+',
    )

    return parser.parse_args()


def main():
    args = load_arguments()
    init_logging(args.verbose)
    logger.info("Starting")

    channels = list(args.psd_files.keys())
    logger.debug("Channels: %s", channels)

    data = load_downsample_mojito(
        args.data_file,
        args.downsample_delta_t,
        channels=channels
    )
    
    logger.info("Calculating PSDs")
    psds = {}
    for channel in channels:
        logger.debug("Calculating PSD for %s", channel)
        # Getting Raw PSD
        psds[channel] = data[channel].psd(
            segment_duration=args.segment_duration
        )

    if args.output_plots is not None:
        logger.info("Plotting PSDs")
        fig, ax = plt.subplots()
        for channel in channels:
            ax.plot(
                psds[channel].sample_frequencies,
                psds[channel].data,
                label=channel
            )


    for channel in channels:
        logger.debug("Saving PSD to %s", args.psd_files[channel])
        psds[channel].save(args.psd_files[channel])

    if args.output_plots is not None:
        ax.set_yscale('log')
        ax.set_xscale('log')
        ax.set_xlabel('Frequency (Hz)')
        ax.set_xlim(1e-6, 0.1)
        ax.set_ylabel('PSD')
        ax.legend(loc='upper left')
        fig.savefig(f"{args.output_plots}/mojito_psd.png")



    logger.info("Done!")

if __name__ == "__main__":
    main()
