"""Estimate the PSD from a mojito data file."""

import argparse
import logging

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
        "--output-files",
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
        "--low-cutoff-freq",
        type=float,
        default=1e-5,
        help="Lower passband edge frequency for the bandpass filter (Hz). "
             "Frequencies below this are attenuated. Default: 1e-5"
    )

    return parser.parse_args()


def main():
    args = load_arguments()
    init_logging(args.verbose)
    logger.info("Starting")

    channels = list(args.output_files.keys())
    logger.debug("Channels: %s", channels)

    data = load_downsample_mojito(
        args.data_file,
        args.downsample_delta_t,
        channels=channels,
        low_cutoff_freq=args.low_cutoff_freq,
    )
    
    for channel in channels:
        logger.info("Saving %s data to %s", channel, args.output_files[channel])
        data[channel].save(args.output_files[channel])

    logger.info("Done!")

if __name__ == "__main__":
    main()
