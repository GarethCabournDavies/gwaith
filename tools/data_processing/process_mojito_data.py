import argparse
import logging

from hwilas import add_hwilas_common_options, init_logging
from hwilas.data_processing.mojito import (
    load_downsample_mojito,
)

logger = logging.getLogger("hwilas_process_mojito_data")

def load_arguments():
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
        "--channels",
        nargs='+',
        required=True,
        help="Channels to process"
    )
    parser.add_argument(
        "--downsample-delta-t",
        type=float,
        default=4,
        help="Sample rate to resample the data to. Must be an integer multiple of the delta_t in the file"
    )
    return parser.parse_args()


def main():
    args = load_arguments()

    init_logging(args.verbose)

    logger.info("Starting")

    data = load_downsample_mojito(
        args.data_file,
        args.downsample_delta_t,
        channels=args.channels
    )

    logger.info("Done!")



if __name__ == "__main__":
    main()