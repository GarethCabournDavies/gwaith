"""
Load a Sangria file, convert to AET channels, optionally downsample, and save to HDF5.
"""

import argparse
import logging

import h5py

from hwilas import add_hwilas_common_options, init_logging
from hwilas.data_processing.sangria import load_downsample_sangria

logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(
        description="Load a Sangria file, convert to AET, optionally downsample, and save to HDF5."
    )
    add_hwilas_common_options(parser)
    parser.add_argument(
        "--data-file",
        required=True,
        help="Path to the input Sangria HDF5 file"
    )
    parser.add_argument(
        "--output-file",
        required=True,
        help="Path to the output HDF5 file"
    )
    parser.add_argument(
        "--channels",
        nargs='+',
        default=['LISA_A', 'LISA_E', 'LISA_T'],
        help="Channels to include in the output. Default: LISA_A LISA_E LISA_T"
    )
    parser.add_argument(
        "--delta-t",
        type=float,
        default=None,
        help="Target sample time for downsampling. If not specified or equal to the "
             "file's delta_t, no downsampling is performed."
    )

    args = parser.parse_args()

    init_logging(args.verbose)

    logger.info("Loading Sangria data from %s", args.data_file)

    data = load_downsample_sangria(
        args.data_file,
        downsample_delta_t=args.delta_t,
        channels=args.channels,
    )

    logger.info("Saving loaded data to %s", args.output_file)

    reference_channel = args.channels[0]
    epoch = float(data[reference_channel].epoch)
    delta_t = float(data[reference_channel].delta_t)

    with h5py.File(args.output_file, 'w') as f:
        f.attrs["epoch"] = epoch
        f.attrs["delta_t"] = delta_t
        for channel in args.channels:
            f.create_dataset(channel, data=data[channel].data)

    logger.info(
        "Done. Saved %d channels with delta_t=%.4f, epoch=%.2f",
        len(args.channels), delta_t, epoch,
    )


if __name__ == "__main__":
    main()