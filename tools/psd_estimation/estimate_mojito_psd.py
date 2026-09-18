"""Estimate the PSD from processed mojito data."""

import argparse
import logging

import h5py
import numpy as np
from gwaith.types import TimeSeries

from hwilas import add_hwilas_common_options, init_logging
from hwilas.psd_estimation.smooth import add_smoothing_options, smooth_psd
from hwilas.psd.interpolate import interpolate
from hwilas.types import KeyValueAction

logger = logging.getLogger(__name__)

def load_arguments():
    parser = argparse.ArgumentParser(
        description="Estimate the PSD from processed mojito data file."
    )
    add_hwilas_common_options(parser)
    parser.add_argument(
        "--data-file",
        required=True,
        help="Path to the processed mojito HDF5 file (output of hwilas_process_mojito_data)."
    )
    parser.add_argument(
        "--output-file",
        required=True,
        nargs="+",
        action=KeyValueAction,
        help="Output files for each channel, given as channel:filename pairs"
    )
    parser.add_argument(
        "--segment-duration",
        default=1576800,
        type=float,
        help="Segment duration, in seconds, to use in Welch's method for PSD "
        "calculation. Default 1576800 (18.25 days)"
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
    logger.info("Starting PSD estimation from processed mojito data")

    channels = list(args.output_file.keys())
    logger.debug("Channels: %s", channels)

    logger.info("Loading processed mojito data from %s", args.data_file)
    with h5py.File(args.data_file, 'r') as f:
        epoch = float(f.attrs.get("epoch", 0.0))
        delta_t = float(f.attrs.get("delta_t", 1.0))
        laser_frequency = f.attrs.get("laser_frequency", None)

        # Load requested channels from HDF5
        timeseries = {}
        for channel in channels:
            if channel not in f:
                raise ValueError(
                    f"Channel {channel} not found in {args.data_file}. "
                    f"Available: {list(f.keys())}"
                )
            data_array = np.asarray(f[channel][:], dtype=float)
            timeseries[channel] = TimeSeries(
                data_array,
                delta_t=delta_t,
                epoch=epoch
            )
            logger.debug(
                "Loaded %s: %d samples, delta_t=%.6f, epoch=%.2f",
                channel,
                len(data_array),
                delta_t,
                epoch
            )

    tlen = len(timeseries[channels[0]])
    delta_f = 1.0 / (tlen * delta_t)

    logger.info("Calculating PSDs")
    full_psds = {}
    for channel in channels:
        logger.debug("Calculating PSD for %s", channel)
        full_psds[channel] = timeseries[channel].psd(args.segment_duration)

    logger.info("Interpolating PSDs to delta_f=%.6e", delta_f)
    psds = {
        channel: interpolate(full_psds[channel], delta_f)
        for channel in channels
    }

    if args.smooth_regions:
        logger.info("Applying smoothing to PSDs")

        psds = smooth_psd(
            psds,
            channels,
            args.smooth_regions,
        )

    logger.info("Saving PSDs")
    for channel in channels:
        output_path = args.output_file[channel]
        logger.debug("Saving PSD for %s to %s", channel, output_path)
        psds[channel].save(output_path)

    if args.testing_plots:
        import os
        os.makedirs(args.testing_plots, exist_ok=True)
        from matplotlib import pyplot as plt
        from matplotlib import use
        use('Agg')

        for channel in channels:
            fig, ax = plt.subplots()
            ax.loglog(
                full_psds[channel].sample_frequencies[1:],
                np.sqrt(full_psds[channel].data[1:])
            )
            ax.set_xlabel('Frequency (Hz)')
            ax.set_ylabel('Strain Hz^{-1}')
            ax.set_title(f'Raw PSD: {channel}')
            fig.savefig(os.path.join(args.testing_plots, f'{channel}_raw_psd.png'))
            plt.close(fig)

            fig, ax = plt.subplots()
            ax.loglog(psds[channel].f_array, np.sqrt(psds[channel].data))
            ax.set_xlabel('Frequency (Hz)')
            ax.set_ylabel(f'Strain/{np.sqrt("Hz")}')
            ax.set_title(f'Interpolated PSD: {channel}')
            fig.savefig(os.path.join(args.testing_plots, f'{channel}_interp_psd.png'))
            plt.close(fig)

    logger.info("Done!")

if __name__ == "__main__":
    main()