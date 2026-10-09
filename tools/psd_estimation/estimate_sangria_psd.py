import argparse
import logging
import os

import h5py
import numpy as np
from gwaith.types import TimeSeries
from gwaith.io import KeyValueOptionAction

from hwilas import add_hwilas_common_options, init_logging
from hwilas.psd.interpolate import interpolate
from hwilas.psd_estimation.smooth import add_smoothing_options, smooth_psd

logger = logging.getLogger(__name__)

def load_arguments():
    parser = argparse.ArgumentParser(
        description="Estimate the PSD from processed sangria data file."
    )
    add_hwilas_common_options(parser)
    parser.add_argument(
        "--data-file",
        required=True,
        help="Path to the processed sangria HDF5 file (output of data processing)."
    )
    parser.add_argument(
        "--output-file",
        required=True,
        action=KeyValueOptionAction,
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

    channels = list(args.output_file.keys())
    logger.debug("Channels: %s", channels)

    logger.info("Loading processed sangria data from %s", args.data_file)
    with h5py.File(args.data_file, 'r') as f:
        epoch = float(f.attrs.get("epoch", 0.0))
        delta_t = float(f.attrs.get("delta_t", 1.0))

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

    if args.smoothing_regions or args.smoothing_corners:
        logger.info("Applying smoothing to PSDs")
        for channel in channels:
            psds[channel] = smooth_psd(
                psds[channel],
                regions=args.smoothing_regions,
                corners=args.smoothing_corners,
            )

    if args.testing_plots:
        from matplotlib import pyplot as plt
        from matplotlib import use
        use('Agg')

        os.makedirs(args.testing_plots, exist_ok=True)

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

        logger.info("Plotting PSDs to %s", args.testing_plots)
        fig, ax = plt.subplots()
        for channel in channels:
            ax.plot(
                psds[channel].sample_frequencies,
                psds[channel],
                c=channel_colours.get(channel, 'tab:blue'),
                label=channel
            )

        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.set_xlabel('Frequency (Hz)')
        ax.set_ylabel('PSD (Hz$^{-1}$)')
        ax.set_title('Power Spectral Density')
        ax.legend()

        if args.smoothing_regions is not None:
            logger.debug("Indicating smoothed regions")
            for region_start, region_end in args.smoothing_regions:
                ax.axvspan(
                    region_start,
                    region_end,
                    color='gray',
                    alpha=0.2,
                )
            ax.axvspan(np.nan, np.nan, color='gray', alpha=0.2, label="Smoothed region(s)")
            ax.legend(loc='upper left')

        fig.savefig(os.path.join(args.testing_plots, 'psd_plot.png'))
        plt.close(fig)

    logger.info("Saving PSDs")
    for channel in channels:
        output_path = args.output_file[channel]
        logger.debug("Saving PSD for %s to %s", channel, output_path)
        # Downstream PSD readers expect frequency/PSD columns in plain text.
        np.savetxt(output_path,
                   np.column_stack((np.asarray(psds[channel].sample_frequencies),
                                    np.asarray(psds[channel].data))),
                   header="frequency_Hz PSD")

    logger.info("Done!")

if __name__ == "__main__":
    main()