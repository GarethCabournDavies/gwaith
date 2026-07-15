import h5py
import logging
import numpy as np
import argparse
from scipy.signal.windows import hann
from matplotlib import pyplot as plt
from matplotlib import use
use('Agg')

import ldc.io.hdf5 as hdfio

import pycbc
from pycbc.types import TimeSeries, FrequencySeries
from pycbc.psd import interpolate
from pycbc.types import MultiDetOptionAction


logger = logging.getLogger(__name__)

def load_arguments():
    parser = argparse.ArgumentParser()
    pycbc.add_common_pycbc_options(parser)
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
        action=MultiDetOptionAction,
        help="Output files for each channel, given as channel:filename pairs"
    )
    parser.add_argument(
        "--testing-plots",
        help="Directory to store plots used for testing"
    )
    def gap_pair(s):
        try:
            # Splits "1e-2,0.1" into (0.01, 0.1)
            start, end = map(float, s.split(','))
            return (start, end)
        except ValueError:
            raise argparse.ArgumentTypeError("Gaps must be start,end (e.g., 1e-2,0.1)")

    parser.add_argument(
        "--smooth-regions",
        type=gap_pair,
        nargs='+',
    )
    return parser.parse_args()


def main():
    args = load_arguments()

    pycbc.init_logging(args.verbose)

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

    logger.info("Plotting PSDs")
    fig, ax = plt.subplots()
    for channel in channels:
        ax.plot(
            psds[channel].sample_frequencies,
            psds[channel],
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

        frequencies = psds[channels[0]].sample_frequencies
        for region_start, region_end in args.smooth_regions:
            ax.axvspan(
                region_start,
                region_end,
                color='gray',
                alpha=0.2,
            )

            xdiff = np.log10(region_end) - np.log10(region_start)
                        
            # Find the index of the left hand side of the smoothed region:
            left_idx = np.argmin(np.abs(frequencies - region_start))
            # Find the index of the right hand side of the smoothed region:
            right_idx = np.argmin(np.abs(frequencies - region_end))

            for channel in channels:
                # Find the value of the left hand side - to do this we take the
                # median of 5 samples up to the left_idx
                left_value = np.median(abs(psds[channel].data[left_idx-5:left_idx]))
                # Find the value of the rigth hand side - again 5 sample median,
                # but this time above the index
                right_value = np.median(abs(psds[channel].data[right_idx:right_idx+5]))


                ax.plot(
                    [frequencies[left_idx], frequencies[right_idx]],
                    [left_value, right_value],
                    color='k',
                )

                # Get the y=mx+c values in log-log space (xdiff is consistent for all channels)
                ydiff = np.log10(right_value) - np.log10(left_value)
                slope = ydiff / xdiff
                intercept = np.log10(left_value) - slope * np.log10(region_start)

                line_section = np.pow(10, slope * np.log10(frequencies) + intercept)
                ax.plot(
                    frequencies[left_idx:right_idx],
                    line_section[left_idx:right_idx],
                    color='red',
                    linestyle=':',
                )
                
                # Now lets smoothly transition between the PSD values and the
                # straight line using a Hann window
                window = hann(right_idx - left_idx)
                smoothed_psd = line_section[left_idx:right_idx] * window + psds[channel].data[left_idx:right_idx] * (1 - window)
                ax.plot(
                    frequencies[left_idx:right_idx],
                    smoothed_psd,
                    label=f'Smoothed {channel}'
                )
                psds[channel].data[left_idx:right_idx] = smoothed_psd

    ax.axvspan(np.nan, np.nan, color='gray', alpha=0.2, label="Smoothed region(s)")
    ax.legend(loc='upper left')
    fig.savefig(f'{args.testing_plots}/psd.png')

    logger.info("Saving PSDs")
    for channel in channels:
        psds[channel].save(args.output_file[channel])

    logger.info("Done!")

if __name__ == "__main__":
    main()