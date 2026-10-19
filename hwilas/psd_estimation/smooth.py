import argparse
import logging

import numpy as np
from scipy.signal.windows import hann

logger = logging.getLogger(__name__)


def smooth_psd(psds_in, channels, smoothing_regions):
    psds = {channel: psds_in[channel].copy() for channel in channels}


    frequencies = psds_in[channels[0]].sample_frequencies
    for region_start, region_end in smoothing_regions:
        if not 0 < region_start < region_end:
            raise ValueError("Smoothing regions require 0 < start < end")

        xdiff = np.log10(region_end) - np.log10(region_start)
                    
        # Find the index of the left hand side of the smoothed region:
        left_idx = np.argmin(np.abs(frequencies - region_start))
        # Find the index of the right hand side of the smoothed region:
        right_idx = np.argmin(np.abs(frequencies - region_end))

        logger.debug("Smoothing %g..%g Hz with a Hann window", region_start, region_end)
        for channel in channels:
            # Find the value of the left hand side - to do this we take the
            # median of 5 samples up to the left_idx
            left_value = np.median(abs(psds_in[channel].data[max(0, left_idx-5):left_idx]))
            # Find the value of the rigth hand side - again 5 sample median,
            # but this time above the index
            right_value = np.median(abs(psds_in[channel].data[right_idx:right_idx+5]))

            if not np.isfinite([left_value, right_value]).all() or min(left_value, right_value) <= 0:
                raise ValueError("Smoothing endpoints must have positive finite PSD values")

            # Get the y=mx+c values in log-log space (xdiff is consistent for all channels)
            ydiff = np.log10(right_value) - np.log10(left_value)
            slope = ydiff / xdiff
            intercept = np.log10(left_value) - slope * np.log10(region_start)

            log_freqs = np.zeros_like(frequencies)
            nonzero = frequencies > 0
            log_freqs[nonzero] = np.log10(frequencies[nonzero])
            line_section = np.power(10, slope * log_freqs + intercept)
            
            # Blend the log-log interpolation using a Hann window.
            window = hann(right_idx - left_idx)
            smoothed_psd = line_section[left_idx:right_idx] * window + psds[channel].data[left_idx:right_idx] * (1 - window)

            psds[channel].data[left_idx:right_idx] = smoothed_psd

    return psds


def add_smoothing_options(parser):
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
