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

from search_utils import (
    load_sangria_timeseries,
    apply_inpainting
)

logger = logging.getLogger(__name__)

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
    action='store_true',
    help="Generate plots used for testing"
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
data_flen = data_tlen // 2 + 1
data_delta_f = 1. / data_tlen


logger.info("Subtracting mean from data")
data = {
    channel: data[channel] - np.mean(data[channel])
    for channel in args.channels
}

logger.info("Padding Data")
padding_samples = int(args.padding_duration / delta_t)
padded_tlen = data_tlen + 2 * args.padding_duration
padded_idxlen = int(padded_tlen / delta_t)
padded_flen = padded_tlen // 2 + 1
padded_delta_f = 1. / padded_tlen

padded_data = {
    channel:pycbc.types.TimeSeries(
        np.zeros(padded_idxlen, dtype=dtype),
        delta_t=delta_t,
        epoch=data_start_time - args.padding_duration
    )
    for channel in args.channels
}
for channel in args.channels:
    padded_data[channel].data[padding_samples:-padding_samples] = data[channel].data

logger.info("Appolying Gaps")
if args.gaps is not None:
    for start, end in args.gaps:
        start_idx = int((start - padded_data[channel]._epoch) / delta_t)
        end_idx = int((end - padded_data[channel]._epoch) / delta_t)
        padded_data[channel][start_idx:end_idx] = 0

logger.info("Loading PSD")
padded_psd = pycbc.psd.from_txt(
    args.psd_file,
    padded_flen,
    padded_delta_f,
    padded_delta_f,
    is_asd_file=False,
)
psd = pycbc.psd.from_txt(
    args.psd_file,
    data_flen,
    data_delta_f,
    data_delta_f,
    is_asd_file=False,
)

# Get a section of data at the start which contains the pad and 2x the 
# padding duration's worth of data
inpainted_data = copy.deepcopy(padded_data)

pad_section_length = int(3 * args.padding_duration / delta_t)
if pad_section_length % 2:
    pad_section_length += 1
# These will be the lengths needed for the padded section + 2 * duration of data
pad_section_tlen = 3 * args.padding_duration 
pad_section_flen = pad_section_length // 2 + 1
pad_section_delta_f = 1. / pad_section_tlen

pad_section_psd = pycbc.psd.from_txt(
    args.psd_file,
    pad_section_flen,
    pad_section_delta_f,
    pad_section_delta_f,
    is_asd_file=False,
)

for channel in args.channels:

    logger.info("Applying inpainting to pad at the start of the data in channel %s", channel)
    start_pad_section = padded_data[channel][0:pad_section_length]
    inpainted_data[channel].data[0:pad_section_length] = apply_inpainting(
        start_pad_section,
        pad_section_psd,
        0,
        padding_samples,
    )

    logger.info("Applying inpainting to pad at the end of the data in channel %s", channel)
    end_pad_section = padded_data[channel][padded_idxlen - pad_section_length:padded_idxlen]
    inpainted_data[channel].data[padded_idxlen - pad_section_length:padded_idxlen] = apply_inpainting(
        end_pad_section,
        pad_section_psd,
        pad_section_length - padding_samples,
        pad_section_length
    )
    


if args.testing_plots:
    fig, ax = plt.subplots()
    for channel in args.channels:
        ax.plot(
            padded_data[channel].sample_times / 86400,
            padded_data[channel],
            label=f"Padded data {channel}"
        )
        ax.plot(
            data[channel].sample_times / 86400,
            data[channel],
            label=f"Original data {channel}"
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
    fig.savefig('output/padded_data.png')
    plt.close(fig)