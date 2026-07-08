"""
This is a script to run the PyCBC matched filtering using a basic template bank search
"""

# Import necessary libraries
import h5py  # Library for interacting with HDF5 files
import argparse  # Library for parsing command-line arguments
import logging  # Library for logging messages
import numpy as np
from matplotlib import pyplot as plt


# Import specific modules from the PyCBC library
import pycbc
import pycbc.psd
from pycbc.events.threshold_cpu import parallel_thresh_cluster
from pycbc.types import TimeSeries
# I am not sure if we need to remove these if people are so against PyCBC's licence

import ldc.io.hdf5 as hdfio

from search_utils import (
    load_sangria_timeseries,
    waveform_from_bank,
)

logger = logging.getLogger(__name__)

# Set up argument parser for command-line arguments
parser = argparse.ArgumentParser()
pycbc.add_common_pycbc_options(parser)

# Add argument for the PSD files (required) with custom action for multiple detectors
parser.add_argument(
    '--psd-file',
    required=True,
)

# Add argument for the bank file (required)
parser.add_argument('--bank-file', required=True)

# Add argument for the data file (required)
parser.add_argument('--input-data-file', required=True)
parser.add_argument('--channel', required=True)

# # Add argument for the start/end times taht will be searched within the data file (required)

parser.add_argument(
    '--start-time',
    required=True,
    type=float,
    help="This is the start time being considered for triggers "
         "from the matched filter. Uses data start time if this is smaller"
)

parser.add_argument(
    '--end-time',
    required=True,
    type=float,
    help="This is the end time being considered for triggers "
         "from the matched filter. Uses data end time if this is larger"
)

parser.add_argument(
    '--padding-time',
    type=float,
    default=86400.,
    help="Time to be removed when considering the "
         "corruption length caused by the finite filter. "
         "In seconds, defaults to 1 day"
)

parser.add_argument(
    '-trigger-time-separation',
    type=float,
    default=3600,
    help="SNR peaks (triggers) must be separated by this amount"
)

parser.add_argument(
    "--snr-threshold",
    type=float,
    default=4,
    help="Remove any SNR peaks which have SNR values below this threshold"
)

parser.add_argument(
    "--cluster-window",
    type=float,
    default=7200,
    help="Window to use when clustering triggers, in seconds. Default 2 hours"
)

parser.add_argument("--template-fraction-range", default="0/1",
                    help="Optional, analyze only part of template bank. Format"
                         " PART/NUM_PARTS")

def parse_template_range(num_templates, rangestr):
    part = int(rangestr.split('/')[0])
    pieces = int(rangestr.split('/')[1])
    tmin = int(num_templates / float(pieces) * part)
    tmax = int(num_templates / float(pieces) * (part+1))
    return tmin, tmax

# Add argument for the lower frequency cutoff with a default value
parser.add_argument('--f-lower', type=float, default=1e-6)

parser.add_argument(
    "--output-file",
    required=True,
    help="File to store output peaks in SNR"
)

parser.add_argument(
    "--testing-plots",
    action='store_true',
    help="Generate plots used for testing"
)

# Parse the command-line arguments provided by the user
args = parser.parse_args()

# Initialize logging for the PyCBC library
if args.verbose is None:
    # Lets have info logging on by default for now
    pycbc.init_logging(1)
else:
    # Adding --verbose adds to the debug logging
    pycbc.init_logging(args.verbose + 1)

# First, load in the data. Get from the file the information on when it starts, its length and its delta t
# For now, this uses the sangria dataset, but we want to build a more sophisticated "get data" function, so that
# we can take in e.g. mojito datasets
data = load_sangria_timeseries(
    args.input_data_file,
)
data = data[args.channel]

data_epoch = data._epoch
delta_t = data.delta_t
original_dlen = len(data)
tlen = original_dlen * delta_t
delta_f = 1. / tlen
flen = tlen // 2 + 1

data_start_time = float(data._epoch)
data_end_time = data_start_time + tlen

logger.info(f'Data from {data_start_time:.0f} to {data_end_time:.0f}')
logger.debug("Original data length: %d", original_dlen)


# Set the defaults required for the waveform parameters
waveform_params_shared = {
    'f_lower': args.f_lower,
    'low-frequency-cutoff': 1e-6, 
    'tdi': '1.5',
    't_offset': 0,
    'cutoff_deltat': 0,
    't_obs_start': int(tlen * delta_t + 0.5),
    'delta_f': delta_f,
    'f_final': 1. / delta_t / 2 + delta_f / 2, # This is a hack to ensure we get the point at f_final returned
    'approximant':'BBHX_PhenomD',
    'mode_array':  [(2,2)],
    'length': flen,
}

# subtracting the mean - I _think_ we want to do this, not 100% sure.
# This is probbaly useful for parallelilsing in time
mean_val = np.mean(data)
data = data - mean_val

# Plot the data:
if args.testing_plots is not None:
    fig, ax = plt.subplots()
    ax.plot((data.sample_times - data_start_time) / 86400, data, label=f"data {args.channel}")
    ax.legend(loc='upper right')
    ax.set_xlim(0, tlen / 86400)
    ax.set_xlabel('Time from epoch (days)')

    fig.savefig('output/data.png')
    plt.close(fig)


padding_time = 30 * 86400 # 30 days in seconds
padding_samples = int(padding_time / delta_t)

logger.info("Padding data with %d samples at start and end", padding_samples)
padded_data = pycbc.types.TimeSeries(
    np.zeros(len(data) + 2 * padding_samples, dtype=data.dtype),
    delta_t=data.delta_t,
    epoch=data._epoch - padding_time
)
padded_data[padding_samples:-padding_samples] = data
data = padded_data

# Ensure data has an odd length to avoid FFT length mismatches
if len(data) % 2 == 0:
    data.resize(len(data) + 1)

padded_dlen = len(data)
tlen = padded_dlen * delta_t
delta_f = 1. / tlen
flen = len(data) // 2 + 1

logger.info("Loading PSD")
psd = pycbc.psd.from_txt(
    args.psd_file,
    flen,
    delta_f,
    delta_f,
    is_asd_file=False,
)


logger.info(f"Beginning filtering with bank %s", args.bank_file)

output_triggers = {
    "template_id": [],
    "end_time": [],
    "snr": [],
}

with h5py.File(args.bank_file, 'r') as bank_file:
    num_templates = len(bank_file['mass1'])

tmin, tmax = parse_template_range(num_templates, args.template_fraction_range)
logger.info("Using template range %d to %d out of %d", tmin, tmax, num_templates)

with h5py.File(args.bank_file, 'r') as bank_file:
    for template_idx in range(tmin, tmax):
        logger.info("Processing template %d", template_idx)
        bank_wf_params = waveform_from_bank(
            bank_file,
            template_idx,
            waveform_params_shared,
            tlen
        )

        logger.debug('Getting SNR series')

        waveform = pycbc.waveform.get_fd_det_waveform(
            ifos=[args.channel], **bank_wf_params
        )

        snr_ts = pycbc.filter.matched_filter(
            waveform[args.channel],
            data,
            psd=psd
        )

        # Zero out the corrupted data at the start and end of the time series
        # so that we don't find any triggers there
        start_idx = padding_samples
        end_idx = len(snr_ts) - padding_samples
        snr_ts_data = snr_ts.data.copy()
        snr_ts_data[0:start_idx] = 0
        snr_ts_data[end_idx:] = 0
        
        # Now do the findchirp-style clustering on the full timeseries
        n_triggers = len(snr_ts_data)
        snrs_out = np.zeros(n_triggers, dtype=np.complex64)
        indices_out = np.zeros(n_triggers, dtype=np.uint32)

        cluster_out = parallel_thresh_cluster(
            snr_ts_data.astype(np.complex64),
            n_triggers,
            snrs_out,
            indices_out,
            np.float32(args.snr_threshold),
            np.uint32(args.cluster_window // delta_t),
            len(snr_ts_data),  # segment size - this is just the whole timeseries
        )

        clustered_indices = indices_out[:cluster_out]
        snrs_out = abs(snr_ts_data[clustered_indices])
        times_out = snr_ts.sample_times[clustered_indices]

        logger.debug("%d triggers remain after clustering", cluster_out)

        if args.testing_plots:
            fig, ax = plt.subplots()
            ax.plot(snr_ts.sample_times / 86400, abs(snr_ts.data))
            ax.scatter(
                times_out / 86400,
                snrs_out,
                c='y',
                marker='*',
            )
            ax.set_xlabel('Time (days)')
            ax.set_ylabel('SNR')
            fig.savefig(f'output/snr_ts_{args.channel}.png')
            plt.close(fig)

        output_triggers['template_id'] += [template_idx] * cluster_out
        output_triggers['end_time'] += times_out.tolist()
        output_triggers['snr'] += snrs_out.tolist()

output_triggers['template_id'] = np.array(output_triggers['template_id'], dtype=np.int32)
output_triggers['end_time'] = np.array(output_triggers['end_time'], dtype=np.float32)
output_triggers['snr'] = np.array(output_triggers['snr'], dtype=np.float32)


logger.info("Outputting to file")
with h5py.File(args.output_file, 'w') as output_trigger_file:
    for key in output_triggers.keys():
        output_trigger_file.create_dataset(key, data=output_triggers[key], compression='gzip')

logger.info('Done!')