"""
This is a script to run matched filtering using a basic template bank search
"""

# Import necessary libraries
import argparse  # Library for parsing command-line arguments
import logging  # Library for logging messages
from matplotlib import pyplot as plt
import h5py  # Library for interacting with HDF5 files
import numpy as np

from scipy.signal import find_peaks
from gwaith.types import TimeSeries, FrequencySeries

from hwilas import add_hwilas_common_options, init_logging
from hwilas.filter import matched_filter_frequencyseries
from hwilas.psd.read import from_txt as psd_from_txt
from hwilas.template_banks.helpers import waveform_from_bank
from hwilas.waveforms import get_fd_waveform

logger = logging.getLogger(__name__)

def main():

    # Set up argument parser for command-line arguments
    parser = argparse.ArgumentParser()
    add_hwilas_common_options(parser)

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
        required=False,
        type=float,
        help="This is the start time being considered for triggers "
            "from the matched filter. Uses data start time if this is smaller"
    )

    parser.add_argument(
        '--end-time',
        required=False,
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

    parser.add_argument(
        '--tdi-generation',
        choices=['1.5','2'],
        default="1.5",
        help="Which generation of TDI to use when generating the waveform"
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
        help="Generate plots used for testing in this directory"
    )

    # Parse the command-line arguments provided by the user
    args = parser.parse_args()

    init_logging(args.verbose)

    logger.info("Starting")

    # First, load in the data. Get from the file the information on when it starts, its length and its delta t
    # For now, this uses the sangria dataset, but we want to build a more sophisticated "get data" function, so that
    # we can take in e.g. mojito datasets
    with h5py.File(args.input_data_file, 'r') as data_file:
        data_epoch = data_file.attrs['epoch']
        delta_t = data_file.attrs['delta_t']
        data = TimeSeries(
            data_file[args.channel][:],
            delta_t=delta_t,
            epoch=data_epoch,
        )
        padding_duration_from_file = float(data_file.attrs.get('padding_duration', 0.0))

    original_dlen = len(data)
    tlen = original_dlen * delta_t
    delta_f = 1. / tlen

    data_start_time = float(data.epoch)
    data_end_time = data_start_time + tlen

    logger.info(f'Data from {data_start_time:.0f} to {data_end_time:.0f} (including padding)')
    logger.debug("Original data length: %d", original_dlen)
    if (args.start_time is None) != (args.end_time is None):
        raise ValueError("Provide both --start-time and --end-time, or neither.")

    if args.start_time is not None and args.end_time is not None:
        requested_start_time = float(args.start_time)
        requested_end_time = float(args.end_time)
    else:
        effective_padding = padding_duration_from_file
        requested_start_time = data_start_time + effective_padding
        requested_end_time = data_end_time - effective_padding

    valid_start_time = max(requested_start_time, data_start_time)
    valid_end_time = min(requested_end_time, data_end_time)
    if valid_end_time <= valid_start_time:
        raise ValueError(
            f"Invalid analysis window [{valid_start_time}, {valid_end_time}]. "
            "Check start/end inputs or the padding_duration attribute in the input file."
        )

    logger.info(
        "Filtering window: %.0f to %.0f",
        valid_start_time,
        valid_end_time,
    )

    # valid_data_start = max(args.start_time, data_start_time + padding_duration)
    # valid_data_end = min(args.end_time, data_end_time + padding_duration)

    # Ensure data has an odd length to avoid FFT length mismatches
    if len(data) % 2 == 0:
        data.resize(len(data) + 1)

    padded_dlen = len(data)
    tlen = padded_dlen * delta_t
    delta_f = 1. / tlen
    flen = len(data) // 2 + 1

    # Set the defaults required for the waveform parameters.
    # Use the final (possibly padded) data length so FFT sizes are consistent.
    waveform_params_shared = {
        'f_lower': args.f_lower,
        'low-frequency-cutoff': 1e-6,
        'tdi': args.tdi_generation,
        't_offset': 0, # This is an internal BBHx parameter, not related to physical duration
        'cutoff_deltat': 0,
        't_obs_start': 30 * 86400.0, # Generate 30 days of physical waveform
        'delta_f': delta_f,
        'f_final': 1. / delta_t / 2 + delta_f / 2, # This is a hack to ensure we get the point at f_final returned
        'approximant': 'BBHX_PhenomHM',
        'mode_array': [(2, 2)],
        'length': int(flen),
    }

    logger.info("Loading PSD")
    psd = psd_from_txt(
        args.psd_file,
        flen,
        delta_f,
        delta_f,
        is_asd_file=False,
    )

    psd = FrequencySeries(
        psd.data,
        delta_f=delta_f,
    )

    # To avoid a division by zero error in the matched filter, we set the
    # low frequency bins of the PSD to infinity.
    psd.data[0] = np.inf
    psd.data[1] = np.inf

    data_fs = data.to_frequencyseries()

    if args.testing_plots is not None:
        whitened_data = (data_fs / psd).to_timeseries()
        fig, ax = plt.subplots()
        ax.plot(whitened_data.sample_times, whitened_data, label=f"whitened data {args.channel}")
        ax.legend(loc='upper right')
        fig.savefig(f'{args.testing_plots}/whitened_data_{args.channel}.png')
        plt.close(fig)

    logger.info("Beginning filtering with bank %s", args.bank_file)

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

            logger.debug('Getting waveform')

            waveform = get_fd_waveform(
                ifos=[args.channel], **bank_wf_params
            )

            logger.debug('Computing SNR')
            snr_ts = matched_filter_frequencyseries(
                waveform[args.channel],
                data_fs,
                psd=psd,
                n_samples=len(data),
                epoch=float(data.epoch),
            )

            # Restrict to a contiguous valid time interval by truncation.
            # This avoids artificially flat zero regions in diagnostic plots.
            start_idx = int(np.ceil((valid_start_time - float(snr_ts.epoch)) / delta_t))
            end_idx = int(np.floor((valid_end_time - float(snr_ts.epoch)) / delta_t)) + 1
            start_idx = max(0, start_idx)
            end_idx = min(len(snr_ts), end_idx)

            if end_idx <= start_idx:
                logger.debug("No valid SNR samples remain for template %d", template_idx)
                continue

            snr_times = np.asarray(snr_ts.sample_times)[start_idx:end_idx]
            snr_data = np.asarray(snr_ts.data)[start_idx:end_idx]
            
            # Find peaks in the SNR timeseries
            clustered_indices, _ = find_peaks(
                abs(snr_data),
                height=args.snr_threshold,
                distance=int(args.cluster_window / delta_t)
            )

            snrs_out = abs(snr_data[clustered_indices])
            times_out = snr_times[clustered_indices]
            logger.debug("%d triggers remain after clustering", len(clustered_indices))

            if args.testing_plots is not None:
                fig, ax = plt.subplots()
                ax.plot(snr_times / 86400, abs(snr_data))
                ax.scatter(
                    times_out / 86400,
                    snrs_out,
                    c='y',
                    marker='*',
                    zorder=50,
                )
                ax.set_xlabel('Time (days)')
                ax.set_ylabel('SNR')
                fig.savefig(f'{args.testing_plots}/snr_timeseries_{args.channel}_{template_idx}.png')
                plt.close(fig)

            output_triggers['template_id'] += [template_idx] * len(clustered_indices)
            output_triggers['end_time'] += times_out.tolist()
            output_triggers['snr'] += snrs_out.tolist()

    output_triggers['template_id'] = np.array(output_triggers['template_id'], dtype=np.int32)
    output_triggers['end_time'] = np.array(output_triggers['end_time'], dtype=np.float32)
    output_triggers['snr'] = np.array(output_triggers['snr'], dtype=np.float32)


    logger.info("Outputting to file")
    with h5py.File(args.output_file, 'w') as output_trigger_file:
        for key, value in output_triggers.items():
            output_trigger_file.create_dataset(
                key,
                data=value,
                compression='gzip'
            )

    logger.info('Done!')

if __name__ == '__main__':
    main()