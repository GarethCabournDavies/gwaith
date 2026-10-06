"""
This is a script to run matched filtering using a basic template bank search
"""

# Import necessary libraries
import argparse  # Library for parsing command-line arguments
import logging  # Library for logging messages
import h5py  # Library for interacting with HDF5 files
import numpy as np

from scipy.signal import find_peaks
from gwaith.types import TimeSeries, FrequencySeries

from hwilas import add_hwilas_common_options, init_logging
from hwilas.filter import matched_filter_frequencyseries
from hwilas.psd.read import from_txt as psd_from_txt
from hwilas.template_banks.helpers import waveform_from_bank
from hwilas.waveforms.constants import TIME_OFFSET_20_DEGREES
from hwilas.waveforms import (
    FASTBHB_CHANNELS,
    get_fd_waveform,
)
from hwilas.waveforms.unified_generation import generate_waveform

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

    parser.add_argument(
        '--waveform-type',
        choices=['fastbhb'],
        default='fastbhb',
        help="Waveform generator to use for the template bank. 'fastbhb' uses the "
             "LDC FastBHB (lisabeta) time-domain generator, which returns AET channels "
             "(LISA_A/LISA_E/LISA_T). The TDI generation is set via --tdi-generation."
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
        "--plot-snr-dir",
        type=str,
        default=None,
        help="Directory to optionally save plots of the SNR timeseries for each processed template."
    )

    # Parse the command-line arguments provided by the user
    args = parser.parse_args()

    init_logging(args.verbose)

    logger.info("Starting")

    # First, load in the data. Get from the file the information on when it starts, its length and its delta t
    # The input is a sangria dataset read directly from an HDF5 file.
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
        # The data is asymmetrically padded, with the padding at the end.
        # The analysis window is from the start of the data to the start
        # of the padding.
        requested_start_time = data_start_time
        requested_end_time = data_end_time - padding_duration_from_file

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
    approximant = 'FastBHB'
    # The FastBHB time-domain generator returns a *time-series* of ``length``
    # samples spanning [start_time, start_time + length * delta_t). Its frequency
    # resolution must match the data's delta_f, so ``length`` is the number of time
    # samples (padded_dlen), not the FFT bin count (flen).
    wf_length = int(padded_dlen)

    waveform_params_shared = {
        'f_lower': args.f_lower,
        'low-frequency-cutoff': 1e-6,
        'tdi': args.tdi_generation,
        't_offset': TIME_OFFSET_20_DEGREES,
        'cutoff_deltat': 0,
        't_obs_start': tlen,
        'delta_f': delta_f,
        'f_final': 1. / delta_t / 2 + delta_f / 2, # This is a hack to ensure we get the point at f_final returned
        'approximant': approximant,
        'mode_array': [(2, 2)],
        'length': wf_length,
        'tc': tlen - 86400.0,
        # True data sampling interval (seconds). Used to build the detector-frame
        # time grid; without it the grid would be derived from the FFT parameters
        # and could mismatch the data.
        'delta_t': delta_t,
    }

    # The FastBHB time-domain generator returns AET channels (LISA_A/E/T).
    if args.channel not in FASTBHB_CHANNELS:
        raise ValueError(
            f"Channel {args.channel} is not a FastBHB AET channel. "
            f"Expected one of {FASTBHB_CHANNELS}."
        )

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

    logger.info("Beginning filtering with bank %s", args.bank_file)

    output_triggers = {
        "template_id": [],
        "end_time": [],
        "snr": [],
    }
    # Templates that were processed but did not yield any trigger rows (e.g. no
    # valid SNR samples in the filtering window). Logged at the end so it is
    # clear which templates are absent from the output file when a run is
    # incomplete (partial range, OOM kill mid-loop, or per-template skips).
    skipped_templates = []

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

            # NOTE: This uses get_fd_waveform directly rather than the unified
            # generate_waveform() interface used in plotting scripts. This is because
            # matched filter search works with template banks (where parameters are
            # pre-built by waveform_from_bank) rather than catalog entries.
            # Both paths handle waveform generation identically; the unified interface
            # is primarily designed for catalog-based workflows.
            # The waveform is generated in the time domain (see get_fd_waveform)
            # so that it can be (over)whitened before the template loop. It is
            # converted back to the frequency domain here for matched filtering.
            # ``get_fd_waveform`` dispatches on the approximant: BBHx returns AET
            # channels while the PhenomX time-domain path returns XYZ channels, so
            # the requested channel must match the chosen waveform type (validated
            # above).
            waveform = get_fd_waveform(
                ifos=[args.channel], **bank_wf_params
            )

            # The waveform is generated with tc shifted earlier (by 1 day) to
            # prevent the lisabeta boundary crash. A frequency-domain phase shift
            # rolls the merger time back to the end of the data (tlen), so the SNR
            # timeseries peak times are correct.
            wf_fs = waveform[args.channel].to_frequencyseries()
            freqs = np.asarray(wf_fs.sample_frequencies)
            
            # Shifting the template backwards in time by dt corresponds to
            # multiplying by exp(+2j * pi * f * dt) in frequency domain for our convention,
            # but we want to shift the effective merger *forward* by 86400s (so it occurs later).
            # A forward shift of the template by dt means h(t - dt), which in FD is exp(-2j*pi*f*dt).
            wf_fs.data *= np.exp(-2j * np.pi * freqs * 86400.0)

            logger.debug('Computing SNR')
            snr_ts = matched_filter_frequencyseries(
                wf_fs,
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
                logger.warning(
                    "No valid SNR samples remain for template %d; skipping. "
                    "This template will be absent from the output file.",
                    template_idx,
                )
                skipped_templates.append(template_idx)
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

            if len(clustered_indices) == 0:
                # No peak cleared the SNR threshold, so this template contributes
                # no rows to the output. Record it so the summary below can name it.
                skipped_templates.append(template_idx)

            if args.plot_snr_dir is not None:
                import os
                from matplotlib import use
                use('Agg')
                from matplotlib import pyplot as plt
                
                os.makedirs(args.plot_snr_dir, exist_ok=True)
                fig, ax = plt.subplots(figsize=(14, 5))
                ax.plot(snr_times / 86400.0, abs(snr_data))
                if len(times_out) > 0:
                    ax.scatter(
                        times_out / 86400.0,
                        snrs_out,
                        c='y',
                        marker='*',
                        zorder=50,
                    )
                if args.snr_threshold is not None:
                    ax.axhline(args.snr_threshold, color='r', linestyle='--', label=f'Threshold ({args.snr_threshold})')
                    ax.legend()
                    
                ax.set_xlabel('Time (days)')
                ax.set_ylabel('SNR')
                ax.set_title(f"Template {template_idx} ({args.channel})")

                output_path = os.path.join(
                    args.plot_snr_dir,
                    f"snr_timeseries_{args.channel}_{template_idx}.png",
                )
                fig.savefig(output_path, bbox_inches='tight')
                plt.close(fig)
                logger.info("Saved SNR plot to %s", output_path)

            output_triggers['template_id'] += [template_idx] * len(clustered_indices)
            output_triggers['end_time'] += times_out.tolist()
            output_triggers['snr'] += snrs_out.tolist()

    output_triggers['template_id'] = np.array(output_triggers['template_id'], dtype=np.int32)
    output_triggers['end_time'] = np.array(output_triggers['end_time'], dtype=np.float32)
    output_triggers['snr'] = np.array(output_triggers['snr'], dtype=np.float32)

    # Summarise which templates are absent from the output file. A template is
    # "missing" if it was processed but produced no rows (no valid SNR samples or
    # no peak above threshold). This makes incomplete runs diagnosable: compare
    # this list against the logged "Using template range X to Y" line to see which
    # templates never ran at all (e.g. the job was OOM-killed mid-loop).
    processed_ids = set(range(tmin, tmax))
    present_ids = set(int(x) for x in output_triggers['template_id'])
    missing_from_output = sorted(processed_ids - present_ids)
    n_processed = len(processed_ids)
    if missing_from_output:
        logger.warning(
            "Matched filter incomplete: %d of %d templates produced no trigger rows. "
            "Missing template ids (absent from output file): %s",
            len(missing_from_output), n_processed, missing_from_output,
        )
    else:
        logger.info(
            "All %d templates in range [%d, %d) produced at least one trigger row.",
            n_processed, tmin, tmax,
        )

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