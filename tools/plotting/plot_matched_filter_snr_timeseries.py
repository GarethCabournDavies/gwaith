"""Plot full matched-filter SNR timeseries per template.

This tool loads the input data file and bank file, recomputes the matched-filter
SNR for each requested template, and produces one PNG plot per template showing
the full SNR timeseries (in days) with clustered trigger peaks overlaid as stars.

Example::

    python tools/plotting/plot_matched_filter_snr_timeseries.py \
        --input-data-file data.hdf \
        --bank-file bank.hdf \
        --psd-file psd.txt \
        --channel LISA_A \
        --output-directory plots/matched_filter \
        --template-ids 0 1 2
"""

import argparse
import logging
import os

import h5py
import numpy as np
from matplotlib import use

use('Agg')
from matplotlib import pyplot as plt
from scipy.signal import find_peaks

from gwaith.types import FrequencySeries, TimeSeries

from hwilas import add_hwilas_common_options, init_logging
from hwilas.filter import matched_filter_frequencyseries
from hwilas.psd.read import from_txt as psd_from_txt
from hwilas.template_banks.helpers import waveform_from_bank
from hwilas.waveforms.constants import TIME_OFFSET_20_DEGREES
from hwilas.waveforms import (
    AET_CHANNELS,
    FASTBHB_CHANNELS,
    LISA_CHANNELS,
    get_fd_waveform,
)

logger = logging.getLogger(__name__)


def load_arguments():
    parser = argparse.ArgumentParser(
        description="Plot full matched-filter SNR timeseries per template."
    )
    add_hwilas_common_options(parser)

    parser.add_argument(
        "--input-data-file",
        required=True,
        help="HDF5 file containing the detector data (with epoch and delta_t attrs)"
    )
    parser.add_argument(
        "--bank-file",
        required=True,
        help="HDF5 template bank file"
    )
    parser.add_argument(
        "--psd-file",
        required=True,
        help="PSD text file (one per detector)"
    )
    parser.add_argument(
        "--channel",
        required=True,
        help="Detector channel to plot"
    )
    parser.add_argument(
        "--output-directory",
        required=True,
        help="Directory in which to save the plots"
    )
    parser.add_argument(
        "--template-ids",
        type=int,
        nargs='+',
        default=None,
        help="Template IDs to plot. If omitted, all templates are plotted."
    )
    parser.add_argument(
        "--start-time",
        type=float,
        default=None,
        help="Start time for the SNR window (defaults to data start)"
    )
    parser.add_argument(
        "--end-time",
        type=float,
        default=None,
        help="End time for the SNR window (defaults to data end minus padding)"
    )
    parser.add_argument(
        "--f-lower",
        type=float,
        default=1e-6,
        help="Low-frequency cutoff for matched filtering (Hz)"
    )
    parser.add_argument(
        "--snr-threshold",
        type=float,
        default=4.0,
        help="SNR threshold for peak detection"
    )
    parser.add_argument(
        "--cluster-window",
        type=float,
        default=7200.0,
        help="Cluster window in seconds for peak detection"
    )
    parser.add_argument(
        "--waveform-type",
        choices=['fastbhb'],
        default='fastbhb',
        help="Waveform generator to use (default: fastbhb)"
    )
    parser.add_argument(
        "--tdi-generation",
        choices=['1.5', '2'],
        default='1.5',
        help="TDI generation for waveform generation"
    )

    return parser.parse_args()


def main():
    args = load_arguments()
    init_logging(args.verbose)
    logger.info("Starting")

    os.makedirs(args.output_directory, exist_ok=True)

    with h5py.File(args.input_data_file, 'r') as data_file:
        data_epoch = float(data_file.attrs['epoch'])
        delta_t = float(data_file.attrs['delta_t'])
        data_array = data_file[args.channel][:]
        padding_duration_from_file = float(data_file.attrs.get('padding_duration', 0.0))

    data = TimeSeries(data_array, delta_t=delta_t, epoch=data_epoch)

    original_dlen = len(data)
    tlen = original_dlen * delta_t
    delta_f = 1.0 / tlen

    data_start_time = float(data.epoch)
    data_end_time = data_start_time + tlen

    logger.info("Data from %.0f to %.0f", data_start_time, data_end_time)

    if (args.start_time is None) != (args.end_time is None):
        raise ValueError("Provide both --start-time and --end-time, or neither.")

    if args.start_time is not None and args.end_time is not None:
        requested_start_time = float(args.start_time)
        requested_end_time = float(args.end_time)
    else:
        requested_start_time = data_start_time
        requested_end_time = data_end_time - padding_duration_from_file

    valid_start_time = max(requested_start_time, data_start_time)
    valid_end_time = min(requested_end_time, data_end_time)

    if len(data) % 2 == 0:
        data.resize(len(data) + 1)

    padded_dlen = len(data)
    tlen = padded_dlen * delta_t
    delta_f = 1.0 / tlen
    flen = len(data) // 2 + 1

    approximant = 'FastBHB'
    # The FastBHB time-domain generator returns a *time-series* of ``length``
    # samples, so its frequency resolution is 1 / (length * delta_t), which must
    # equal the data's delta_f. Hence length is the number of time samples
    # (padded_dlen), not the FFT bin count.
    wf_length = int(padded_dlen)

    waveform_params_shared = {
        'f_lower': args.f_lower,
        'low-frequency-cutoff': 1e-6,
        'tdi': args.tdi_generation,
        't_offset': TIME_OFFSET_20_DEGREES,
        'cutoff_deltat': 0,
        't_obs_start': tlen,
        'delta_f': delta_f,
        'f_final': 1.0 / delta_t / 2 + delta_f / 2,
        'approximant': approximant,
        'mode_array': [(2, 2)],
        'length': wf_length,
        'tc': tlen,
        'delta_t': delta_t,
    }

    logger.info("Loading PSD")
    psd_raw = psd_from_txt(args.psd_file, flen, delta_f, delta_f, is_asd_file=False)
    psd = FrequencySeries(psd_raw.data, delta_f=delta_f)
    psd.data[0] = np.inf
    psd.data[1] = np.inf

    data_fs = data.to_frequencyseries()

    with h5py.File(args.bank_file, 'r') as bank_file:
        num_templates = len(bank_file['mass1'])

    if args.template_ids is not None:
        template_ids = [t for t in args.template_ids if 0 <= t < num_templates]
    else:
        template_ids = list(range(num_templates))

    logger.info("Plotting %d templates", len(template_ids))

    with h5py.File(args.bank_file, 'r') as bank_file:
        for template_idx in template_ids:
            logger.info("Processing template %d", template_idx)

            bank_wf_params = waveform_from_bank(
                bank_file, template_idx, waveform_params_shared, tlen
            )

            waveform = get_fd_waveform(ifos=[args.channel], **bank_wf_params)

            snr_ts = matched_filter_frequencyseries(
                waveform[args.channel].to_frequencyseries(),
                data_fs,
                psd=psd,
                n_samples=len(data),
                epoch=float(data.epoch),
            )

            start_idx = int(np.ceil((valid_start_time - float(snr_ts.epoch)) / delta_t))
            end_idx = int(np.floor((valid_end_time - float(snr_ts.epoch)) / delta_t)) + 1
            start_idx = max(0, start_idx)
            end_idx = min(len(snr_ts), end_idx)

            if end_idx <= start_idx:
                logger.warning("No valid SNR samples for template %d; skipping", template_idx)
                continue

            snr_times = np.asarray(snr_ts.sample_times)[start_idx:end_idx]
            snr_data = np.asarray(snr_ts.data)[start_idx:end_idx]

            clustered_indices, _ = find_peaks(
                abs(snr_data),
                height=args.snr_threshold,
                distance=int(args.cluster_window / delta_t)
            )

            snrs_out = abs(snr_data[clustered_indices])
            times_out = snr_times[clustered_indices]

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
            ax.set_xlabel('Time (days)')
            ax.set_ylabel('SNR')
            ax.set_title(f"Template {template_idx} ({args.channel})")

            output_path = os.path.join(
                args.output_directory,
                f"snr_timeseries_{args.channel}_{template_idx}.png",
            )
            fig.savefig(output_path, bbox_inches='tight')
            plt.close(fig)
            logger.info("Saved plot to %s", output_path)

    logger.info("Done!")


if __name__ == "__main__":
    main()
