"""Estimate the PSD from processed mojito data.

Accepts one or more chunked-data HDF5 files. When multiple files are given,
each channel is concatenated across all chunks before Welch's method is
applied, so the estimate pools segments from every chunk and is far smoother
than a single-chunk estimate.
"""

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
        nargs="+",
        help="Path(s) to the processed mojito HDF5 file(s) (output of "
             "hwilas_process_mojito_data). When multiple files are given, all "
             "chunks are concatenated per channel before Welch's method is "
             "applied, so the PSD estimate uses far more independent segments "
             "and is much smoother than estimating from a single chunk."
    )
    parser.add_argument(
        "--output-file",
        required=True,
        nargs="+",
        action=KeyValueAction,
        help="Output files for each channel, given as channel:filename pairs"
    )
    parser.add_argument(
        "--unsmoothed-output-file", nargs="+", action=KeyValueAction,
        help="Optional raw Welch PSD outputs as channel:filename pairs (native frequency grid)",
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
    if args.unsmoothed_output_file and set(args.unsmoothed_output_file) - set(channels):
        raise ValueError("Unsmoothed outputs must refer to selected channels")
    logger.debug("Channels: %s", channels)

    # Load every data file, concatenating each channel across files so that the
    # Welch estimate pools segments from all chunks. Files must share the same
    # delta_t and provide every requested channel.
    timeseries = {}
    data_files = args.data_file
    for i, data_file in enumerate(data_files):
        logger.info("Loading processed mojito data %d/%d: %s",
                    i + 1, len(data_files), data_file)
        with h5py.File(data_file, 'r') as f:
            epoch = float(f.attrs.get("epoch", 0.0))
            delta_t = float(f.attrs.get("delta_t", 1.0))

            for channel in channels:
                if channel not in f:
                    raise ValueError(
                        f"Channel {channel} not found in {data_file}. "
                        f"Available: {list(f.keys())}"
                    )
                data_array = np.asarray(f[channel][:], dtype=float)
                if channel in timeseries:
                    # All chunks must share the same sampling rate.
                    existing_dt = timeseries[channel].delta_t
                    if existing_dt != delta_t:
                        raise ValueError(
                            f"delta_t in {data_file} ({delta_t}) does not match "
                            f"the first file ({existing_dt}). All chunks must "
                            "share the same sampling rate."
                        )
                    timeseries[channel] = TimeSeries(
                        np.concatenate([timeseries[channel].data, data_array]),
                        delta_t=delta_t,
                        epoch=epoch,
                    )
                else:
                    timeseries[channel] = TimeSeries(
                        data_array,
                        delta_t=delta_t,
                        epoch=epoch,
                    )
                logger.debug(
                    "Loaded %s from %s: %d samples (total so far: %d), "
                    "delta_t=%.6f, epoch=%.2f",
                    channel, data_file, len(data_array),
                    len(timeseries[channel]), delta_t, epoch,
                )

    tlen = len(timeseries[channels[0]])
    delta_t = timeseries[channels[0]].delta_t
    delta_f = 1.0 / (tlen * delta_t)

    logger.info(
        "Estimating PSDs from %d file(s), total duration per channel: %.3g s",
        len(data_files), tlen * delta_t,
    )
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
    import os
    for channel in channels:
        output_path = args.output_file[channel]
        logger.debug("Saving PSD for %s to %s", channel, output_path)
        out_dir = os.path.dirname(output_path)
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)
        # Downstream PSD readers expect frequency/PSD columns in plain text.
        np.savetxt(output_path,
                   np.column_stack((np.asarray(psds[channel].sample_frequencies),
                                    np.asarray(psds[channel].data))),
                   header="frequency_Hz PSD")

    for channel, output_path in (args.unsmoothed_output_file or {}).items():
        out_dir = os.path.dirname(output_path)
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)
        raw = full_psds[channel]
        logger.debug("Saving unsmoothed %s PSD to %s", channel, output_path)
        np.savetxt(output_path, np.column_stack((np.asarray(raw.sample_frequencies), raw.data)),
                   header="frequency_Hz PSD_unsmoothed")

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
            ax.loglog(
                psds[channel].sample_frequencies,
                np.sqrt(psds[channel].data)
            )
            ax.set_xlabel('Frequency (Hz)')
            ax.set_ylabel('Strain (Hz^{-1/2})')
            ax.set_title(f'Interpolated PSD: {channel}')
            fig.savefig(os.path.join(args.testing_plots, f'{channel}_interp_psd.png'))
            plt.close(fig)

    logger.info("Done!")

if __name__ == "__main__":
    main()