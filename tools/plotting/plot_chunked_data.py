"""Produce frequency-domain plots (spectrum and PSD) of chunked data.

This tool loads a chunked-data HDF5 file produced by ``hwilas_split_pad_data``
(e.g. ``mojito_padded_chunk{N}.hdf`` or ``LDC2_sangria_hm_training_padded.hdf``)
and produces two frequency-domain figures per channel, computed over the
**analysis window only** (the 30-day padded section is excluded):

* **Spectrum** -- one-sided FFT amplitude spectrum ``|D(f)|`` vs frequency
  (log-log).
* **PSD** -- Welch estimate (``scipy.signal.welch``) over the analysis window
  (log-log), optionally overlaid with a smoothed reference PSD file so that the
  chunk's noise can be compared against the model used for whitening.

Example::

    python tools/plotting/plot_chunked_data.py \
        --data-file output/mojito/round_0/chunk_0/data_processing/mojito_padded_chunk0.hdf \
        --psd-file psd_estimation/output/A_mojito_SMOOTHED_PSD.txt \
        --channels LISA_A \
        --analysis-start-time 0 \
        --analysis-end-time 10542720 \
        --output-directory plots/chunked/mojito/chunk_0
"""

import argparse
import logging
import os

import h5py
import numpy as np
from scipy import signal

from matplotlib import use

use('Agg')
from matplotlib import pyplot as plt

from gwaith.types import TimeSeries

from hwilas import add_hwilas_common_options, init_logging
from hwilas.psd.read import from_txt as psd_from_txt

logger = logging.getLogger(__name__)

CHANNEL_COLORS = {
    'LISA_A': 'tab:blue',
    'LISA_E': 'tab:orange',
    'LISA_T': 'tab:green',
}

# Default Welch window length (seconds). 1 day gives a smooth PSD estimate for
# the ~4-month analysis windows used in the search.
DEFAULT_WELCH_WINDOW_S = 86400.0
# Fraction of overlap between consecutive Welch segments.
WELCH_OVERLAP_FRACTION = 0.5


def load_arguments():
    parser = argparse.ArgumentParser(
        description="Produce frequency-domain plots (spectrum and PSD) of chunked data."
    )
    add_hwilas_common_options(parser)

    parser.add_argument(
        "--data-files",
        required=True,
        nargs="+",
        help="HDF5 files of chunked, padded + inpainted data "
             "(output of hwilas_split_pad_data)"
    )
    parser.add_argument(
        "--psd-file",
        required=False,
        default=None,
        help="Smoothed reference PSD file to overlay on the Welch PSD plot. "
             "If not given, only the Welch estimate is plotted."
    )
    parser.add_argument(
        "--channel",
        default=None,
        help="Channel to plot."
    )

    parser.add_argument(
        "--output",
        required=True,
        help="File in which to save the plot."
    )


    return parser.parse_args()


def load_data(data_file, channel):
    """Load channel data and file attributes from a chunked-data HDF5 file.

    Returns ``(data, attrs)`` where ``data`` maps each requested channel to a
    :class:`gwaith.types.TimeSeries` and ``attrs`` holds the relevant file
    attributes (epoch, delta_t, padding duration).
    """
    with h5py.File(data_file, 'r') as f:
        epoch = float(f.attrs["epoch"])
        delta_t = float(f.attrs["delta_t"])

        data = TimeSeries(
            np.asarray(f[channel], dtype=np.float64),
            epoch=epoch,
            delta_t=delta_t,
        )

        attrs = {
            "epoch": epoch,
            "delta_t": delta_t,
            "padding_duration": float(f.attrs.get("padding_duration", 0.0)),
        }
    return data, attrs


def slice_to_analysis_window(ts, analysis_start_time, analysis_end_time):
    """Slice a TimeSeries to the analysis window (relative to the file epoch).

    The returned TimeSeries keeps the same ``delta_t`` so that FFT / Welch
    frequency axes are correct. The sample times are shifted so that they are
    relative to the *start* of the analysis window.
    """
    dt = ts.delta_t
    n = len(ts.data)

    i0 = int(round(analysis_start_time / dt))
    i1 = int(round(analysis_end_time / dt))
    i0 = max(0, min(i0, n))
    i1 = max(i0, min(i1, n))

    if i1 - i0 < 2:
        raise ValueError(
            f"Analysis window [{analysis_start_time:.3g}, {analysis_end_time:.3g}] s "
            f"contains fewer than two samples (got {i1 - i0}). Check "
            "--analysis-start-time / --analysis-end-time against the data file."
        )

    sliced = ts.data[i0:i1]
    # New epoch so that sample_times are relative to the analysis-window start.
    new_epoch = ts.epoch + analysis_start_time
    return TimeSeries(sliced, epoch=new_epoch, delta_t=dt)


def compute_spectrum(ts):
    """Return ``(freqs, amplitude)`` for the one-sided FFT amplitude spectrum."""
    fs = ts.to_frequencyseries()
    # One-sided: take up to and including the Nyquist bin.
    n = len(fs.data)
    half = n // 2 + 1
    freqs = fs.frequencies[:half]
    amp = np.abs(fs.data[:half])
    return freqs, amp



def _load_reference_psd(psd_file, delta_f, length):
    """Load a reference PSD file and interpolate it to the plot frequency grid.

    Returns ``(freqs, psd)`` arrays on the same grid as the Welch estimate.
    ``low_freq_cutoff`` is set just above zero so that the full band is kept.
    """
    low_freq_cutoff = max(delta_f, 1e-6)
    ref = psd_from_txt(psd_file, length=length, delta_f=delta_f,
                       low_freq_cutoff=low_freq_cutoff)
    return ref.frequencies, ref.data


def main():
    args = load_arguments()
    init_logging(args.verbose)
    logger.info("Starting")

    fig, ax = plt.subplots()

    for i, data_file in enumerate(args.data_files):
        logger.info("File %d", i)
        data, attrs = load_data(data_file, args.channel)

        # Convert to a frequency series
        freqs, amp = compute_spectrum(data)

        mask = freqs > 0
        ax.loglog(freqs[mask], amp[mask],lw=1.5)
        delta_f = np.diff(freqs)[0]
        length = len(freqs)

    ref_freqs, ref_psd = _load_reference_psd(args.psd_file, delta_f, length)
    ax.loglog(
        ref_freqs,
        ref_psd,
        color='red',
        lw=1.5, ls='--',
        label="Reference (smoothed) PSD"
    )

    fig.savefig(args.output, bbox_inches='tight')
    plt.close(fig)
    # logger.info("Saved PSD plot for %s to %s", channel, output_path)

    logger.info("Done!")


if __name__ == "__main__":
    main()

