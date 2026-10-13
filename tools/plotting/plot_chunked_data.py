"""Produce frequency-domain plots (spectrum and PSD) of chunked data.

Loads the chunked-data HDF5 files containing processed analysis windows
(e.g. ``mojito_padded_chunk{N}.hdf`` or ``LDC2_sangria_hm_training_padded.hdf``)
and produces two figures per channel, computed over the **analysis window only**
(the 30-day padded section is excluded):

* **Spectrum** -- one-sided FFT amplitude spectrum ``|D(f)|`` vs frequency
  (log-log). All chunks are overlaid on a single axis so that transient features
  (e.g. MBHB coalescences) can be identified across the full data set.
* **PSD** -- Welch estimate (``scipy.signal.welch``) of each chunk's analysis
  window, with all chunks overlaid and a smoothed reference PSD file on top so
  that the noise in each chunk can be compared against the model used for
  whitening. By default, a single "combined" Welch estimate computed over all
  chunks concatenated is also plotted (thick black line); this uses far more
  independent segments than any single chunk and therefore gives a much smoother
  estimate of the noise PSD. Disable with ``--no-combined-psd``.

One pair of figures is produced per channel requested via ``--channels``, so the
same command works for both Mojito and Sangria data (which share the same
LISA_A / LISA_E AET channels).

Example::

    hwilas_plot_chunked_data \
        --data-files output/mojito/round_0/chunk_*/data_processing/*.hdf \
        --psd-file psd_estimation/output/A_mojito_SMOOTHED_PSD.txt \
        --channels LISA_A LISA_E \
        --output-directory plots/chunked/mojito
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
             "(processed channel data). All chunks are overlaid on the "
             "same axes."
    )
    parser.add_argument(
        "--psd-file",
        required=False,
        default=None,
        help="Smoothed reference PSD file to overlay on the Welch PSD plot. "
             "If not given, only the per-chunk Welch estimates are plotted."
    )
    parser.add_argument(
        "--channels",
        nargs='+',
        default=None,
        help="Channels to plot (one pair of figures per channel). "
             "Default: all channels in the data files."
    )
    parser.add_argument(
        "--output-directory",
        required=True,
        help="Directory in which to save the plots. Files are named "
             "<channel>_spectrum.png and <channel>_psd.png."
    )
    parser.add_argument(
        "--welch-window-s",
        type=float,
        default=DEFAULT_WELCH_WINDOW_S,
        help=f"Welch window length in seconds. Default: {DEFAULT_WELCH_WINDOW_S:.0f} "
             f"(1 day)."
    )

    parser.add_argument(
        "--no-combined-psd",
        action="store_true",
        help="Do not plot the combined (all-chunks) Welch PSD estimate. By "
             "default, in addition to the per-chunk Welch estimates, a single "
             "Welch estimate computed over all chunks concatenated is plotted "
             "(thick black line). This uses more independent segments and so "
             "gives a much smoother estimate of the noise PSD."
    )

    return parser.parse_args()


def load_data(data_file, channels):
    """Load channel data and file attributes from a chunked-data HDF5 file.

    Returns ``(data, attrs)`` where ``data`` maps each requested channel to a
    :class:`gwaith.types.TimeSeries` and ``attrs`` holds the relevant file
    attributes (epoch, delta_t, padding duration).
    """
    with h5py.File(data_file, 'r') as f:
        epoch = float(f.attrs["epoch"])
        delta_t = float(f.attrs["delta_t"])

        if channels is None:
            channel_names = list(f.keys())
        else:
            missing = [c for c in channels if c not in f]
            if missing:
                raise ValueError(
                    f"Channel(s) {missing} not found in {data_file}. "
                    f"Available channels: {list(f.keys())}"
                )
            channel_names = list(channels)

        data = {}
        for channel in channel_names:
            data[channel] = TimeSeries(
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

def compute_spectrum(ts):
    """Return ``(freqs, amplitude)`` for the one-sided FFT amplitude spectrum."""
    fs = ts.to_frequencyseries()
    # One-sided: take up to and including the Nyquist bin.
    n = len(fs.data)
    half = n // 2 + 1
    freqs = fs.frequencies[:half]
    amp = np.abs(fs.data[:half])
    return freqs, amp


def compute_welch_psd(ts, window_s):
    """Return ``(freqs, psd)`` from a Welch PSD estimate of the time series."""
    return compute_welch_psd_from_array(ts.data, window_s, ts.delta_t)


def compute_welch_psd_from_array(data, window_s, dt):
    """Return ``(freqs, psd)`` from a Welch PSD estimate of a sample array."""
    nperseg = int(round(window_s / dt))
    if nperseg < 16:
        raise ValueError(
            f"Welch window of {window_s:.3g} s is too short for delta_t={dt}s "
            f"(only {nperseg} samples). Increase --welch-window-s."
        )
    nfft = 1 << int(np.ceil(np.log2(nperseg)))  # next power of two
    freqs, psd = signal.welch(
        data,
        fs=1.0 / dt,
        window='hann',
        nperseg=nperseg,
        noverlap=int(nperseg * WELCH_OVERLAP_FRACTION),
        nfft=nfft,
        detrend='constant',
    )
    return freqs, psd


def _load_reference_psd(psd_file, delta_f, length):
    """Load a reference PSD file and interpolate it to the plot frequency grid.

    Returns ``(freqs, psd)`` arrays on the same grid as the Welch estimate.
    ``low_freq_cutoff`` is set just above zero so that the full band is kept.
    The smoothed PSD files produced by the pipeline are true PSDs (not ASDs),
    hence ``is_asd_file=False``.
    """
    low_freq_cutoff = max(delta_f, 1e-6)
    ref = psd_from_txt(
        psd_file, length=length, delta_f=delta_f,
        low_freq_cutoff=low_freq_cutoff, is_asd_file=False,
    )
    return ref.frequencies, ref.data


def _chunk_label(data_file):
    """Return a short human-readable label for a chunk data file.

    Uses the parent directory name (e.g. ``chunk_3``) when available, falling
    back to the file stem otherwise.
    """
    parent = os.path.basename(os.path.dirname(data_file))
    if parent and parent not in ('.', '..'):
        return parent
    return os.path.splitext(os.path.basename(data_file))[0]


def plot_channel(channel, chunk_data, reference_psd, output_directory,
                 window_s, plot_combined=True):
    """Produce the spectrum and PSD figures for a single channel.

    ``chunk_data`` is an ordered list of ``(label, ts)`` pairs (one per chunk,
    already sliced to the analysis window). ``reference_psd`` is either
    ``(freqs, psd)`` or ``None``. When ``plot_combined`` is true, a single
    Welch estimate over all chunks concatenated is also plotted as a thick
    black line. Saves ``<channel>_spectrum.png`` and ``<channel>_psd.png``
    into ``output_directory``.
    """
    os.makedirs(output_directory, exist_ok=True)

    # Spectrum: all chunk FFTs overlaid, log-log.
    fig, ax = plt.subplots(figsize=(12, 6))
    for label, ts in chunk_data:
        freqs, amp = compute_spectrum(ts)
        mask = freqs > 0
        ax.loglog(freqs[mask], amp[mask], lw=0.8, alpha=0.7, label=label)

    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel(r"$|D(f)|$")
    ax.set_title(f"Chunk spectra -- {channel}")
    ax.grid(True, which='both', alpha=0.3)
    ax.legend(loc='upper right')
    spectrum_path = os.path.join(output_directory, f"{channel}_spectrum.png")
    fig.savefig(spectrum_path, bbox_inches='tight')
    plt.close(fig)
    logger.info("Saved spectrum plot for %s to %s", channel, spectrum_path)

    # PSD: per-chunk Welch estimates overlaid + optional combined estimate.
    fig, ax = plt.subplots(figsize=(12, 6))
    for label, ts in chunk_data:
        freqs, psd = compute_welch_psd(ts, window_s)
        mask = freqs > 0
        ax.loglog(freqs[mask], psd[mask], lw=0.8, alpha=0.7, label=label)

    if plot_combined and len(chunk_data) > 1:
        # Single Welch estimate over all chunks concatenated: uses far more
        # independent segments than any single chunk, so it is much smoother.
        dt = chunk_data[0][1].delta_t
        combined = np.concatenate([ts.data for _, ts in chunk_data])
        freqs, psd = compute_welch_psd_from_array(combined, window_s, dt)
        mask = freqs > 0
        ax.loglog(
            freqs[mask], psd[mask],
            color='black', lw=2.0, alpha=1.0,
            label=f"Combined ({len(chunk_data)} chunks)",
        )

    if reference_psd is not None:
        ref_freqs, ref_psd = reference_psd
        mask = ref_freqs > 0
        ax.loglog(
            ref_freqs[mask], ref_psd[mask],
            color='red', lw=1.5, ls='--',
            label="Reference (smoothed) PSD",
        )

    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel(r"$S_n(f)$")
    ax.set_title(f"Chunk Welch PSDs -- {channel}")
    ax.grid(True, which='both', alpha=0.3)
    ax.legend(loc='upper right')
    psd_path = os.path.join(output_directory, f"{channel}_psd.png")
    fig.savefig(psd_path, bbox_inches='tight')
    plt.close(fig)
    logger.info("Saved PSD plot for %s to %s", channel, psd_path)


def main():
    args = load_arguments()
    init_logging(args.verbose)
    logger.info("Starting")


    # Load every chunk file once, slicing each channel to the analysis window.
    channels = None
    attrs_ref = None
    # channel -> list of (label, sliced TimeSeries)
    chunk_data = {}

    for i, data_file in enumerate(args.data_files):
        logger.info("File %d: %s", i, data_file)
        data, attrs = load_data(data_file, args.channels)
        if channels is None:
            channels = list(data.keys())
        else:
            if set(channels) != set(data.keys()):
                raise ValueError(
                    f"Channels in {data_file} ({list(data.keys())}) do not match "
                    f"the first file ({channels})."
                )

        if attrs_ref is None:
            attrs_ref = attrs
        else:
            if attrs["delta_t"] != attrs_ref["delta_t"]:
                raise ValueError(
                    f"delta_t in {data_file} ({attrs['delta_t']}) does not match "
                    f"the first file ({attrs_ref['delta_t']}). All chunks must "
                    "share the same sampling rate."
                )

        label = _chunk_label(data_file)
        for channel in channels:
            chunk_data.setdefault(channel, []).append((label, data[channel]))

    logger.info(
        "Loaded %d chunk(s) x %d channel(s): %s",
        len(args.data_files), len(channels), channels,
    )

    # Load the reference PSD once, on the grid of a single-chunk Welch estimate.
    reference_psd = None
    if args.psd_file is not None:
        ref_ts = chunk_data[channels[0]][0][1]
        dt = ref_ts.delta_t
        delta_f = 1.0 / (len(ref_ts) * dt)
        length = len(ref_ts) // 2 + 1
        reference_psd = _load_reference_psd(args.psd_file, delta_f, length)

    for channel in channels:
        plot_channel(
            channel,
            chunk_data[channel],
            reference_psd,
            args.output_directory,
            window_s=args.welch_window_s,
            plot_combined=not args.no_combined_psd,
        )

    logger.info("Done! Plots written to %s", args.output_directory)


if __name__ == "__main__":
    main()

