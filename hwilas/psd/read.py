"""Read PSD/ASD data from text files and build frequency series."""

from __future__ import annotations

import logging

import numpy as np
from pycbc.types import FrequencySeries

logger = logging.getLogger(__name__)


def from_numpy_arrays(freq_data, noise_data, length, delta_f, low_freq_cutoff):
    """Interpolate PSD data to a target frequency resolution and length."""
    freq_data = np.asarray(freq_data, dtype=float)
    noise_data = np.asarray(noise_data, dtype=float)

    if freq_data.ndim != 1 or noise_data.ndim != 1:
        raise ValueError("freq_data and noise_data must be 1D arrays")
    if freq_data.size != noise_data.size:
        raise ValueError("freq_data and noise_data must have the same length")
    if freq_data.size < 2:
        raise ValueError("Need at least two PSD samples for interpolation")

    if freq_data[0] > low_freq_cutoff:
        raise ValueError(
            f"Lowest frequency in input PSD data ({freq_data[0]} Hz) is "
            f"higher than requested low-frequency cutoff ({low_freq_cutoff} Hz)"
        )

    positive = freq_data > 0
    if not np.any(positive):
        raise ValueError("Input PSD frequencies must include positive values")
    freq_data = freq_data[positive]
    noise_data = noise_data[positive]

    kmin = int(low_freq_cutoff / delta_f)
    flow = kmin * delta_f

    data_start = np.searchsorted(freq_data, flow) - 1
    data_start = max(0, data_start)
    freq_data = freq_data[data_start:]
    noise_data = noise_data[data_start:]

    if (length - 1) * delta_f > freq_data[-1]:
        logger.warning(
            "Requested number of samples exceeds highest available PSD frequency; "
            "using max available frequency instead (requested %f Hz, available %f Hz)",
            (length - 1) * delta_f,
            freq_data[-1],
        )
        length = int(freq_data[-1] / delta_f + 1)

    psd = np.zeros(int(length), dtype=np.float64)

    start_bin = max(kmin, 1)
    if start_bin < length:
        sample_freqs = np.arange(start_bin, length, dtype=np.float64) * delta_f
        log_f = np.log(freq_data)
        log_s = np.log(noise_data)
        psd[start_bin:] = np.exp(
            np.interp(np.log(sample_freqs), log_f, log_s, left=log_s[0], right=log_s[-1])
        )

    return FrequencySeries(psd, delta_f=delta_f)


def from_txt(filename, length, delta_f, low_freq_cutoff, is_asd_file=True):
    """Read a two-column ASD/PSD text file and return interpolated PSD."""
    file_data = np.loadtxt(filename)
    if file_data.ndim != 2 or file_data.shape[1] < 2:
        raise ValueError(f"Expected a two-column ASD/PSD file, got shape {file_data.shape}")

    if (file_data < 0).any() or np.logical_not(np.isfinite(file_data)).any():
        raise ValueError(f"Invalid data in {filename}")

    freq_data = file_data[:, 0]
    noise_data = file_data[:, 1]
    if is_asd_file:
        noise_data = noise_data ** 2

    return from_numpy_arrays(freq_data, noise_data, length, delta_f, low_freq_cutoff)
