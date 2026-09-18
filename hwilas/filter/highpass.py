"""Zero-phase Butterworth IIR highpass filtering."""

from __future__ import annotations

import numpy as np
from scipy.signal import butter, sosfiltfilt


def highpass_filter_scipy(
    data: np.ndarray,
    low_cutoff_freq: float | None,
    sample_rate: float,
    filter_order: int = 5,
) -> np.ndarray:
    """Apply a zero-phase Butterworth IIR highpass filter.

    This removes low-frequency content (below ``low_cutoff_freq``) while
    preserving the passband with negligible phase distortion. It is designed as
    a *direct* highpass second-order-sections (SOS) prototype and applied with
    :func:`scipy.signal.sosfiltfilt`, which filters forward and backward so the
    net phase shift is zero and edge transients are reflected rather than
    truncated.

    Unlike an FIR least-squares design, this does not depend on the data's FFT
    resolution: it correctly attenuates cutoffs that sit within a single
    frequency bin of the dataset (e.g. ``1e-5`` Hz on ~30 days of data), which
    is exactly the regime where an FIR highpass degenerates to all-pass and is
    skipped.

    Args:
        data: Input time-domain signal.
        low_cutoff_freq: Cutoff frequency (Hz). Frequencies below this are
            attenuated. Must be strictly between 0 and the Nyquist frequency.
            If None, the input is returned unchanged.
        sample_rate: Sample rate of the input signal (Hz).
        filter_order: Order of the Butterworth prototype.

    Returns:
        Filtered signal of the same length as the input, or the input itself if
        ``low_cutoff_freq`` is None.

    Raises:
        ValueError: If ``low_cutoff_freq`` is not strictly between 0 and the
            Nyquist frequency, or if the data is too short to filter.
    """
    if low_cutoff_freq is None:
        return data

    nyquist = sample_rate / 2.0
    if not (0 < low_cutoff_freq < nyquist):
        raise ValueError(
            f"low_cutoff_freq ({low_cutoff_freq} Hz) must be between 0 and the "
            f"Nyquist frequency ({nyquist} Hz)"
        )

    n = len(data)
    # sosfiltfilt pads by padlen (default ~3 * max(len(a)) for each section); a
    # very short signal would raise. Guard explicitly for a clear error message.
    if n < 4 * filter_order:
        raise ValueError(
            f"Signal too short ({n} samples) to apply a zero-phase highpass "
            f"filter of order {filter_order}"
        )

    sos = butter(filter_order, low_cutoff_freq / nyquist, btype="high", output="sos")
    return sosfiltfilt(sos, data)
