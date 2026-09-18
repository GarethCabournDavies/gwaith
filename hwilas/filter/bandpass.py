"""Zero-phase Butterworth IIR bandpass filtering and cutoff dispatching."""

from __future__ import annotations

import logging

import numpy as np
from scipy.signal import butter, sosfiltfilt

from .highpass import highpass_filter_scipy
from .lowpass import lowpass_filter_scipy

logger = logging.getLogger(__name__)


def bandpass_filter_scipy(
    data: np.ndarray,
    low_cutoff_freq: float,
    high_cutoff_freq: float,
    sample_rate: float,
    filter_order: int = 5,
) -> np.ndarray:
    """Apply a zero-phase Butterworth IIR bandpass filter.

    This preserves content inside the passband
    ``[low_cutoff_freq, high_cutoff_freq]`` while attenuating frequencies below
    the lower edge and above the upper edge, with negligible phase distortion.
    It is designed as a *direct* bandpass second-order-sections (SOS) prototype
    and applied with :func:`scipy.signal.sosfiltfilt`, which filters forward and
    backward so the net phase shift is zero and edge transients are reflected
    rather than truncated.

    Unlike an FIR least-squares design, this does not depend on the data's FFT
    resolution: it correctly attenuates cutoffs that sit within a single
    frequency bin of the dataset, which is exactly the regime where an FIR
    bandpass degenerates to all-pass and is skipped.

    Args:
        data: Input time-domain signal.
        low_cutoff_freq: Lower passband edge frequency (Hz). Frequencies below
            this are attenuated. Must be strictly greater than 0 and strictly
            less than ``high_cutoff_freq``.
        high_cutoff_freq: Upper passband edge frequency (Hz). Frequencies above
            this are attenuated. Must be strictly between ``low_cutoff_freq``
            and the Nyquist frequency.
        sample_rate: Sample rate of the input signal (Hz).
        filter_order: Order of the Butterworth prototype.

    Returns:
        Filtered signal of the same length as the input.

    Raises:
        ValueError: If the cutoffs are not strictly ordered between 0 and the
            Nyquist frequency, or if the data is too short to filter.
    """
    nyquist = sample_rate / 2.0
    if not (0 < low_cutoff_freq < high_cutoff_freq < nyquist):
        raise ValueError(
            f"Bandpass cutoffs must satisfy "
            f"0 < low ({low_cutoff_freq} Hz) < high ({high_cutoff_freq} Hz) < "
            f"Nyquist ({nyquist} Hz)"
        )

    n = len(data)
    # sosfiltfilt pads by padlen (default ~3 * max(len(a)) for each section); a
    # very short signal would raise. Guard explicitly for a clear error message.
    if n < 4 * filter_order:
        raise ValueError(
            f"Signal too short ({n} samples) to apply a zero-phase bandpass "
            f"filter of order {filter_order}"
        )

    sos = butter(
        filter_order,
        [low_cutoff_freq / nyquist, high_cutoff_freq / nyquist],
        btype="bandpass",
        output="sos",
    )
    return sosfiltfilt(sos, data)


def apply_bandpass_filter(
    data: np.ndarray,
    low_cutoff_freq: float | None = None,
    high_cutoff_freq: float | None = None,
    sample_rate: float | None = None,
    filter_order: int = 5,
) -> np.ndarray:
    """Apply the appropriate zero-phase Butterworth IIR filter for the inputs.

    This is a convenience dispatcher that inspects which cutoff frequencies are
    provided and routes to the matching single-purpose filter:

    * neither cutoff            → returns ``data`` unchanged
    * ``low_cutoff_freq`` only  → :func:`highpass_filter_scipy`
    * ``high_cutoff_freq`` only → :func:`lowpass_filter_scipy`
    * both                      → :func:`bandpass_filter_scipy`

    Args:
        data: Input time-domain signal.
        low_cutoff_freq: Lower passband edge frequency (Hz). Frequencies below
            this are attenuated. If None, no low-frequency attenuation is
            applied.
        high_cutoff_freq: Upper passband edge frequency (Hz). Frequencies above
            this are attenuated. If None, no high-frequency attenuation is
            applied.
        sample_rate: Sample rate of the input signal (Hz). Required when at
            least one cutoff frequency is provided.
        filter_order: Order of the Butterworth prototype.

    Returns:
        Filtered signal of the same length as the input, or the input itself if
        no cutoff frequencies are given.

    Raises:
        ValueError: If a cutoff is out of range for its filter, if both cutoffs
            are provided but not strictly ordered, or if ``sample_rate`` is
            missing when a filter is required.
    """
    if low_cutoff_freq is None and high_cutoff_freq is None:
        return data

    if sample_rate is None:
        raise ValueError(
            "sample_rate must be provided when at least one cutoff frequency is "
            "given"
        )

    if low_cutoff_freq is not None and high_cutoff_freq is not None:
        return bandpass_filter_scipy(
            data,
            low_cutoff_freq=low_cutoff_freq,
            high_cutoff_freq=high_cutoff_freq,
            sample_rate=sample_rate,
            filter_order=filter_order,
        )

    if low_cutoff_freq is not None:
        return highpass_filter_scipy(
            data,
            low_cutoff_freq=low_cutoff_freq,
            sample_rate=sample_rate,
            filter_order=filter_order,
        )

    return lowpass_filter_scipy(
        data,
        high_cutoff_freq=high_cutoff_freq,
        sample_rate=sample_rate,
        filter_order=filter_order,
    )