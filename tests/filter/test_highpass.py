"""Unit tests for the zero-phase Butterworth IIR highpass filter."""

import numpy as np
import pytest

from hwilas.filter import highpass_filter_scipy


def test_highpass_filter_scipy_attenuates_low_frequencies():
    """Test the Butterworth IIR highpass attenuates sub-cutoff frequencies.

    The key property of the IIR design is that it works even when the cutoff
    sits within a single frequency bin of the data's FFT resolution -- the
    regime where an FIR highpass degenerates to an all-pass filter and is
    skipped. Here the 1e-5 Hz cutoff is far below one FFT bin (duration ~30
    days -> df ~ 4e-7 Hz), yet the IIR highpass still removes sub-cutoff
    content while leaving the in-band tone intact.

    The stop tone is placed *below* the cutoff (2e-6 Hz): a tone at the cutoff
    itself sits in the transition band and is only ~3 dB down by design, so it
    would not be strongly attenuated regardless of filter quality.
    """
    sample_rate = 0.25  # Nyquist = 0.125 Hz, like 4-second-cadence data
    duration = 30 * 86400  # ~30 days
    num_samples = int(sample_rate * duration)
    times = np.arange(num_samples) / sample_rate

    freq_stop = 2e-6  # below the cutoff -> must be strongly attenuated
    freq_pass = 1e-2  # well inside passband -> preserved
    signal = (
        np.sin(2 * np.pi * freq_stop * times)
        + np.sin(2 * np.pi * freq_pass * times)
    )

    low_cutoff_freq = 1e-5
    filtered_signal = highpass_filter_scipy(
        signal, low_cutoff_freq=low_cutoff_freq, sample_rate=sample_rate
    )

    assert len(filtered_signal) == len(signal)

    fft_original = np.fft.rfft(signal)
    fft_filtered = np.fft.rfft(filtered_signal)
    freqs = np.fft.rfftfreq(num_samples, 1 / sample_rate)
    df = 1.0 / duration

    idx_pass = np.argmin(np.abs(freqs - freq_pass))

    # The in-band tone is preserved (zero-phase, ~unity gain in passband).
    assert np.isclose(
        np.abs(fft_original[idx_pass]), np.abs(fft_filtered[idx_pass]), rtol=0.2
    )

    # Sub-cutoff energy is strongly attenuated. Use an energy ratio over a clean
    # band below the cutoff (starting a couple of bins above DC to avoid the
    # spectral-leakage tail from the non-bin-aligned stop tone) rather than a
    # single bin, which is unreliable for sub-bin frequencies.
    band = (freqs >= 2 * df) & (freqs < low_cutoff_freq)
    energy_ratio = (
        np.sum(np.abs(fft_filtered[band]) ** 2)
        / np.sum(np.abs(fft_original[band]) ** 2)
    )
    assert energy_ratio < 0.1


def test_highpass_filter_scipy_removes_dc_and_low_energy():
    """Test that a DC offset plus low-frequency content is removed.

    A 1e-5 Hz cutoff must remove DC and suppress near-DC energy.
    The total spectral energy below the cutoff should drop by orders of
    magnitude while in-band energy is preserved.
    """
    sample_rate = 0.25
    duration = 30 * 86400
    num_samples = int(sample_rate * duration)
    times = np.arange(num_samples) / sample_rate

    # DC offset + a low tone below cutoff + an in-band tone.
    signal = (
        5.0
        + np.sin(2 * np.pi * 1e-5 * times)
        + 0.1 * np.sin(2 * np.pi * 1e-3 * times)
    )

    low_cutoff_freq = 1e-5
    filtered_signal = highpass_filter_scipy(
        signal, low_cutoff_freq=low_cutoff_freq, sample_rate=sample_rate
    )

    freqs = np.fft.rfftfreq(num_samples, 1 / sample_rate)
    below = freqs < low_cutoff_freq
    # The in-band check starts well above the cutoff (2e-3 Hz), past the
    # Butterworth transition region where gain is still ramping up. Measuring
    # from the cutoff itself would include that rolloff and understate the
    # passband gain even though the filter preserves in-band content.
    above = (freqs >= 2e-3) & (freqs < 1e-2)

    Fo = np.abs(np.fft.rfft(signal))
    Ff = np.abs(np.fft.rfft(filtered_signal))

    # Energy below the cutoff is almost entirely removed (including DC, which
    # sosfiltfilt's odd padding drives to zero).
    ratio_below = np.sum(Ff[below] ** 2) / np.sum(Fo[below] ** 2)
    assert ratio_below < 1e-3

    # In-band energy is preserved (same total power, within tolerance).
    ratio_above = np.sum(Ff[above] ** 2) / np.sum(Fo[above] ** 2)
    assert 0.5 < ratio_above < 1.5


def test_highpass_filter_scipy_none_cutoff_returns_input():
    """Test that a None cutoff returns the input unchanged."""
    data = np.random.randn(1000)
    result = highpass_filter_scipy(data, low_cutoff_freq=None, sample_rate=1000)
    assert result is data


def test_highpass_filter_scipy_invalid_cutoff_raises():
    """Test that an invalid cutoff raises ValueError."""
    data = np.random.randn(1000)
    with pytest.raises(ValueError, match="must be between 0 and the Nyquist"):
        highpass_filter_scipy(data, low_cutoff_freq=1.0, sample_rate=1.0)
