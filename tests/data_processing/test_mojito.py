"""Unit tests for the mojito module."""

from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from gwaith.types import TimeSeries

from hwilas.data_processing.mojito import (
    load_downsample_mojito,
    bandpass_filter_scipy,
    _apply_edge_taper,
)


def test_bandpass_filter_scipy():
    """Test the scipy-based bandpass filter."""
    sample_rate = 1000
    duration = 20  # Long enough to resolve low frequencies in the FFT
    num_samples = sample_rate * duration
    times = np.linspace(0, duration, num_samples, endpoint=False)

    # Signal with three frequencies: one below passband, one in passband, one above
    freq_low_stop = 5
    freq_pass = 50
    freq_high_stop = 200
    signal = (
        np.sin(2 * np.pi * freq_low_stop * times)
        + np.sin(2 * np.pi * freq_pass * times)
        + np.sin(2 * np.pi * freq_high_stop * times)
    )

    low_cutoff_freq = 20
    high_cutoff_freq = 100
    filter_width = 20
    filtered_signal = bandpass_filter_scipy(
        signal, low_cutoff_freq, high_cutoff_freq, filter_width, sample_rate
    )

    fft_original = np.fft.rfft(signal)
    fft_filtered = np.fft.rfft(filtered_signal)
    freqs = np.fft.rfftfreq(num_samples, 1 / sample_rate)

    # Find the magnitude at the three frequencies (use argmin for exact bin match)
    idx_low = np.argmin(np.abs(freqs - freq_low_stop))
    idx_pass = np.argmin(np.abs(freqs - freq_pass))
    idx_high = np.argmin(np.abs(freqs - freq_high_stop))

    mag_original_low = np.abs(fft_original[idx_low])
    mag_filtered_low = np.abs(fft_filtered[idx_low])
    mag_original_pass = np.abs(fft_original[idx_pass])
    mag_filtered_pass = np.abs(fft_filtered[idx_pass])
    mag_original_high = np.abs(fft_original[idx_high])
    mag_filtered_high = np.abs(fft_filtered[idx_high])

    # The pass frequency should be largely unaffected
    assert np.isclose(mag_original_pass, mag_filtered_pass, rtol=0.1)
    # The low and high stop frequencies should be significantly attenuated
    assert mag_filtered_low < mag_original_low * 0.01
    assert mag_filtered_high < mag_original_high * 0.01


def test_bandpass_filter_scipy_highpass_only():
    """Test that providing only low_cutoff_freq applies a highpass filter."""
    sample_rate = 1000
    duration = 20
    num_samples = sample_rate * duration
    times = np.linspace(0, duration, num_samples, endpoint=False)

    freq_low_stop = 5
    freq_pass = 50
    signal = (
        np.sin(2 * np.pi * freq_low_stop * times)
        + np.sin(2 * np.pi * freq_pass * times)
    )

    low_cutoff_freq = 20
    filtered_signal = bandpass_filter_scipy(
        signal, low_cutoff_freq=low_cutoff_freq, sample_rate=sample_rate
    )

    fft_original = np.fft.rfft(signal)
    fft_filtered = np.fft.rfft(filtered_signal)
    freqs = np.fft.rfftfreq(num_samples, 1 / sample_rate)

    idx_low = np.argmin(np.abs(freqs - freq_low_stop))
    idx_pass = np.argmin(np.abs(freqs - freq_pass))

    # Low frequency should be attenuated, passband unaffected
    assert np.abs(fft_filtered[idx_low]) < np.abs(fft_original[idx_low]) * 0.01
    assert np.isclose(
        np.abs(fft_original[idx_pass]), np.abs(fft_filtered[idx_pass]), rtol=0.1
    )


def test_bandpass_filter_scipy_lowpass_only():
    """Test that providing only high_cutoff_freq applies a lowpass filter."""
    sample_rate = 1000
    duration = 20
    num_samples = sample_rate * duration
    times = np.linspace(0, duration, num_samples, endpoint=False)

    freq_pass = 50
    freq_high_stop = 200
    signal = (
        np.sin(2 * np.pi * freq_pass * times)
        + np.sin(2 * np.pi * freq_high_stop * times)
    )

    high_cutoff_freq = 100
    filtered_signal = bandpass_filter_scipy(
        signal, high_cutoff_freq=high_cutoff_freq, sample_rate=sample_rate
    )

    fft_original = np.fft.rfft(signal)
    fft_filtered = np.fft.rfft(filtered_signal)
    freqs = np.fft.rfftfreq(num_samples, 1 / sample_rate)

    idx_pass = np.argmin(np.abs(freqs - freq_pass))
    idx_high = np.argmin(np.abs(freqs - freq_high_stop))

    # High frequency should be attenuated, passband unaffected
    assert np.abs(fft_filtered[idx_high]) < np.abs(fft_original[idx_high]) * 0.01
    assert np.isclose(
        np.abs(fft_original[idx_pass]), np.abs(fft_filtered[idx_pass]), rtol=0.1
    )


def test_bandpass_filter_scipy_no_cutoffs_returns_input():
    """Test that providing no cutoff frequencies returns the input unchanged."""
    data = np.random.randn(1000)
    result = bandpass_filter_scipy(data, sample_rate=1000)
    assert result is data


def test_bandpass_filter_scipy_very_low_cutoff():
    """Test highpass with a very low cutoff (e.g. 0.0001 Hz on downsampled data).

    This exercises the edge case where the lower band edge is extremely close
    to 0 and floating-point rounding could push it negative.
    """
    sample_rate = 1.0  # Nyquist = 0.5 Hz, like 2-second-cadence downsampled data
    duration = 2000  # Need > 3*filter_order samples for filtfilt
    num_samples = int(sample_rate * duration)
    times = np.linspace(0, duration, num_samples, endpoint=False)

    # A signal at 0.3 Hz (in passband) and 0.005 Hz (below cutoff)
    freq_pass = 0.3
    freq_stop = 0.005
    signal = (
        np.sin(2 * np.pi * freq_pass * times)
        + np.sin(2 * np.pi * freq_stop * times)
    )

    low_cutoff_freq = 0.0001
    filtered_signal = bandpass_filter_scipy(
        signal, low_cutoff_freq=low_cutoff_freq, sample_rate=sample_rate
    )

    # Should not raise and should return same length
    assert len(filtered_signal) == len(signal)


def test_bandpass_filter_scipy_highpass_skipped_when_too_close_to_dc():
    """Test that a highpass cutoff too close to DC is skipped.

    With the default bin-based transition width, a low-frequency cutoff whose
    stop band [0, fc - width/2] spans fewer than one frequency bin cannot form
    meaningful attenuation and should be skipped (data returned unchanged)
    rather than producing a degenerate all-pass filter.
    """
    sample_rate = 0.25  # Nyquist = 0.125 Hz, like 4-second-cadence data
    duration = 2000
    num_samples = int(sample_rate * duration)
    times = np.linspace(0, duration, num_samples, endpoint=False)

    signal = np.sin(2 * np.pi * 0.01 * times) + np.random.randn(num_samples) * 1e-3

    # Cutoff far below the transition-band half-width: must be skipped.
    low_cutoff_freq = 1e-7
    result = bandpass_filter_scipy(
        signal, low_cutoff_freq=low_cutoff_freq, sample_rate=sample_rate
    )

    # Skipped entirely -> identical to the input (no filtering applied).
    assert np.array_equal(result, signal)


def test_bandpass_filter_scipy_highpass_attenuates_low_frequencies():
    """Test that a buildable highpass cutoff actually attenuates low frequencies.

    With a bin-based transition width and a realistic long dataset
    (4-month cadence), sub-cutoff power must be strongly reduced while
    in-band power is preserved. A 1e-5 Hz
    cutoff would NOT be buildable here (its stop band spans < 1 bin) and is
    covered by the skip test instead; we use a buildable 1e-3 Hz cutoff.
    """
    sample_rate = 0.25  # Nyquist = 0.125 Hz
    duration = 4 * 30 * 86400  # ~4 months, like a full LISA dataset
    num_samples = int(sample_rate * duration)
    times = np.linspace(0, duration, num_samples, endpoint=False)

    freq_stop = 5e-4  # below cutoff -> should be attenuated
    freq_pass = 1e-2  # well inside passband -> preserved
    signal = (
        np.sin(2 * np.pi * freq_stop * times)
        + np.sin(2 * np.pi * freq_pass * times)
    )

    low_cutoff_freq = 1e-3
    filtered_signal = bandpass_filter_scipy(
        signal, low_cutoff_freq=low_cutoff_freq, sample_rate=sample_rate
    )

    fft_original = np.fft.rfft(signal)
    fft_filtered = np.fft.rfft(filtered_signal)
    freqs = np.fft.rfftfreq(num_samples, 1 / sample_rate)

    idx_stop = np.argmin(np.abs(freqs - freq_stop))
    idx_pass = np.argmin(np.abs(freqs - freq_pass))

    # Low frequency attenuated, passband preserved.
    assert np.abs(fft_filtered[idx_stop]) < np.abs(fft_original[idx_stop]) * 0.1
    assert np.isclose(
        np.abs(fft_original[idx_pass]), np.abs(fft_filtered[idx_pass]), rtol=0.2
    )


@patch("hwilas.data_processing.mojito.MojitoL1File")
def test_load_downsample_mojito(mock_mojito_file):
    """Test loading and downsampling a mock Mojito file."""
    # 1. Setup mock Mojito file
    original_sample_rate = 16
    duration = 2000  # Long enough for filtfilt padding (needs > 3*filter_order samples)
    num_samples = original_sample_rate * duration
    mock_epoch = 1234567890.0

    # Create a mock for the file object itself
    mock_file_instance = MagicMock()
    mock_mojito_file.return_value.__enter__.return_value = mock_file_instance

    # Mock tdis attributes
    mock_file_instance.tdis.time_sampling.dt = 1.0 / original_sample_rate
    mock_file_instance.tdis.time_sampling.epoch = mock_epoch
    mock_file_instance.tdis.time_sampling.size = num_samples

    # Create synthetic data for channels
    times = np.arange(num_samples) / original_sample_rate
    # Use a low frequency sine wave that should survive downsampling
    low_freq_signal = np.sin(2 * np.pi * 1 * times)
    mock_file_instance.tdis.a2 = low_freq_signal

    # 2. Call the function
    downsample_rate = 4
    downsample_delta_t = 1.0 / downsample_rate
    channels = ["LISA_A"]
    # Nyquist of downsampled cadence is 1/4/2 = 0.125 Hz, so the 1 Hz test
    # signal is above it. Use a low cutoff that keeps the signal in-band.
    result = load_downsample_mojito(
        "dummy_path.h5", downsample_delta_t, channels, low_cutoff_freq=0.01
    )

    # 3. Assertions
    assert "LISA_A" in result
    output_ts = result["LISA_A"]

    assert isinstance(output_ts, TimeSeries)
    assert output_ts.epoch == mock_epoch
    assert np.isclose(output_ts.delta_t, downsample_delta_t)

    expected_num_samples = num_samples // (
        original_sample_rate // downsample_rate
    )
    assert len(output_ts) == expected_num_samples

    # Check that the signal is still present (not a full correctness check)
    assert np.std(output_ts.data) > 0.1


def test_load_downsample_mojito_value_errors():
    """Test that load_downsample_mojito raises ValueErrors for invalid inputs."""
    with patch("hwilas.data_processing.mojito.MojitoL1File") as mock_mojito_file:
        mock_file_instance = MagicMock()
        mock_mojito_file.return_value.__enter__.return_value = mock_file_instance
        mock_file_instance.tdis.time_sampling.dt = 0.1
        with pytest.raises(ValueError, match="Cannot load Mojito file without downsampling"):
            load_downsample_mojito("dummy.h5", downsample_delta_t=None)

        with pytest.raises(ValueError, match="must be an integer multiple"):
            load_downsample_mojito("dummy.h5", downsample_delta_t=0.15)


def test_apply_edge_taper_one_day_shape():
    """The edge taper should zero endpoints and keep the untapered centre."""
    data = np.ones(20, dtype=np.float64)
    _apply_edge_taper(data, taper_samples=4)

    assert np.isclose(data[0], 0.0)
    assert np.isclose(data[-1], 0.0)
    assert np.allclose(data[4:-4], 1.0)
