"""Unit tests for the mojito module."""

from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from gwaith.types import TimeSeries

from hwilas.data_processing.mojito import (
    load_downsample_mojito,
    lowpass_filter_scipy,
)


def test_lowpass_filter_scipy():
    """Test the scipy-based lowpass filter."""
    sample_rate = 1000
    duration = 2
    num_samples = sample_rate * duration
    times = np.linspace(0, duration, num_samples, endpoint=False)

    # Signal with two frequencies: one to keep, one to filter out
    freq_pass = 50
    freq_stop = 200
    signal = np.sin(2 * np.pi * freq_pass * times) + np.sin(
        2 * np.pi * freq_stop * times
    )

    cutoff_freq = 100
    filter_width = 20
    filtered_signal = lowpass_filter_scipy(
        signal, cutoff_freq, filter_width, sample_rate
    )

    # Check that the high frequency component is attenuated
    fft_original = np.fft.rfft(signal)
    fft_filtered = np.fft.rfft(filtered_signal)
    freqs = np.fft.rfftfreq(num_samples, 1 / sample_rate)

    # Find the magnitude at the two frequencies
    mag_original_pass = np.abs(fft_original[np.isclose(freqs, freq_pass, atol=1)])[0]
    mag_filtered_pass = np.abs(fft_filtered[np.isclose(freqs, freq_pass, atol=1)])[0]
    mag_original_stop = np.abs(fft_original[np.isclose(freqs, freq_stop, atol=1)])[0]
    mag_filtered_stop = np.abs(fft_filtered[np.isclose(freqs, freq_stop, atol=1)])[0]

    # The pass frequency should be largely unaffected
    assert np.isclose(mag_original_pass, mag_filtered_pass, rtol=0.1)
    # The stop frequency should be significantly attenuated
    assert mag_filtered_stop < mag_original_stop * 0.01


@patch("hwilas.data_processing.mojito.MojitoL1File")
def test_load_downsample_mojito(mock_mojito_file):
    """Test loading and downsampling a mock Mojito file."""
    # 1. Setup mock Mojito file
    original_sample_rate = 16
    duration = 10
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
    result = load_downsample_mojito("dummy_path.h5", downsample_delta_t, channels)

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
    with pytest.raises(ValueError, match="Cannot load Mojito file without downsampling"):
        load_downsample_mojito("dummy.h5", downsample_delta_t=None)

    with patch("hwilas.data_processing.mojito.MojitoL1File") as mock_mojito_file:
        mock_file_instance = MagicMock()
        mock_mojito_file.return_value.__enter__.return_value = mock_file_instance
        mock_file_instance.tdis.time_sampling.dt = 0.1
        with pytest.raises(ValueError, match="must be an integer multiple"):
            load_downsample_mojito("dummy.h5", downsample_delta_t=0.15)
