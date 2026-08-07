"""Unit tests for the psd_estimation.smooth module."""

import argparse

import numpy as np
import pytest
from pycbc.types import FrequencySeries

from hwilas.psd_estimation.smooth import add_smoothing_options, smooth_psd


@pytest.fixture
def mock_psd():
    """Create a mock PSD for testing."""
    delta_f = 0.1
    flen = 2000
    freqs = np.arange(flen) * delta_f
    # A simple PSD with a bump to be smoothed
    data = np.ones(flen) * 0.1
    data[100:200] = np.linspace(0.1, 1.0, 100)  # Bump
    psd = FrequencySeries(data, delta_f=delta_f)
    return {"CH1": psd}


def test_smooth_psd(mock_psd):
    """Test the PSD smoothing function."""
    channels = ["CH1"]
    # Region to smooth over the bump
    smoothing_regions = [(11.0, 19.0)]  # Corresponds to indices 110-190

    smoothed_psds = smooth_psd(mock_psd, channels, smoothing_regions)

    assert "CH1" in smoothed_psds
    original_psd = mock_psd["CH1"]
    smoothed_psd = smoothed_psds["CH1"]

    assert isinstance(smoothed_psd, FrequencySeries)
    assert not np.array_equal(original_psd.data, smoothed_psd.data)

    # Check that the region outside the smoothing is unchanged
    assert np.all(original_psd.data[:105] == smoothed_psd.data[:105])
    assert np.all(original_psd.data[195:] == smoothed_psd.data[195:])

    # Check a point in the middle of the smoothed region (index 150)
    # Here, the Hann window is 1, so the value should be the line
    left_val = np.median(original_psd.data[105:110])
    right_val = np.median(original_psd.data[190:195])
    log_slope = (np.log10(right_val) - np.log10(left_val)) / (
        np.log10(19.0) - np.log10(11.0)
    )
    log_intercept = np.log10(left_val) - log_slope * np.log10(11.0)
    
    freq_at_150 = 150 * original_psd.delta_f
    expected_line_val = 10 ** (log_slope * np.log10(freq_at_150) + log_intercept)
    
    # At the center of the Hann window, the smoothed value is a mix
    # window value is 1 at the center of the region, so it should be purely the line
    # The hann window is over the indices, not the frequencies
    mid_index = 150
    # The window is hann(80), mid point is at index 40 of the window
    # which corresponds to index 150 of the array. hann(40) is 1.0
    assert np.isclose(smoothed_psd.data[mid_index], expected_line_val, rtol=1e-3)


def test_add_smoothing_options():
    """Test the argparse helper for smoothing options."""
    parser = argparse.ArgumentParser()
    add_smoothing_options(parser)

    # Test valid input
    args = parser.parse_args(["--smooth-regions", "1e-2,0.1", "0.5,1.0"])
    assert args.smooth_regions == [(0.01, 0.1), (0.5, 1.0)]

    # Test invalid input
    with pytest.raises(SystemExit):  # argparse raises SystemExit on error
        parser.parse_args(["--smooth-regions", "1e-2_0.1"])
