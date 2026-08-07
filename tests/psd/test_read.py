"""Unit tests for the psd.read module."""

import logging
from unittest.mock import patch

import numpy as np
import pytest
from pycbc.types import FrequencySeries

from hwilas.psd.read import from_numpy_arrays, from_txt


@pytest.fixture
def simple_psd_data():
    """Generate simple power-law PSD data for testing."""
    freqs = np.array([10.0, 20.0, 30.0, 100.0])
    # Simple S_h(f) = 1 / f
    psd_vals = 1.0 / freqs
    return freqs, psd_vals


def test_from_numpy_arrays_basic(simple_psd_data):
    """Test basic functionality of from_numpy_arrays."""
    freq_data, noise_data = simple_psd_data
    length = 101
    delta_f = 1.0
    low_freq_cutoff = 10.0

    psd_series = from_numpy_arrays(
        freq_data, noise_data, length, delta_f, low_freq_cutoff
    )

    assert isinstance(psd_series, FrequencySeries)
    assert psd_series.delta_f == delta_f
    assert len(psd_series) == length

    assert psd_series[50] < noise_data[2]  # < 1/30
    assert psd_series[50] > noise_data[3]  # > 1/100


def test_from_numpy_arrays_errors(simple_psd_data):
    """Test ValueErrors in from_numpy_arrays."""
    freq_data, noise_data = simple_psd_data
    with pytest.raises(ValueError, match="must be 1D"):
        from_numpy_arrays(np.atleast_2d(freq_data), noise_data, 100, 1.0, 10)
    with pytest.raises(ValueError, match="same length"):
        from_numpy_arrays(freq_data, noise_data[:-1], 100, 1.0, 10)
    with pytest.raises(ValueError, match="at least two"):
        from_numpy_arrays(freq_data[:1], noise_data[:1], 100, 1.0, 10)
    
    # Test case where data does not go down to low_freq_cutoff
    with pytest.raises(ValueError, match="higher than requested"):
        from_numpy_arrays(freq_data, noise_data, 100, 1.0, 9.0)


def test_from_numpy_arrays_warning(simple_psd_data, caplog):
    """Test the warning for insufficient frequency range."""
    freq_data, noise_data = simple_psd_data
    with caplog.at_level(logging.WARNING):
        psd_series = from_numpy_arrays(freq_data, noise_data, 200, 1.0, 10)
    
    assert "exceeds highest available" in caplog.text
    # The length should be truncated
    expected_len = int(freq_data[-1] / 1.0 + 1)
    assert len(psd_series) == expected_len


@patch("hwilas.psd.read.np.loadtxt")
def test_from_txt_asd(mock_loadtxt, simple_psd_data):
    """Test reading an ASD file."""
    freqs, psd_vals = simple_psd_data
    asd_vals = np.sqrt(psd_vals)
    mock_loadtxt.return_value = np.c_[freqs, asd_vals]

    psd_series = from_txt("dummy.txt", 101, 1.0, 10.0, is_asd_file=True)
    assert isinstance(psd_series, FrequencySeries)
    assert np.isclose(psd_series[10], 1.0/10.0)


@patch("hwilas.psd.read.np.loadtxt")
def test_from_txt_psd(mock_loadtxt, simple_psd_data):
    """Test reading a PSD file."""
    freqs, psd_vals = simple_psd_data
    mock_loadtxt.return_value = np.c_[freqs, psd_vals]

    psd_series = from_txt("dummy.txt", 101, 1.0, 10.0, is_asd_file=False)
    assert isinstance(psd_series, FrequencySeries)
    assert np.isclose(psd_series[10], 1.0/10.0)


@patch("hwilas.psd.read.np.loadtxt")
def test_from_txt_errors(mock_loadtxt):
    """Test ValueErrors in from_txt."""
    with pytest.raises(ValueError, match="Expected a two-column"):
        mock_loadtxt.return_value = np.array([1, 2, 3])
        from_txt("dummy.txt", 100, 1.0, 10)
    with pytest.raises(ValueError, match="Invalid data"):
        mock_loadtxt.return_value = np.array([[10, 1], [20, -1]])
        from_txt("dummy.txt", 100, 1.0, 10)
