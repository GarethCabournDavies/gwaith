"""Unit tests for the sangria module."""

from unittest.mock import patch

import numpy as np
import pytest
from gwaith.types import TimeSeries

from hwilas.data_processing.sangria import (
    load_downsample_sangria,
    load_sangria_timeseries,
)


@patch("hwilas.data_processing.sangria.hdfio.load_array")
def test_load_downsample_sangria_no_downsample(mock_load_array):
    """Test loading sangria data without downsampling."""
    mock_data = {
        "X": np.random.randn(1000),
        "Y": np.random.randn(1000),
        "Z": np.random.randn(1000),
    }
    mock_descr = {"dt": 0.1, "t0": 100.0}
    mock_load_array.return_value = (mock_data, mock_descr)

    result = load_downsample_sangria("dummy.h5", downsample_delta_t=0.1)
    assert "LISA_A" in result
    assert isinstance(result["LISA_A"], TimeSeries)
    assert result["LISA_A"].delta_t == 0.1


@patch("hwilas.data_processing.sangria.hdfio.load_array")
def test_load_downsample_sangria_with_downsample(mock_load_array):
    """Test loading and downsampling sangria data."""
    original_dt = 0.1
    downsample_dt = 0.2
    num_samples = 1000
    mock_data = {
        "X": np.arange(num_samples),
        "Y": np.arange(num_samples),
        "Z": np.arange(num_samples),
    }
    mock_descr = {"dt": original_dt, "t0": 100.0}
    mock_load_array.return_value = (mock_data, mock_descr)

    result = load_downsample_sangria("dummy.h5", downsample_delta_t=downsample_dt)
    expected_samples = num_samples // int(downsample_dt / original_dt)

    assert result["LISA_A"].delta_t == downsample_dt
    assert len(result["LISA_A"]) == expected_samples


@patch("hwilas.data_processing.sangria.hdfio.load_array")
def test_load_downsample_sangria_value_error(mock_load_array):
    """Test ValueError for invalid downsampling delta_t."""
    mock_descr = {"dt": 0.1, "t0": 100.0}
    mock_load_array.return_value = ({"X": np.array([1])}, mock_descr)
    with pytest.raises(ValueError, match="must be an integer multiple"):
        load_downsample_sangria("dummy.h5", downsample_delta_t=0.15)


@patch("hwilas.data_processing.sangria.hdfio.load_attributes")
@patch("hwilas.data_processing.sangria.hdfio.load_array")
def test_load_sangria_timeseries(mock_load_array, mock_load_attrs):
    """Test loading a sangria file into TimeSeries."""
    mock_data = {
        "X": np.random.randn(100),
        "Y": np.random.randn(100),
        "Z": np.random.randn(100),
    }
    mock_attrs = {"dt": 0.1, "t0": 100.0}
    mock_load_array.return_value = (mock_data, {})
    mock_load_attrs.return_value = mock_attrs

    result = load_sangria_timeseries("dummy.h5")

    assert "LISA_A" in result
    assert isinstance(result["LISA_A"], TimeSeries)
    assert result["LISA_A"].delta_t == 0.1
    assert result["LISA_A"].epoch == 100.0
