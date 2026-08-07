"""Unit tests for the psd.interpolate module."""

import numpy as np
import pytest
from pycbc.types import FrequencySeries as PycbcFrequencySeries

from hwilas.psd.interpolate import _to_numpy, interpolate


@pytest.fixture
def mock_fs_class():
    """A mock frequency series class for testing."""

    class MockFrequencySeries:
        def __init__(self, data, delta_f, epoch=0.0, f0=0.0):
            self.data = np.asarray(data)
            self.delta_f = delta_f
            self.epoch = epoch
            self.f0 = f0

        @property
        def sample_frequencies(self):
            return np.arange(len(self.data)) * self.delta_f

        def __len__(self):
            return len(self.data)

        def __array__(self, dtype=None):
            return self.data.astype(dtype) if dtype is not None else self.data

    return MockFrequencySeries


def test_to_numpy():
    """Test the _to_numpy helper function."""
    assert isinstance(_to_numpy([1, 2, 3]), np.ndarray)
    assert isinstance(_to_numpy(np.array([1, 2, 3])), np.ndarray)

    class MockWithNumpy:
        def numpy(self):
            return np.array([4, 5, 6])

    assert np.array_equal(_to_numpy(MockWithNumpy()), np.array([4, 5, 6]))


def test_interpolate_basic(mock_fs_class):
    """Test the basic interpolation logic."""
    original_delta_f = 1.0
    original_data = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    series = mock_fs_class(original_data, delta_f=original_delta_f)
    interpolated = interpolate(series, 0.5)
    assert isinstance(interpolated, mock_fs_class)
    assert interpolated.delta_f == 0.5


def test_interpolate_with_pycbc():
    """Test interpolation with a PycbcFrequencySeries."""
    original_delta_f = 1.0
    original_data = np.array([1.0, 2.0, 3.0, 4.0, 5.0], dtype=np.float64)
    series = PycbcFrequencySeries(original_data, delta_f=original_delta_f, epoch=100)
    interpolated = interpolate(series, 0.5)
    assert isinstance(interpolated, PycbcFrequencySeries)
    assert interpolated.delta_f == 0.5
    assert interpolated.start_time == series.start_time
