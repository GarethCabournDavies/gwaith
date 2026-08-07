"""Unit tests for the hwilas.data_processing.tdi module."""

import numpy as np
from gwaith.types import TimeSeries

from hwilas.data_processing.tdi import (
    AET,
    linear_AET_conversion,
    to_timeseries,
)


def test_to_timeseries():
    """Test the conversion of a dictionary of arrays to gwaith TimeSeries."""
    data = {"CH1": np.random.randn(100), "CH2": np.random.randn(100)}
    delta_t = 0.1
    epoch = 12345.0
    ts_dict = to_timeseries(data, delta_t, epoch=epoch)

    assert "CH1" in ts_dict
    assert "CH2" in ts_dict
    assert isinstance(ts_dict["CH1"], TimeSeries)
    assert ts_dict["CH1"].delta_t == delta_t
    assert ts_dict["CH1"].epoch == epoch
    assert np.array_equal(ts_dict["CH1"].data, data["CH1"])
    assert np.array_equal(ts_dict["CH2"].data, data["CH2"])


def test_linear_AET_conversion():
    """Test the conversion from XYZ numpy arrays to AET numpy arrays."""
    X = np.array([1.0])
    Y = np.array([1.0])
    Z = np.array([1.0])
    aet_data = linear_AET_conversion(X, Y, Z)

    assert "LISA_A" in aet_data
    assert "LISA_E" in aet_data
    assert "LISA_T" in aet_data
    assert np.isclose(aet_data["LISA_A"], 0.0)
    assert np.isclose(aet_data["LISA_E"], 0.0)
    assert np.isclose(aet_data["LISA_T"], np.sqrt(3))

def test_AET():
    """Test the XYZ to AET conversion."""
    X = np.array([1.0, 2.0])
    Y = np.array([1.0, 2.0])
    Z = np.array([1.0, 2.0])
    delta_t = 0.5
    epoch = 100

    aet_ts = AET(X, Y, Z, delta_t, epoch)

    assert "LISA_A" in aet_ts
    assert "LISA_E" in aet_ts
    assert "LISA_T" in aet_ts

    assert isinstance(aet_ts["LISA_A"], TimeSeries)
    assert aet_ts["LISA_A"].delta_t == delta_t
    assert aet_ts["LISA_A"].epoch == epoch

    expected_A = (Z - X) / np.sqrt(2.0)
    expected_E = (X - 2.0 * Y + Z) / np.sqrt(6.0)
    expected_T = (X + Y + Z) / np.sqrt(3.0)

    assert np.allclose(aet_ts["LISA_A"].data, expected_A)
    assert np.allclose(aet_ts["LISA_E"].data, expected_E)
    assert np.allclose(aet_ts["LISA_T"].data, expected_T)
