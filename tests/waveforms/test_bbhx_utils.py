"""Unit tests for the bbhx_utils module."""

from unittest.mock import patch

import numpy as np
import pytest

from hwilas.waveforms.bbhx_utils import (
    ARGS_KEYS,
    build_bbhx_parameter_array,
    get_bbhx_parameter_dict,
    transform_to_bbhx_parameters,
)


@pytest.fixture
def full_param_dict():
    """A dictionary with all required parameters."""
    return {key: float(i) for i, key in enumerate(ARGS_KEYS)}


@patch("hwilas.waveforms.bbhx_utils.convert_frame_parameters")
def test_get_bbhx_parameter_dict(mock_convert, full_param_dict):
    """Test the construction of the BBHx parameter dictionary."""
    mock_convert.return_value = full_param_dict
    result = get_bbhx_parameter_dict(full_param_dict, parameter_frame="LISA")
    mock_convert.assert_called_once_with(
        full_param_dict,
        source_frame="LISA",
        target_frame="SSB",
        t0=0.0,
    )
    assert result == full_param_dict


@patch("hwilas.waveforms.bbhx_utils.convert_frame_parameters")
def test_get_bbhx_parameter_dict_missing_keys(mock_convert):
    """Test that a ValueError is raised if keys are missing."""
    mock_convert.return_value = {"m1": 1.0}  # Missing most keys
    with pytest.raises(ValueError, match="missing"):
        get_bbhx_parameter_dict({}, parameter_frame="LISA")


@patch("hwilas.waveforms.bbhx_utils._core_transforms")
def test_transform_to_bbhx_parameters(mock_core_transforms):
    """Test the transformation to BBHx parameters."""
    # Return a dict with extra keys that should be removed
    mock_core_transforms.return_value = {"m1": 1.0, "extra_key": 2.0}
    x = np.array([[1.0]])
    input_params = ["m1"]
    
    result = transform_to_bbhx_parameters(x, input_params)
    mock_core_transforms.assert_called_once()
    assert "extra_key" not in result
    assert "m1" in result


def test_build_bbhx_parameter_array(full_param_dict):
    """Test the construction of the BBHx parameter array."""
    param_array = build_bbhx_parameter_array(full_param_dict)
    assert isinstance(param_array, np.ndarray)
    assert param_array.dtype == float
    assert len(param_array) == len(ARGS_KEYS)
    # Check that the order is correct
    assert param_array[0] == full_param_dict["m1"]
    assert param_array[-1] == full_param_dict["t_ref"]
