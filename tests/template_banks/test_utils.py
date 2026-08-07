"""Unit tests for the template_banks.utils module."""

import numpy as np
import pytest

from hwilas.template_banks.utils import parse_mode_array, waveform_from_bank


@pytest.mark.parametrize(
    "input_modes, expected_modes",
    [
        ([(2, 2), (3, 3)], [(2, 2), (3, 3)]),  # Already correct
        ("22 33 44", [(2, 2), (3, 3), (4, 4)]),  # Space-separated string
        (b"22 3-3", [(2, 2), (3, -3)]),  # Bytes string
        ([22, 33, 44], [(2, 2), (3, 3), (4, 4)]),  # List of ints
        ([22.0, 33.0], [(2, 2), (3, 3)]),  # List of floats
        ("2-2", [(2, -2)]),  # Single signed mode
        (22, [(2, 2)]),  # Single int
        (None, None),  # None value
    ],
)
def test_parse_mode_array(input_modes, expected_modes):
    """Test the parsing of the mode_array parameter."""
    params = {"mode_array": input_modes}
    result = parse_mode_array(params)
    assert result["mode_array"] == expected_modes


def test_parse_mode_array_no_key():
    """Test parse_mode_array when the key is not present."""
    params = {}
    result = parse_mode_array(params)
    assert "mode_array" not in result


def test_parse_mode_array_error():
    """Test that parse_mode_array raises an error for invalid format."""
    params = {"mode_array": "2"}  # Invalid string format
    with pytest.raises(ValueError, match="Unknown lm mode string format"):
        parse_mode_array(params)


def test_waveform_from_bank():
    """Test the construction of a waveform dictionary from a bank file."""
    # Mock bank file (dictionary of arrays)
    bank_file = {
        "mass1": np.array([10.0, 20.0]),
        "mass2": np.array([5.0, 15.0]),
        "inclination": np.array([0.1, 0.2]),
        "polarization": np.array([0.3, 0.4]),
        "spin1z": np.array([0.5, 0.6]),
        "spin2z": np.array([-0.5, -0.6]),
        "eclipticlatitude": np.array([0.7, 0.8]),
        "eclipticlongitude": np.array([0.9, 1.0]),
        "distance": np.array([100.0, 200.0]),
        "mode_array": np.array(["22", "22 33"]),
        "f_lower": np.array([20.0, 25.0]),
    }

    waveform_params_shared = {"f_ref": 10.0}
    idx = 1
    data_length = 4096

    wf_params = waveform_from_bank(
        bank_file, idx, waveform_params_shared, data_length
    )

    assert wf_params["tc"] == data_length
    assert wf_params["mass1"] == 20.0
    assert wf_params["mass2"] == 15.0
    assert wf_params["inclination"] == 0.2
    assert wf_params["mode_array"] == [(2, 2), (3, 3)]
    # f_ref should be the max of shared and bank f_lower
    assert wf_params["f_ref"] == 25.0
