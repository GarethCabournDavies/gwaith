"""Unit tests for the bbhx_phenom module."""

from unittest.mock import patch, MagicMock
import numpy as np
import pytest

# Mock bbhx and pycbc before importing the module under test
import sys

# Mock bbhx
mock_bbhx = MagicMock()
mock_bbhx.utils.constants.YRSID_SI = 31536000.0
mock_bbhx.utils.constants.L_SI = 2.5e9
sys.modules["bbhx"] = mock_bbhx
sys.modules["bbhx.utils"] = mock_bbhx.utils
sys.modules["bbhx.utils.constants"] = mock_bbhx.utils.constants
sys.modules["bbhx.waveformbuild"] = mock_bbhx.waveformbuild
# We avoid mocking pycbc.types directly to prevent breaking pycbc's internal
# 'scheme' mechanism, which was causing a RuntimeError.


from hwilas.waveforms.bbhx_phenom import get_fd_waveform_from_bbhx


@pytest.fixture
def waveform_params():
    """A default set of waveform parameters."""
    return {
        "mass1": 1e6,
        "mass2": 1e5,
        "distance": 1000.0,
        "f_lower": 1e-4,
        "delta_f": 1e-5,
        "length": 10000,
        "approximant": "BBHX_PhenomD",
        "t_obs_start": 0.0,
        "tc": 0.0,
        "eclipticlongitude": 1.0,
        "eclipticlatitude": 0.5,
        "polarization": 0.2,
        "ifos": ["LISA_A", "LISA_E", "LISA_T"],
        "mode_array": [(2,2)]
    }


@patch("hwilas.waveforms.bbhx_phenom.get_bbhx_parameter_dict")
@patch("hwilas.waveforms.bbhx_phenom.FrequencySeries", new_callable=MagicMock)
def test_get_fd_waveform_from_bbhx_call(mock_fs_class, mock_get_params, waveform_params):
    """Test the basic call structure of get_fd_waveform_from_bbhx."""
    # Mock the waveform generator call to return a dummy waveform
    dummy_wave = np.ones((3, waveform_params["length"]))
    mock_bbhx.waveformbuild.BBHWaveformFD.return_value.return_value = (dummy_wave,)
    # Mock the parameter conversion to return a predictable dict
    mock_get_params.return_value = {
        "m1": 1.0, "m2": 1.0, "chi1z": 0.0, "chi2z": 0.0,
        "distance": 1.0, "phi_ref": 0.0, "f_ref": 0.0, "inc": 0.0,
        "lam": 0.0, "beta": 0.0, "psi": 0.0, "t_ref": 0.0
    }

    get_fd_waveform_from_bbhx(**waveform_params)

    # Check that the parameter conversion utility was called
    mock_get_params.assert_called_once()
    # Check that the BBHx waveform generator was instantiated and called
    mock_bbhx.waveformbuild.BBHWaveformFD.assert_called_once()
    mock_bbhx.waveformbuild.BBHWaveformFD.return_value.assert_called_once()

    # Check that the FrequencySeries class was called to construct the output
    assert mock_fs_class.called


@patch("hwilas.waveforms.bbhx_phenom.get_bbhx_parameter_dict")
@patch("hwilas.waveforms.bbhx_phenom.FrequencySeries", new_callable=MagicMock)
def test_get_fd_waveform_from_bbhx_output(
    mock_fs_class, mock_get_params, waveform_params
):
    """Test the output of the waveform function."""
    dummy_wave = np.ones((3, waveform_params["length"]))
    mock_bbhx.waveformbuild.BBHWaveformFD.return_value.return_value = (dummy_wave,)

    output = get_fd_waveform_from_bbhx(**waveform_params)

    assert "LISA_A" in output
    assert "LISA_E" in output
    assert "LISA_T" in output
    # Check that the output values are the objects created by the mocked class
    assert output["LISA_A"] is mock_fs_class.return_value


@patch("hwilas.waveforms.bbhx_phenom.get_bbhx_parameter_dict")
@patch("hwilas.waveforms.bbhx_phenom.FrequencySeries")
def test_get_fd_waveform_from_bbhx_tdi2_rescaling(
    mock_fs_class, mock_get_params, waveform_params
):
    """Test the TDI 2.0 rescaling."""
    dummy_wave = np.ones((3, waveform_params["length"]))
    mock_bbhx.waveformbuild.BBHWaveformFD.return_value.return_value = (dummy_wave,)

    # Mock the FrequencySeries so we can check the multiplication
    mock_fs_instance = MagicMock()
    mock_fs_instance.sample_frequencies.data = np.arange(waveform_params["length"]) * waveform_params["delta_f"]
    mock_fs_class.return_value = mock_fs_instance

    get_fd_waveform_from_bbhx(tdi='2.0', **waveform_params)

    # Check that the rescale factor was applied
    assert mock_fs_instance.__imul__.called


def test_get_fd_waveform_from_bbhx_invalid_length(waveform_params):
    """Test that an invalid length raises a ValueError."""
    waveform_params["length"] = 0
    with pytest.raises(ValueError, match="Waveform length must be positive"):
        get_fd_waveform_from_bbhx(**waveform_params)
