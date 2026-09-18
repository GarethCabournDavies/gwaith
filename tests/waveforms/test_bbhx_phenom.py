"""Unit tests for the bbhx_phenom module."""

from unittest.mock import patch, MagicMock
import numpy as np
import pytest


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


@pytest.fixture
def mock_bbhwaveformfd():
    """Patch BBHWaveformFD in the bbhx_phenom module namespace."""
    with patch("hwilas.waveforms.bbhx_phenom.BBHWaveformFD") as mock_cls:
        yield mock_cls


def _set_dummy_waveform(mock_bbhwaveformfd, length):
    """Configure the BBHWaveformFD mock to return a dummy waveform."""
    dummy_wave = np.ones((3, length))
    mock_bbhwaveformfd.return_value.return_value = dummy_wave


@patch("hwilas.waveforms.bbhx_phenom.get_bbhx_parameter_dict")
@patch("hwilas.waveforms.bbhx_phenom.FrequencySeries", new_callable=MagicMock)
def test_get_fd_waveform_from_bbhx_call(
    mock_fs_class, mock_get_params, waveform_params, mock_bbhwaveformfd
):
    """Test the basic call structure of get_fd_waveform_from_bbhx."""
    _set_dummy_waveform(mock_bbhwaveformfd, waveform_params["length"])
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
    mock_bbhwaveformfd.assert_called_once()
    mock_bbhwaveformfd.return_value.assert_called_once()

    # Check that the FrequencySeries class was called to construct the output
    assert mock_fs_class.called


@patch("hwilas.waveforms.bbhx_phenom.get_bbhx_parameter_dict")
@patch("hwilas.waveforms.bbhx_phenom.FrequencySeries", new_callable=MagicMock)
def test_get_fd_waveform_from_bbhx_output(
    mock_fs_class, mock_get_params, waveform_params, mock_bbhwaveformfd
):
    """Test the output of the waveform function."""
    _set_dummy_waveform(mock_bbhwaveformfd, waveform_params["length"])

    output = get_fd_waveform_from_bbhx(**waveform_params)

    assert "LISA_A" in output
    assert "LISA_E" in output
    assert "LISA_T" in output
    # With align_tc_zero=True, cyclic_time_shift should be called on each FS
    mock_fs_class.return_value.cyclic_time_shift.assert_called()
    # The output for each IFO should be the result of (shifted_fs * rescale)
    shifted = mock_fs_class.return_value.cyclic_time_shift.return_value
    assert output["LISA_A"] == shifted.__mul__.return_value


@patch("hwilas.waveforms.bbhx_phenom.get_bbhx_parameter_dict")
@patch("hwilas.waveforms.bbhx_phenom.FrequencySeries")
def test_get_fd_waveform_from_bbhx_tdi2_rescaling(
    mock_fs_class, mock_get_params, waveform_params, mock_bbhwaveformfd
):
    """Test the TDI 2.0 rescaling."""
    _set_dummy_waveform(mock_bbhwaveformfd, waveform_params["length"])

    # Mock the FrequencySeries so we can check the multiplication
    mock_fs_instance = MagicMock()
    mock_fs_class.return_value = mock_fs_instance

    get_fd_waveform_from_bbhx(tdi='2.0', **waveform_params)

    # With align_tc_zero=True, the code does:
    #   fs = FrequencySeries(...)
    #   shifted = fs.cyclic_time_shift(shift_amount)
    #   output[ifo] = shifted * rescale
    # So __mul__ should be called on the result of cyclic_time_shift
    mock_fs_instance.cyclic_time_shift.return_value.__mul__.assert_called()


def test_get_fd_waveform_from_bbhx_invalid_length(waveform_params):
    """Test that an invalid length raises a ValueError."""
    waveform_params["length"] = 0
    with pytest.raises(ValueError, match="Waveform length must be positive"):
        get_fd_waveform_from_bbhx(**waveform_params)
