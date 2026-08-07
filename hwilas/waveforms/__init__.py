"""
This is a wrapper for the pycbc waveform generation function.
"""

from pycbc.waveform import get_fd_det_waveform

def get_fd_waveform(
    approximant,
    **waveform_params
):
    """
    A wrapper for pycbc's get_fd_det_waveform.
    It allows specifying the waveform length in samples instead of the
    final frequency.
    """
    
    if "length" in waveform_params:
        flen = waveform_params["length"]
        delta_f = waveform_params["delta_f"]
        
        # This logic is taken from the test suite for the hwilas waveform generator
        # it is a trick to ensure that the pycbc waveform generator returns the
        # correct number of samples
        f_nyquist = (flen - 1) * delta_f
        
        params = waveform_params.copy()
        params.pop("length")
        params["f_final"] = f_nyquist + delta_f / 2
        
        return get_fd_det_waveform(approximant=approximant, **params)

    return get_fd_det_waveform(
        approximant=approximant,
        **waveform_params
    )
