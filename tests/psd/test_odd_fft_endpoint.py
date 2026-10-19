"""Padding odd time arrays must not drop the final target PSD bin."""
import numpy as np
from hwilas.psd.read import from_numpy_arrays


def test_odd_fft_padding_retains_target_length(caplog):
    source_n, target_n, dt = 101, 111, 4.
    freq = np.fft.rfftfreq(source_n, dt)
    noise = 1+freq
    target_df = 1/(target_n*dt)
    result = from_numpy_arrays(freq, noise, target_n//2+1, target_df, target_df)
    assert len(result) == target_n//2+1
    np.testing.assert_allclose(result.data[-1],noise[-1])
    assert not caplog.records


def test_material_frequency_shortfall_still_truncates(caplog):
    freq = np.array([0.,.01,.02,.03])
    result = from_numpy_arrays(freq,np.ones(4),101,.001,.001)
    assert len(result) == 31
    assert 'exceeds highest available' in caplog.text
