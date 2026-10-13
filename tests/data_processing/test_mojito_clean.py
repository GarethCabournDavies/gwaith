"""Check anti-alias suppression and continuity of block processing."""
import numpy as np
from scipy.signal import resample_poly
from hwilas.data_processing.mojito_clean import process_channel


def test_clean_crossfade_and_alias_rejection():
    fs = 4.
    n = 86400
    t = np.arange(n)/fs
    x = np.sin(2*np.pi*.02*t)+np.sin(2*np.pi*.3*t)
    y = process_channel(lambda a, b: x[a:b], n, keep_days=.08, pad_days=.04,
                        taper_days=.01, xfade_days=.02)
    tt = np.arange(len(y))*4
    interior = (tt > 1500) & (tt < tt[-1]-1500)
    error = y[interior]-np.sin(2*np.pi*.02*tt[interior])
    assert np.std(error) < .005
    assert len(y) == n//16
