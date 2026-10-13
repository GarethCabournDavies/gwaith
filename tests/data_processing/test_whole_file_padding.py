"""Whole-file padding preserves samples, units and absolute time coordinates."""
import h5py
import numpy as np
import pytest
from gwaith.types import TimeSeries, FrequencySeries
from hwilas.data_processing.inpainting import gate_and_paint


def test_coloured_inpainting_zeros_overwhitened_gap():
    n=64
    rng=np.random.default_rng(18)
    x=rng.normal(size=n)
    spectrum=1+np.arange(n//2+1)**2/20
    inverse=1/spectrum
    # Build the overwhitening operator and solve its gap equations directly.
    K=np.column_stack([np.fft.irfft(np.fft.rfft(np.eye(n)[:,i])*inverse,n=n) for i in range(n)])
    missing=np.arange(0,8)
    observed=np.arange(8,n)
    expected=np.linalg.solve(K[np.ix_(missing,missing)],-K[np.ix_(missing,observed)]@x[observed])
    result=gate_and_paint(TimeSeries(x,epoch=0.,delta_t=4.),[0],[8],FrequencySeries(spectrum,delta_f=1/(4*n)))
    np.testing.assert_allclose(result.data[missing],expected,rtol=1e-9,atol=1e-10)
    np.testing.assert_array_equal(result.data[observed],x[observed])
