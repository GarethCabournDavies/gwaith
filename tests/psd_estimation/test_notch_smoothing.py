"""Wide smoothing intervals retain the Hann blending convention."""
import numpy as np
from gwaith.types import FrequencySeries
from hwilas.psd_estimation.smooth import smooth_psd


def test_wide_regions_use_hann_blend():
    from scipy.signal.windows import hann
    f=np.arange(12501)*1e-5
    data=np.ones(len(f))
    regions=[(.02,.04),(.05,.07),(.08,.1)]
    for centre in [.030075,.06015,.090225]:
        data[abs(f-centre)<.0015]=1e-6
    original=FrequencySeries(data,delta_f=1e-5)
    result=smooth_psd({'A':original},['A'],regions)['A']
    expected=data.copy()
    for lo,hi in regions:
        a=np.argmin(abs(f-lo)); b=np.argmin(abs(f-hi))
        w=hann(b-a)
        expected[a:b]=w+data[a:b]*(1-w)
    np.testing.assert_allclose(result.data,expected)
    np.testing.assert_array_equal(original.data,data)


def test_hann_edges_retain_notch_at_boundary():
    f=np.arange(10001)*1e-5
    data=np.ones(len(f));data[abs(f-.030075)<.0001]=1e-6
    result=smooth_psd({'A':FrequencySeries(data,delta_f=1e-5)},['A'],[(.03,.09)])['A']
    assert result.data[np.argmin(abs(f-.030075))] < 1e-3
