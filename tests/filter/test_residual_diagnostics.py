import h5py
import numpy as np
import pytest
from hwilas.search.residual_diagnostics import read_window,match_catalogue_to_fits


@pytest.mark.parametrize('raw_dt',[.25,2.5])
def test_read_raw_slice_units_cadence_and_fractional_origin(tmp_path,raw_dt):
    path=tmp_path/'slice.h5'
    start=raw_dt;n=round(1000/raw_dt);frequency=.01;laser=2.816e14
    with h5py.File(path,'w') as f:
        f.attrs.update(fs=1/raw_dt,i0=1,laser_frequency=laser)
        f['a2']=laser*np.sin(2*np.pi*frequency*(start+np.arange(n)*raw_dt))
    values=read_window(path,['LISA_A'],200.,64,4.,laser,fractional_default=False)['LISA_A']
    np.testing.assert_allclose(values,np.sin(2*np.pi*frequency*(200.+np.arange(64)*4)),atol=.005)


def test_missing_coverage_is_nan(tmp_path):
    path=tmp_path/'processed.hdf'
    with h5py.File(path,'w') as f:
        f.attrs.update(epoch=100.,delta_t=4.,divide_by_laser_frequency=True)
        f['LISA_A']=[1.,2.,3.]
    values=read_window(path,['LISA_A'],96.,5,4.,2.816e14)['LISA_A']
    np.testing.assert_array_equal(values,[np.nan,1.,2.,3.,np.nan])


def test_close_injections_do_not_share_one_fit():
    matches=match_catalogue_to_fits([dict(source_id=1,tc=100000.),
        dict(source_id=2,tc=149108.)],[dict(tc=100020.)],3600.)
    assert matches=={1:0,2:None}


def test_noiseless_match_preserves_phase_and_channel_weighting():
    from hwilas.search.residual_diagnostics import noiseless_match
    x=np.sin(2*np.pi*np.arange(64)/8)
    injection={'LISA_A':x,'LISA_E':x}
    psds={'LISA_A':([.001,.5],[1.,1.]),'LISA_E':([.001,.5],[4.,4.])}
    assert noiseless_match(injection,{c:2*x for c in injection},1.,psds,.001,.5)==pytest.approx(1.)
    assert noiseless_match(injection,{c:-x for c in injection},1.,psds,.001,.5)==pytest.approx(-1.)
    assert noiseless_match(injection,{'LISA_A':x,'LISA_E':-x},1.,psds,.001,.5)==pytest.approx(.6)
    assert np.isnan(noiseless_match(injection,{c:np.zeros(64) for c in injection},1.,psds,.001,.5))
    missing=x.copy();missing[0]=np.nan
    assert np.isnan(noiseless_match({'LISA_A':missing,'LISA_E':x},injection,1.,psds,.001,.5))


def test_raw_timestamps_preserve_source_origin_offset(tmp_path):
    from hwilas.search.residual_diagnostics import raw_time_origin
    reference=tmp_path/'reference.h5'; source=tmp_path/'source.h5'
    origin=12345678.0;laser=2.816e14
    with h5py.File(reference,'w') as f:
        f.attrs.update(fs=4.,i0=100)
        f['t']=origin+25+np.arange(4)*.25
    start=100.+50.5
    with h5py.File(source,'w') as f:
        f.attrs.update(fs=.4,i0=40,laser_frequency=laser)
        t=start+np.arange(1000)*2.5
        f['t']=origin+t
        f['a2']=laser*np.sin(2*np.pi*.01*t)
    ref=raw_time_origin([reference,source])
    assert ref==pytest.approx(origin)
    values=read_window(source,['LISA_A'],400.,64,4.,laser,
                       fractional_default=False,raw_time_reference=ref)['LISA_A']
    np.testing.assert_allclose(values,np.sin(2*np.pi*.01*(400+np.arange(64)*4)),atol=.005)
    with pytest.raises(ValueError,match='shared raw_time_reference'):
        read_window(source,['LISA_A'],400.,64,4.,laser)


def test_raw_slice_call_infers_shared_origin_from_siblings(tmp_path):
    origin=12345678.0
    reference=tmp_path/'mbhb_source2_slice.h5'
    source=tmp_path/'mbhb_source10_slice.h5'
    with h5py.File(reference,'w') as f:
        f.attrs.update(fs=4.,i0=100)
        f['t']=origin+25+np.arange(4)*.25
    with h5py.File(source,'w') as f:
        f.attrs.update(fs=.4,i0=40,divide_by_laser_frequency=True)
        t=150.5+np.arange(1000)*2.5
        f['t']=origin+t
        f['a2']=np.sin(2*np.pi*.01*t)
    inferred=read_window(source,['LISA_A'],400.,64,4.,2.816e14)['LISA_A']
    explicit=read_window(source,['LISA_A'],400.,64,4.,2.816e14,
                         raw_time_reference=origin)['LISA_A']
    np.testing.assert_array_equal(inferred,explicit)
