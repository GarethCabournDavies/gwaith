"""All search backends use the same short-template sampling contract."""
import numpy as np
from gwaith.types import TimeSeries
from hwilas.waveforms import short_templates as module


def test_non_fast_search_template_dispatch_and_taper(monkeypatch):
    calls=[]
    def generate(model,tdi,**params):
        calls.append((model,tdi,params))
        return {c:TimeSeries(np.ones(params['length']),epoch=params['start_time'],delta_t=params['delta_t'])
                for c in ['LISA_A','LISA_E','LISA_T']}
    monkeypatch.setattr(module,'get_fd_waveform',generate)
    monkeypatch.setattr(module,'reference_frequency',lambda *a:0.)
    settings=dict(approximant='example',orbit='esa',data_time_origin=1234.,delta_t=4.,
        f_lower=1e-5,f_upper=.09,tdi='2',pre_merger_seconds=400.,post_merger_seconds=200.,
        taper_seconds=40.,laser_frequency=2.,generator_keywords={'custom':True})
    wave=module.short_template({'tc':0.},settings,modes=[(2,2)])
    model,tdi,params=calls[0]
    assert model=='example' and tdi=='2'
    assert params['custom'] and params['f_ref']==0. and params['orbit']=='esa'
    assert params['mode_array']==[(2,2)]
    assert params['start_time']+100*4==params['tc']
    assert len(wave['LISA_A'])==150
    assert wave['LISA_A'][0]==0. and wave['LISA_A'][-1]==2.
