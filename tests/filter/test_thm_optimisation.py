"""THM follow-up retains physical scaling and suppresses out-of-band aliases."""
import numpy as np
import pytest
from gwaith.types import TimeSeries
from tools.search import optimise_snr_peaks as module
from hwilas.conversions import mchirp_eta_from_mass1_mass2


def test_thm_resampling_rejects_signal_above_detector_nyquist(monkeypatch):
    from hwilas.waveforms import phenomthm
    monkeypatch.setattr(module, '_thm_orbit', lambda kind: object())
    frequency = [.02]
    def generate(approximant, **params):
        t = params['start_time']+np.arange(params['length'])*params['delta_t']
        return {c: TimeSeries(np.sin(2*np.pi*frequency[0]*t), epoch=params['start_time'],
                              delta_t=params['delta_t']) for c in params['ifos']}
    monkeypatch.setattr(phenomthm, 'get_td_waveform_from_phenomthm', generate)
    monkeypatch.setattr(module, 'lower_frequency_for_duration',
                        lambda *args: pytest.fail('THM must not use PhenomD frequency timing'))
    mc,eta = mchirp_eta_from_mass1_mass2(1e6,5e5)
    position=np.array([mc,eta,.1,-.1,1.,.2,.8,1000.,.3,.4])
    data={'LISA_A': TimeSeries(np.zeros(2048),epoch=0.,delta_t=4.)}
    shared=dict(waveform_model='phenomthm', f_lower=.001, f_ref=.001, f_upper=.09,
                thm_oversampling=4, data_time_origin=12345678., template_scale=1.)
    low=module.build_aligned_waveform(position,data,shared,raise_err=True)['LISA_A'].data
    frequency[0]=.2
    high=module.build_aligned_waveform(position,data,shared,raise_err=True)['LISA_A'].data
    assert np.linalg.norm(high) < .02*np.linalg.norm(low)
    frequency[0]=.02
    scaled=module.build_aligned_waveform(position,data,{**shared,'template_scale':3.},raise_err=True)['LISA_A'].data
    np.testing.assert_allclose(scaled,3*low,atol=1e-12)


def test_log_mass_is_decoded_before_waveform_and_saved_as_physical(monkeypatch):
    captured = []
    def build(position, *args):
        captured.append(position.copy())
        return {'LISA_A': object()}
    monkeypatch.setattr(module, 'build_aligned_waveform', build)
    monkeypatch.setattr(module, 'profile_merger_time', lambda *args: (12., 80.))
    data = {'LISA_A': TimeSeries(np.zeros(32), epoch=0., delta_t=4.)}
    position = np.array([np.log10(11.8e6), .2, 0., 0., 1., .2, .8, .3, .4])
    statistic, physical = module.profile_physical_fit(
        position, data, {}, {'data_fs': {'LISA_A': object()}}, ['LISA_A'],
        tc_reference=64., time_bounds=(0., 100.), logarithmic_mass=True)
    assert statistic == 12.
    assert captured[0][0] == pytest.approx(11.8e6)
    assert physical[0] == pytest.approx(11.8e6)
    assert physical[7] == 80.
    assert position[0] == np.log10(11.8e6)
