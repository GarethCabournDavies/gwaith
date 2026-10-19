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
        assert params['response_nodes'] == 4096
        assert params['padding'] == 1200
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
                thm_oversampling=4, data_time_origin=12345678., template_scale=1.,
                generator_keywords={'response_nodes':4096, 'padding':1200})
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


def test_oem_notice_is_debug_only_and_other_warnings_are_preserved(caplog):
    import logging
    logger = logging.getLogger('lisaorbits.oem')
    notice = module.OEMInterpolationNotice()
    logger.addFilter(notice)
    message = ('OEM preferred interpolation method ignored, using spline '
               'interpolation (see InterpolatedOrbits for details)')
    try:
        with caplog.at_level(logging.INFO, logger=logger.name):
            logger.warning(message)
            logger.warning('Unexpected orbit warning')
        assert [r.getMessage() for r in caplog.records] == ['Unexpected orbit warning']
        caplog.clear()
        with caplog.at_level(logging.DEBUG, logger=logger.name):
            logger.warning(message)
        assert caplog.records[0].levelno == logging.DEBUG
    finally:
        logger.removeFilter(notice)


def test_refinement_reseeds_phase_and_compares_in_final_model(monkeypatch):
    calls = []
    def profile(candidate, **kwargs):
        assert kwargs['waveform_params_shared']['waveform_model'] == 'phenomthm'
        assert not kwargs['logarithmic_mass']
        calls.append(candidate.copy())
        physical = candidate.copy()
        physical[7] = 123.
        # The original bank seed is better in THM despite the other broad-model winner.
        return (20. if candidate[0] == 2. else 10.) + np.cos(candidate[9]-np.pi), physical
    monkeypatch.setattr(module, 'profile_physical_fit', profile)
    broad = np.zeros(10); broad[0] = 1.
    bank = np.zeros(10); bank[0] = 2.
    lower = np.zeros(10); upper = np.ones(10); upper[9] = 2*np.pi
    statistic, best = module.reseed_refinement([broad, bank],
        {'waveform_params_shared': {'waveform_model':'phenomthm'}, 'logarithmic_mass':True},
        lower, upper, 4)
    assert statistic == 21.
    assert best[0] == 2.
    assert best[7] == 123.
    assert best[9] == np.pi
    assert len(calls) == 10


def test_event_switches_model_before_refinement_and_output(tmp_path, monkeypatch):
    import h5py
    from types import SimpleNamespace
    from gwaith.types import FrequencySeries
    path = tmp_path/'data.hdf'
    with h5py.File(path, 'w') as f:
        f.attrs.update(epoch=0., delta_t=4.)
        for channel in ('LISA_A','LISA_E','LISA_T'):
            f[channel] = np.zeros(100)
    minimum = dict(mchirp=1e5, eta=.1, spin1z=-.9, spin2z=-.9,
        eclipticlongitude=0., eclipticlatitude=-1., inclination=0., tc=190.,
        polarization=0., coa_phase=0.)
    maximum = dict(mchirp=2e6, eta=.25, spin1z=.9, spin2z=.9,
        eclipticlongitude=2*np.pi, eclipticlatitude=1., inclination=np.pi, tc=210.,
        polarization=np.pi, coa_phase=2*np.pi)
    args = SimpleNamespace(waveform_model='fastbhb', refinement_waveform_model='phenomthm',
        orbit='analytic', refinement_orbit='esa', refinement_f_ref=1e-4,
        generator_keywords={}, refinement_generator_keywords={'response_nodes':4096}, f_lower=3e-5, f_upper=.09,
        f_ref=3e-5, tdi_generation='2', fit_channels=['LISA_A','LISA_E'],
        optimisation_channels=['LISA_A','LISA_E'], output_timeseries_channels=['LISA_A','LISA_E','LISA_T'],
        optimisation_window_before=86400., optimisation_window_after=14400.,
        minimum_bounds=minimum, maximum_bounds=maximum, bounds_window_relative_to_initial={},
        seed=42, n_particles=4, n_iterations=1, n_processes=1, optimizer='pso',
        phase_seed_count=2, polish_evaluations=5, mass_coordinate='log', data_time_origin=12345678.)
    bank = dict(mass1=1e6, mass2=5e5, spin1z=0., spin2z=0., eclipticlongitude=1.,
        eclipticlatitude=.2, inclination=.8, polarization=.3, distance=1e4, coa_phase=.4)
    monkeypatch.setattr(module, 'psd_from_txt', lambda path,n,df,*a,**k:
                        FrequencySeries(np.ones(n),delta_f=df))
    args.psd_files = {}; args.psd_file = 'unused'
    models = []
    def profile(v, **kwargs):
        model = kwargs['waveform_params_shared']['waveform_model']
        full = np.insert(v,7,kwargs['tc_reference']) if len(v)==9 else v.copy()
        if kwargs.get('logarithmic_mass'):
            full[0] = 10**full[0]
        return (100. if model=='fastbhb' else 5.), full
    monkeypatch.setattr(module, 'profile_physical_fit', profile)
    monkeypatch.setattr(module, 'compute_statistic_core', lambda *a,**k: 5.)
    def population(args, population, lower, upper, kwargs, idx):
        assert kwargs['waveform_params_shared']['waveform_model']=='fastbhb'
        assert kwargs['waveform_params_shared']['generator_keywords']=={}
        return -100., population[0]
    monkeypatch.setattr(module, 'optimise_population', population)
    def build(v, data, shared, *a, waveform_channels=None, **k):
        models.append(shared['waveform_model'])
        assert shared['generator_keywords']=={'response_nodes':4096}
        assert shared['orbit']=='esa'
        assert shared['f_ref']==1e-4
        first=next(iter(data.values()))
        return {c:FrequencySeries(np.zeros(len(first)//2+1,dtype=complex),
                    delta_f=1/(len(first)*first.delta_t)) for c in waveform_channels or data}
    monkeypatch.setattr(module, 'build_aligned_waveform', build)
    monkeypatch.setattr(module, 'coherent_fit', lambda *a: (6.,1.))
    record, epoch, segments = module._optimise_event(
        (args,0,1,str(path),200.,200.,0,50.,bank,4.,False,-1))
    assert models and set(models)=={'phenomthm'}
    assert record['initial_fit_stat']==5.  # Broad-model statistic must not trigger fallback.
    assert record['optimised_stat']==6.
    assert record['f_lower']==3e-5
    assert set(segments)==set(args.output_timeseries_channels)
