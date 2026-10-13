import json
import sys
from pathlib import Path

import h5py
import numpy as np
import pytest

from hwilas.search.rounds import previous_window_flags, combine_parameters, subtract_files
from tools.search.check_search_residuals import check_residuals


def write_series(path, values, epoch=100., dt=4.):
    with h5py.File(path,'w') as f:
        f.attrs.update(epoch=epoch,delta_t=dt,divide_by_laser_frequency=True,laser_frequency=2.816e14)
        for c in ('LISA_A','LISA_E','LISA_T'):
            f[c] = values


def test_previous_window_flags_are_informational_and_half_open(tmp_path):
    path = tmp_path/'first.hdf'
    with h5py.File(path,'w') as f:
        f['tc'] = [10.,30.]
        f['optimisation_window_start'] = [0.,20.]
        f['optimisation_window_end'] = [20.,40.]
    flags,indices = previous_window_flags([0.,19.,20.,40.],path)
    np.testing.assert_array_equal(flags,[True,True,True,False])
    np.testing.assert_array_equal(indices,[0,0,1,-1])


def test_multi_round_subtraction_and_catalogue(tmp_path):
    data,s0,s1,r0,final = [tmp_path/f'{name}.hdf' for name in ('data','s0','s1','r0','final')]
    noise = np.linspace(-1e-20,1e-20,100)
    signal0,signal1 = np.full(100,2e-20),np.full(100,3e-20)
    write_series(data,noise+signal0+signal1)
    write_series(s0,signal0)
    write_series(s1,signal1)
    channels = ['LISA_A','LISA_E','LISA_T']
    subtract_files(data,[s0],r0,channels,chunk_size=17)
    subtract_files(data,[s0,s1],final,channels,chunk_size=17)
    with h5py.File(final) as f:
        np.testing.assert_allclose(f['LISA_A'][:],noise,atol=1e-35,rtol=1e-14)
        assert f.attrs['divide_by_laser_frequency']
        assert f.attrs['delta_t'] == 4.
        assert f.attrs['epoch'] == 100.
    paths = []
    for i,n in [(0,1),(1,2)]:
        path = tmp_path/f'p{i}.hdf'
        paths.append(path)
        with h5py.File(path,'w') as f:
            f['tc'] = np.arange(n)+10.*i
            f['mchirp'] = np.ones(n)*1e6
            f['in_previous_window'] = np.full(n,bool(i))
    combined_path = tmp_path/'parameters.hdf'
    table = combine_parameters(paths,combined_path,tmp_path/'parameters.csv')
    np.testing.assert_array_equal(table['search_round'],[0,1,1])
    np.testing.assert_array_equal(table['in_previous_window'],[False,True,True])
    report,_ = check_residuals(data,final,[s0,s1],channels,parameter_file=combined_path,chunk_size=32)
    assert report['subtraction_consistent']
    assert report['fitted_signal_count'] == 3
    json.dumps(report)
    with h5py.File(final,'r+') as f:
        f['LISA_A'][10] += 1e-20
    report,_ = check_residuals(data,final,[s0,s1],channels,chunk_size=32)
    assert not report['subtraction_consistent']


def test_second_round_keeps_sub40_candidate(tmp_path):
    from types import SimpleNamespace
    from tools.search.optimise_snr_peaks import get_events_to_optimise
    path = tmp_path/'coinc.hdf'
    with h5py.File(path,'w') as f:
        f['template_id'] = [0]
        for c in ('LISA_A','LISA_E'):
            f[f'time_{c}'] = [200000.]
            f[f'snr_{c}'] = [10.]
            f[f'trigger_id_{c}'] = [0]
    args = SimpleNamespace(input_file=path,optimisation_channels=['LISA_A','LISA_E'],
                           statistic_threshold=0.,minimum_peak_separation=86400.)
    snrs,_,_,_ = get_events_to_optimise(args)
    assert len(snrs) == 1
    assert snrs[0] < 40.


@pytest.mark.parametrize('n_processes,event_workers',[(1,1),(2,1),(1,2)])
def test_real_fastbhb_two_round_fit_and_residual(tmp_path,monkeypatch,n_processes,event_workers):
    pytest.importorskip('ldc.waveform.lisabeta')
    from gwaith.types import TimeSeries, FrequencySeries
    from tools.search import optimise_snr_peaks as optimise
    from hwilas.template_banks.io import write_bank
    from hwilas.conversions import mchirp_eta_from_mass1_mass2
    monkeypatch.chdir(tmp_path)
    dt,n = 4.,194400
    channels = ['LISA_A','LISA_E','LISA_T']
    times = [216000.,265108.]
    p = dict(mass1=2e6,mass2=1e6,spin1z=.1,spin2z=-.1,eclipticlongitude=1.,
             eclipticlatitude=.2,inclination=.8,polarization=.4,coa_phase=.3,distance=1e4)
    mc,eta = mchirp_eta_from_mass1_mass2(p['mass1'],p['mass2'])
    shared = dict(f_lower=3e-5,f_upper=.09,tdi='2',distance=p['distance'],template_scale=1.)
    injected = {c:np.zeros(n) for c in channels}
    rng = np.random.default_rng(17)
    noise = {c:rng.normal(scale=1e-20,size=n) for c in channels}
    for tc,target in zip(times,[100.,20.]):
        epoch = tc-172800.
        data = {c:TimeSeries(np.zeros(86400),delta_t=dt,epoch=epoch) for c in channels[:2]}
        v = np.array([mc,eta,.1,-.1,1.,.2,.8,tc,.4,.3])
        wf = optimise.build_aligned_waveform(v,data,shared,True,waveform_channels=channels)
        if tc == times[0]:
            distant = optimise.build_aligned_waveform(
                v,data,{**shared,'distance':2*p['distance']},True,waveform_channels=channels)
            for c in channels:
                np.testing.assert_allclose(distant[c].data,.5*wf[c].data,rtol=1e-10,atol=0.)
        psd = {c:FrequencySeries(np.full(len(wf[c]),8e-40),delta_f=wf[c].delta_f) for c in channels[:2]}
        rho,_ = optimise.coherent_fit(wf,wf,psd,channels[:2])
        offset = round(epoch/dt)
        for c in channels:
            injected[c][offset:offset+86400] += target/rho*np.fft.irfft(wf[c].data,n=86400)/dt
    source = tmp_path/'data.hdf'
    with h5py.File(source,'w') as f:
        f.attrs.update(epoch=0.,delta_t=dt,divide_by_laser_frequency=True)
        for c in channels:
            f[c] = injected[c]+noise[c]
    bank = tmp_path/'bank.hdf'
    write_bank(bank,[p],dict(f_lower=3e-5,f_upper=.09,pre_merger_days=1.))
    psd_path = tmp_path/'psd.txt'
    np.savetxt(psd_path,np.column_stack([np.linspace(0,.125,1000),np.full(1000,8e-40)]))
    seeds = dict(mchirp=mc,eta=eta,spin1z=.1,spin2z=-.1,eclipticlongitude=1.,
                 eclipticlatitude=.2,inclination=.8,polarization=.4,coa_phase=.3)
    lower = [f'{key}:{value-1e-6*max(1.,abs(value))}' for key,value in seeds.items()]
    upper = [f'{key}:{value+1e-6*max(1.,abs(value))}' for key,value in seeds.items()]
    parameters,signals = [],[]
    residual = source
    for r in range(2):
        coinc = tmp_path/f'coinc{r}.hdf'
        with h5py.File(coinc,'w') as f:
            candidate_times = times if r==0 else times[1:]
            candidate_snrs = [100.,20.] if r==0 else [20.]
            f['template_id'] = np.zeros(len(candidate_times),dtype=int)
            for c in channels[:2]:
                f[f'time_{c}'] = candidate_times
                f[f'snr_{c}'] = np.asarray(candidate_snrs)/np.sqrt(2)
                f[f'trigger_id_{c}'] = np.arange(len(candidate_times))
        parameter,signal = tmp_path/f'parameters{r}.hdf',tmp_path/f'signal{r}.hdf'
        argv = ['optimise','--input-file',str(coinc),'--bank-file',str(bank),
            '--data-files',str(residual),'--psd-file',str(psd_path),'--tdi-generation','2',
            '--output-parameter-file',str(parameter),'--output-timeseries-file',str(signal),
            '--statistic-threshold',str(40 if r==0 else 0),'--n-particles',str(n_processes),
            '--n-processes',str(n_processes),'--event-workers',str(event_workers),
            '--n-iterations','1','--polish-evaluations','0',
            '--optimisation-window-before','172800','--optimisation-window-after','172800',
            '--f-lower','3e-5','--bounds-window-relative-to-initial','tc:0.1',
            '--minimum-bounds',*lower,'--maximum-bounds',*upper]
        if r:
            argv += ['--previous-parameter-file',str(parameters[0])]
        else:
            first_argv = argv.copy()
        monkeypatch.setattr(sys,'argv',argv)
        optimise.main()
        parameters.append(parameter)
        signals.append(signal)
        residual = tmp_path/f'residual{r}.hdf'
        subtract_files(source,signals,residual,channels)
    if event_workers>1:
        # Fit both overlapping windows together and compare serial/parallel outputs.
        for workers in [1,2]:
            argv = first_argv.copy()
            for flag,value in [('--event-workers',str(workers)),('--statistic-threshold','0'),
                    ('--minimum-peak-separation','0'),
                    ('--n-particles','2'),('--polish-evaluations','20'),
                    ('--output-parameter-file',str(tmp_path/f'compare_parameters{workers}.hdf')),
                    ('--output-timeseries-file',str(tmp_path/f'compare_signal{workers}.hdf'))]:
                if flag in argv:
                    argv[argv.index(flag)+1] = value
                else:
                    argv += [flag,value]
            monkeypatch.setattr(sys,'argv',argv)
            optimise.main()
        for kind in ['parameters','signal']:
            with h5py.File(tmp_path/f'compare_{kind}1.hdf') as serial, h5py.File(tmp_path/f'compare_{kind}2.hdf') as parallel:
                for key in serial:
                    np.testing.assert_array_equal(serial[key][:],parallel[key][:])
                if kind=='parameters':
                    assert len(parallel['tc'])==2
                    assert np.all(np.diff(parallel['initial_stat'][:])<0)
                    assert np.all(parallel['polish_evaluations'][:]>0)
    with h5py.File(parameters[1]) as f:
        assert len(f['tc']) == 1
        assert f['initial_stat'][0] < 40.
        assert f['in_previous_window'][0]
        assert f['previous_event_index'][0] == 0
    catalogue = tmp_path/'all_parameters.hdf'
    combined = combine_parameters(parameters,catalogue,tmp_path/'parameters.csv')
    assert len(combined['tc']) == 2
    report,_ = check_residuals(source,residual,signals,channels,parameter_file=catalogue)
    assert report['subtraction_consistent']
    with h5py.File(residual) as f:
        model_error = sum(np.sum((f[c][:]-noise[c])**2) for c in channels[:2])
        original_signal_energy = sum(np.sum(injected[c]**2) for c in channels[:2])
    assert model_error/original_signal_energy < .02
    # Search both channels at the same samples and cluster across templates.
    from tools.search import matched_filter_search, trigger_merge, check_search_residuals
    trigger = tmp_path/'final_joint.hdf'
    monkeypatch.setattr(sys,'argv',['matched_filter','--input-data-file',str(residual),
        '--bank-file',str(bank),'--psd-files',f'LISA_A:{psd_path}',f'LISA_E:{psd_path}',
        '--channels','LISA_A','LISA_E','--tdi-generation','2',
        '--statistic-threshold','4','--output-file',str(trigger)])
    matched_filter_search.main()
    final_coinc = tmp_path/'final_candidates.hdf'
    monkeypatch.setattr(sys,'argv',['trigger_merge','--trigger-files',str(trigger),
        '--cluster-across-templates','--output-file',str(final_coinc)])
    trigger_merge.main()
    report_file = tmp_path/'report.json'
    monkeypatch.setattr(sys,'argv',['check_residuals','--data-file',str(source),
        '--residual-file',str(residual),'--signal-files',*[str(path) for path in signals],
        '--parameter-file',str(catalogue),'--coincidence-file',str(final_coinc),
        '--output-file',str(report_file),'--output-plot',str(tmp_path/'residual_psd.png')])
    check_search_residuals.main()
    report = json.loads(report_file.read_text())
    assert report['subtraction_consistent']
    assert report['residual_coincidences_above_diagnostic_threshold'] == 0
    assert (tmp_path/'residual_psd.png').exists()


def test_rebuilt_model_contains_only_retained_parameters(tmp_path,monkeypatch):
    from hwilas.search.rounds import rebuild_signal_model
    from hwilas.search import residual_diagnostics
    paths=[]
    for i,times in enumerate(([100.],[101.,5000.])):
        p=tmp_path/f'params{i}.hdf';paths.append(p)
        with h5py.File(p,'w') as f:
            f['tc']=times
            f['optimisation_window_start']=np.full(len(times),100.)
            f['optimisation_window_end']=np.full(len(times),100.+8*4)
    combined=tmp_path/'combined.hdf'
    rows=combine_parameters(paths,combined,close_window=3600.,exclude_later_close=True)
    np.testing.assert_array_equal(rows['tc'],[100.,5000.])
    data,model,residual=[tmp_path/f'{n}.hdf' for n in ('data','model','residual')]
    write_series(data,np.full(8,6000.))
    monkeypatch.setattr(residual_diagnostics,'waveform_window',
        lambda p,a,c,e,n,dt,laser_scale:{ch:np.full(n,p['tc']) for ch in c})
    rebuild_signal_model(data,combined,model,['LISA_A','LISA_E','LISA_T'])
    subtract_files(data,[model],residual,['LISA_A','LISA_E','LISA_T'])
    with h5py.File(model) as f:np.testing.assert_array_equal(f['LISA_A'][:],np.full(8,5100.))
    with h5py.File(residual) as f:np.testing.assert_array_equal(f['LISA_A'][:],np.full(8,900.))
