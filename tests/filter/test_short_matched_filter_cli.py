"""The existing search CLI detects short templates and writes standard channel triggers."""
import h5py
import numpy as np
import pytest
from hwilas.template_banks.io import write_bank
from tools.search.matched_filter_search import main


@pytest.mark.parametrize('channel',['LISA_A','LISA_E'])
@pytest.mark.parametrize('inspiral_days', [1., 3.])
def test_short_template_cli_preserves_arrival_times(tmp_path,monkeypatch,channel,inspiral_days):
    dt=4.; n=345600
    support = inspiral_days*86400.+3600.
    j=np.arange(round((support+86400.)/dt)); centre=round(support/dt)
    wave=np.exp(-((j-centre)/400)**2)*np.cos(.2*(j-centre)+.0001*(j-centre)**2)
    x=np.random.default_rng(19).normal(size=n)
    start=54000
    x[start:start+len(wave)] += 5*wave
    data,bank,psd,out=[tmp_path/name for name in ('data.hdf','bank.hdf','psd.txt','triggers.hdf')]
    with h5py.File(data,'w') as f:
        f.attrs.update(epoch=1000.,delta_t=dt,divide_by_laser_frequency=True)
        f[channel]=x
    p=dict(mass1=1e6,mass2=5e5,spin1z=0.,spin2z=0.,eclipticlatitude=0.,
           eclipticlongitude=0.,inclination=1.,polarization=.2,coa_phase=0.,distance=1e4)
    bank_cutoff = 3e-5 if channel == 'LISA_A' else 1e-4
    monkeypatch.setattr('tools.search.matched_filter_search.duration_from_frequency',
                        lambda params, frequency: inspiral_days*86400.)
    write_bank(bank,[p],dict(f_lower=bank_cutoff,f_upper=.09,pre_merger_days=1.))
    f=np.linspace(0,.125,1000)
    np.savetxt(psd,np.column_stack([f,np.full(len(f),8.)]))
    def short(params,settings,modes):
        assert params['tc']==2*86400.
        assert settings['laser_frequency']==1.
        assert settings['f_lower']==bank_cutoff
        assert settings['pre_merger_seconds']==support
        assert settings['post_merger_seconds']==86400.
        return {channel:wave}
    monkeypatch.setattr('tools.search.matched_filter_search.short_template',short)
    from tools.search import matched_filter_search as search
    original_filter = search.matched_filter_frequencyseries
    def filter_with_edge_peak(*args, **kwargs):
        assert kwargs["n_samples"] >= n-round(3*86400/dt)+round(2*86400/dt)
        snr = original_filter(*args, **kwargs)
        # A loud peak inside the full template length must never reach the output.
        snr.data[round(1.75*86400/dt)] = 1e6
        return snr
    monkeypatch.setattr(search, 'matched_filter_frequencyseries', filter_with_edge_peak)

    monkeypatch.setattr('sys.argv',['hwilas_matched_filter_search','--input-data-file',str(data),
        '--bank-file',str(bank),'--psd-file',str(psd),'--channel',channel,
        '--tdi-generation','2','--snr-threshold','8','--output-file',str(out)])
    main()
    with h5py.File(out) as f:
        assert set(f)=={'template_id','end_time','statistic'}
        np.testing.assert_array_equal(f.attrs['analysis_intervals'],
            [[1000+2*86400,1000+n*dt-5*86400]])
        peak=np.argmax(f['statistic'][:])
        assert abs(f['end_time'][peak]-(1000+(start+centre)*dt)) <= dt
        assert np.all(f['end_time'][:] >= 1000+max(2*86400, len(wave)*dt))
        assert np.all(f['end_time'][:] < 1000+n*dt-5*86400)


@pytest.mark.parametrize('days', [.2, 6.])
def test_template_geometry_tracks_frequency_duration(monkeypatch, days):
    from hwilas.waveforms import short_fastbhb as short
    captured = {}
    def template(params, start, n, settings, modes):
        captured.update(start=start, n=n, settings=settings)
        return {'LISA_A': np.ones(n), 'LISA_E': np.ones(n)}
    monkeypatch.setattr(short, 'template', template)
    settings = dict(delta_t=4., pre_merger_seconds=days*86400.,
                    post_merger_seconds=86400., taper_seconds=3600.)
    p = {'tc': 172800.}
    result = short.short_template(p, settings)
    n, merger = short.template_geometry(p, settings)
    assert n == round((days+1)*86400/4)
    assert merger == round(days*86400/4)
    assert captured['settings']['reference_tc'] >= merger*4
    assert captured['start']-p['tc']+captured['settings']['reference_tc'] >= 0
    assert len(result['LISA_A']) == n


def test_local_search_aligns_variable_duration_bank(monkeypatch):
    from hwilas.search import blind
    dt, n = 3600., 720
    merger, start = 73, 48
    wave = np.zeros(97); wave[merger] = 1.
    data = np.zeros(n); data[start+merger] = 1.
    cfg = dict(waveform=dict(delta_t=dt, taper_seconds=3600.),
               search=dict(channel='LISA_A', whole_file=True, snr_threshold=.001))
    monkeypatch.setattr(blind, '_G', dict(cfg=cfg,
        meta=dict(n=n, edge=36, dt=dt, samples=n, epoch=0.),
        starts=[0], Q=(dt*np.fft.rfft(data))[None, None, :],
        IS=np.ones((1, 1, n//2+1))))
    monkeypatch.setattr(blind, 'duration_from_frequency', lambda *a: 3*86400.)
    monkeypatch.setattr(blind, 'short_template', lambda *a, **k: {'LISA_A': wave})
    _, rows = blind.search_template((0, dict(f_lower=.001, pre_merger_seconds=3*86400.)))
    assert rows[np.argmax(rows[:, 2]), 1] == (start+merger)*dt
