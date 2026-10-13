"""Verify common-time network detection, time labels, and incomplete-run handling."""
import json
from pathlib import Path
import h5py
import numpy as np
import pytest
from hwilas.search import blind
from hwilas.search.config import load_config
from hwilas.template_banks.io import write_bank

CONFIG = Path(__file__).resolve().parents[2]/'examples/search/mojito_local.toml'


def setup(tmp_path):
    cfg = load_config(CONFIG)
    cfg['search']['whole_file'] = False
    cfg['search'].pop('psd_A', None)
    cfg['search'].pop('psd_E', None)
    cfg['search'].update(data_file=str(tmp_path/'data.h5'), bank_file=str(tmp_path/'bank.h5'),
                         output_dir=str(tmp_path/'out'), gate_days=0., apply_ratio_veto=False)
    p = dict(mass1=1e6, mass2=5e5, spin1z=0., spin2z=0., eclipticlatitude=0.,
             eclipticlongitude=0., inclination=1., polarization=.2, coa_phase=0., distance=1e4)
    write_bank(tmp_path/'bank.h5', [p], cfg['waveform'])
    n, _, _ = blind.geometry(cfg)
    dt = cfg['waveform']['delta_t']
    rng = np.random.default_rng(3)
    nt = round(2*86400/dt)
    j = np.arange(nt)
    centre = round(1.5*86400/dt)
    chirp = np.exp(-((j-centre)/400)**2)*np.cos(.2*(j-centre)+.0001*(j-centre)**2)
    tm = {'LISA_A': chirp, 'LISA_E': .6*chirp}
    k0 = round(8.5*86400/dt)
    with h5py.File(tmp_path/'data.h5', 'w') as f:
        f.attrs.update(epoch=1234., delta_t=dt)
        for c, h in tm.items():
            x = rng.normal(size=n)
            x[k0:k0+nt] += 5*h
            f[c] = x
    return cfg, p, tm


def test_common_time_recovery_and_stale_input(tmp_path, monkeypatch):
    cfg, p, tm = setup(tmp_path)
    blind.prepare(cfg)
    blind.init_worker(cfg)
    monkeypatch.setattr(blind, 'short_template', lambda *a, **k: tm)
    i, rows = blind.search_template((0, p))
    best = rows[np.argmax(rows[:, 2])]
    assert abs(best[1]-(1234+10*86400)) <= 4
    assert best[2] > 80
    cfg['search']['snr_threshold'] = 9.
    with pytest.raises(ValueError, match='changed since prepare'):
        blind.checked_manifest(cfg)


def test_incomplete_and_empty_collect(tmp_path):
    cfg, _, _ = setup(tmp_path)
    blind.prepare(cfg)
    with pytest.raises(ValueError, match='incomplete'):
        blind.collect(cfg)
    partial = blind.collect(cfg, allow_partial=True)
    assert not partial['complete'] and partial['final_candidates'] == []
    out = Path(cfg['search']['output_dir'])/'triggers'
    out.mkdir()
    np.save(out/'t0.npy', np.empty((0, 3)))
    result = blind.collect(cfg)
    assert result['complete'] and result['final_candidates'] == []


def test_clustering_keeps_loudest():
    rows = np.array([[0, 1., 10.], [1, 2., 20.], [2, 100., 9.]])
    assert blind.cluster(rows, 10).tolist() == [[1., 2., 20.], [2., 100., 9.]]


def test_fractional_frequency_units_match_Hz(tmp_path):
    cfg, _, _ = setup(tmp_path)
    blind.prepare(cfg)
    baseline = np.load(Path(cfg['search']['output_dir'])/'Q.npy').copy()
    with h5py.File(cfg['search']['data_file'], 'r+') as f:
        f.attrs.update(divide_by_laser_frequency=True, laser_frequency=cfg['waveform']['laser_frequency'])
        for c in ('LISA_A', 'LISA_E'):
            f[c][:] = f[c][:]/cfg['waveform']['laser_frequency']
    cfg['search']['output_dir'] = str(tmp_path/'fractional')
    meta = blind.prepare(cfg)
    assert meta['input_units'] == 'fractional_frequency'
    assert np.allclose(np.load(tmp_path/'fractional'/'Q.npy'), baseline, rtol=1e-5, atol=1e-6)


def test_whole_file_zero_padding_preserves_valid_extent(tmp_path, monkeypatch):
    cfg, p, tm = setup(tmp_path)
    cfg['search']['whole_file'] = True
    # An odd, non-fast input requires FFT padding at the end.
    with h5py.File(cfg['search']['data_file'], 'r+') as f:
        for c in ('LISA_A', 'LISA_E'):
            x = f[c][:-1]
            del f[c]
            f[c] = x
    m = blind.prepare(cfg)
    assert m['chunks'] == 1
    assert m['n'] > m['samples']
    assert m['last_valid_time'] == 1234+(m['samples']-m['edge'])*4
    blind.init_worker(cfg)
    monkeypatch.setattr(blind, 'short_template', lambda *a, **k: tm)
    _, rows = blind.search_template((0,p))
    best = rows[np.argmax(rows[:,2])]
    assert abs(best[1]-(1234+10*86400)) <= 4
    assert np.all(rows[:,1] < m['last_valid_time'])


@pytest.mark.parametrize('channel', ['LISA_A', 'LISA_E'])
def test_per_channel_recovery_and_trigger_export(tmp_path, monkeypatch, channel):
    cfg, p, tm = setup(tmp_path)
    cfg['search']['channel'] = channel
    m = blind.prepare(cfg)
    assert m['channels'] == [channel]
    assert np.load(Path(cfg['search']['output_dir'])/'Q.npy').shape[1] == 1
    blind.init_worker(cfg)
    monkeypatch.setattr(blind, 'short_template', lambda *a, **k: tm)
    _, rows = blind.search_template((0,p))
    best = rows[np.argmax(rows[:,2])]
    assert abs(best[1]-(1234+10*86400)) <= 4
    out = Path(cfg['search']['output_dir'])/'triggers'
    out.mkdir()
    np.save(out/'t0.npy',rows)
    blind.collect(cfg)
    with h5py.File(out.parent/'triggers.hdf') as f:
        assert f.attrs['channel'] == channel
        assert f.attrs['complete']
        assert set(f) == {'template_id','end_time','snr'}
        assert len(f['end_time']) == len(rows)
        assert np.all(f['template_id'][:] == 0)
        np.testing.assert_allclose(np.sort(f['snr'][:]),np.sort(rows[:,2]))


def test_unselected_channel_does_not_affect_preparation(tmp_path):
    cfg, _, _ = setup(tmp_path)
    cfg['search']['channel'] = 'LISA_A'
    with h5py.File(cfg['search']['data_file'],'r+') as f:
        f['LISA_E'][:] = np.nan
    assert blind.prepare(cfg)['channels'] == ['LISA_A']
