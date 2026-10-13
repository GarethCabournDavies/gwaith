"""Placement decisions and configurable population distributions."""
from pathlib import Path
import numpy as np
import pytest
from hwilas.search.config import load_config, draw
from hwilas.template_banks.stochastic import Bank

CONFIG = Path(__file__).resolve().parents[2]/"examples/template_banks/mojito_bank.toml"


def test_priors_and_reproducibility():
    cfg = load_config(CONFIG)
    rows = draw(np.random.default_rng(7), 20000, cfg["prior"])
    assert rows == draw(np.random.default_rng(7), 20000, cfg["prior"])
    m1, m2 = np.array([[p['mass1'], p['mass2']] for p in rows]).T
    assert np.all((m1+m2 >= 3e5) & (m1+m2 <= 5e7))
    assert np.all((m1/m2 >= 1) & (m1/m2 <= 10))
    assert abs(np.mean(np.cos([p['inclination'] for p in rows]))) < .02
    assert abs(np.mean(np.sin([p['eclipticlatitude'] for p in rows]))) < .02
    assert abs(np.mean(np.log(m1+m2)) - np.mean(np.log([3e5, 5e7]))) < .03


def test_invalid_prior(tmp_path):
    p = tmp_path/'bad.toml'
    p.write_text('[prior.q]\nmin=0.5\nmax=2.0\n')
    with pytest.raises(ValueError, match='q >= 1'):
        load_config(p)


def test_duplicate_and_orthogonal_match():
    cfg = load_config(CONFIG)
    bank = Bank(cfg)
    nf = bank.n//2+1
    p = dict(mass1=1e6, mass2=5e5)
    one = np.zeros(nf, np.complex64); one[10] = 1
    two = np.zeros(nf, np.complex64); two[20] = 1
    W = {c: one for c in bank.W}
    sig = {c: 1. for c in bank.W}
    assert bank.best_match(p, W, sig) == 0
    bank.add(p, W, sig)
    assert bank.best_match(p, W, sig) == pytest.approx(1, abs=1e-6)
    assert bank.best_match(p, {c: two for c in bank.W}, sig) == pytest.approx(0, abs=1e-6)


def test_psd_dc_and_fractional_frequency_units(tmp_path):
    from hwilas.template_banks.stochastic import init_worker, _STATE
    cfg = load_config(CONFIG)
    freq = np.array([0., 1e-5, .01, .125])
    fractional = np.array([0., 2e-40, 3e-40, 4e-40])
    path = tmp_path/'psd.txt'
    np.savetxt(path, np.column_stack([freq, fractional]))
    cfg['bank'].update(psd_A=str(path), psd_E=str(path), psd_units='fractional_frequency')
    init_worker(cfg)
    scaled = _STATE['psd']['LISA_A'].copy()
    np.savetxt(path,np.column_stack([freq,fractional*cfg['waveform']['laser_frequency']**2]))
    cfg['bank']['psd_units'] = 'Hz'
    init_worker(cfg)
    np.testing.assert_allclose(_STATE['psd']['LISA_A'],scaled)


def test_bank_psd_must_cover_band(tmp_path):
    from hwilas.template_banks.stochastic import init_worker
    cfg = load_config(CONFIG)
    path=tmp_path/'psd.txt'
    np.savetxt(path,[[0.,0.],[.001,1.],[.01,1.]])
    cfg['bank'].update(psd_A=str(path),psd_E=str(path))
    with pytest.raises(ValueError,match='full placement frequency band'):
        init_worker(cfg)


def test_converged_build_report_is_json_serializable(tmp_path, monkeypatch):
    import json
    import h5py
    from hwilas.template_banks import stochastic as sb
    cfg=load_config(CONFIG)
    cfg['bank'].update(output_file=str(tmp_path/'bank.hdf'),window=2,batch=2,max_proposals=2,tolerance=.9)
    n=round(2*86400/cfg['waveform']['delta_t'])
    wave=np.zeros(n//2+1,np.complex64);wave[10]=1
    def generate(p):
        return p,{c:wave for c in sb.CHANNELS},{c:1. for c in sb.CHANNELS},None
    class Pool:
        def __enter__(self): return self
        def __exit__(self,*a): pass
        def map(self,fn,items): return list(map(fn,items))
    class Context:
        def Pool(self,*a,**k): return Pool()
    monkeypatch.setattr(sb,'init_worker',lambda cfg: None)
    monkeypatch.setattr(sb,'generate',generate)
    monkeypatch.setattr(sb.mp,'get_context',lambda *a: Context())
    result=sb.build(cfg,nproc=1)
    assert result['converged'] is True
    assert json.loads(json.dumps(result))['converged'] is True
    with h5py.File(cfg['bank']['output_file']) as f:
        assert f.attrs['converged'] and len(f['mass1'])==1
        first = {k:f[k][:] for k in f}
    # A fresh run replaces the same output and repeats seeded placement.
    sb.build(cfg,nproc=1)
    with h5py.File(cfg['bank']['output_file']) as f:
        for k, values in first.items():
            np.testing.assert_array_equal(f[k][:],values)
