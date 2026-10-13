"""Exhaustive recovery equivalence and complete shard accounting."""
import json
import numpy as np
import pytest
from hwilas.template_banks.stochastic import Bank, CHANNELS
from hwilas.template_banks.verification import best_match
from tools.template_banks.verify_bank import merge


def test_rejector_preserves_exhaustive_match():
    cfg = {'waveform': {'delta_t': 2700.}, 'bank': {'max_lag_seconds': 5400.}}
    bank = Bank(cfg)
    rng = np.random.default_rng(42)
    nf = bank.n // 2 + 1
    waves = {}
    for c in CHANNELS:
        x = (rng.normal(size=(130, nf)) + 1j*rng.normal(size=(130, nf))).astype('complex64')
        x /= np.linalg.norm(x, axis=1)[:, None]
        waves[c] = x
    # The first template is identical; later disjoint spectra can be rejected.
    W = {c: waves[c][0].copy() for c in CHANNELS}
    for c in CHANNELS:
        W[c][nf//2:] = 0
        W[c] /= np.linalg.norm(W[c])
        waves[c][0] = W[c]
        waves[c][64:, :nf//2] = 0
        waves[c][64:] /= np.linalg.norm(waves[c][64:], axis=1)[:, None]
    sig = dict(zip(CHANNELS, [1., 3.]))
    for i in range(130):
        bank.add({'mass1': 1e6, 'mass2': 5e5}, {c: waves[c][i] for c in CHANNELS}, sig)
    exhaustive = best_match(W, sig, waves, bank.n, bank.lag_idx, rejector=False)
    bounded = best_match(W, sig, waves, bank.n, bank.lag_idx)
    assert bounded[:2] == exhaustive[:2]
    assert bounded[3] >= 66
    assert bounded[0] == pytest.approx(bank.best_match(bank.params[0], W, sig, threshold=2), abs=1e-6)
    for i in [1, 50, 100]:
        W = {c: waves[c][i] for c in CHANNELS}
        assert best_match(W, sig, waves, bank.n, bank.lag_idx)[:2] == best_match(W, sig, waves, bank.n, bank.lag_idx, rejector=False)[:2]


def test_merge_rejects_missing_duplicate_and_incompatible(tmp_path):
    provenance = dict(n=2, seed=1, min_match=.9)
    paths = [tmp_path/'a.json', tmp_path/'b.json']
    for i, p in enumerate(paths):
        p.write_text(json.dumps(dict(provenance=provenance, results=[dict(index=i, match=.95, error=None)])))
    assert merge(paths)['summary']['fraction_below_min_match'] == 0
    with pytest.raises(ValueError, match='Missing'):
        merge(paths[:1])
    with pytest.raises(ValueError, match='Duplicate'):
        merge([paths[0], paths[0]])
    paths[1].write_text(json.dumps(dict(provenance={**provenance, 'seed': 2}, results=[])))
    with pytest.raises(ValueError, match='Incompatible'):
        merge(paths)


def test_cached_recovery_is_independent_of_scheduling(tmp_path, monkeypatch):
    from hwilas.template_banks import verification as v
    from hwilas.search.config import load_config
    from hwilas.template_banks.io import write_bank
    from pathlib import Path
    cfg = load_config(Path(__file__).resolve().parents[2]/'examples/template_banks/mojito_bank.toml')
    cfg['waveform']['delta_t'] = 2700.
    psd = tmp_path/'psd.txt'
    psd.write_text('0.00001 1\n0.1 1\n')
    cfg['bank'].update(psd_A=str(psd), psd_E=str(psd))
    params = dict(mass1=1e6, mass2=5e5, spin1z=0., spin2z=0.,
                  eclipticlatitude=0., eclipticlongitude=0., inclination=0.,
                  polarization=0., coa_phase=0., distance=10000.)
    bank = tmp_path/'bank.hdf'
    write_bank(bank, [params], cfg['waveform'])
    wave = np.zeros(33, np.complex64)
    wave[3] = 1
    monkeypatch.setattr(v.sb, 'init_worker', lambda cfg: None)
    monkeypatch.setattr(v.sb, 'generate', lambda p: (p, {c: wave for c in CHANNELS}, {c: 1. for c in CHANNELS}, None))
    cache = tmp_path/'cache'
    v.prepare(cfg, bank, cache)
    v.init_recovery(cache, 77, True, 64)
    sequential = [v.recover(i) for i in range(6)]
    shuffled = sorted([v.recover(i) for i in [0, 2, 4, 1, 3, 5]], key=lambda r: r['index'])
    assert sequential == shuffled
    assert all(r['match'] == pytest.approx(1., abs=1e-6) for r in sequential)
    assert isinstance(v._STATE['spectra'][CHANNELS[0]], np.memmap)
    with pytest.raises(ValueError, match='empty'):
        v.prepare(cfg, bank, cache)


def test_failed_injections_mark_coverage_incomplete(tmp_path):
    report = tmp_path/'report.json'
    report.write_text(json.dumps(dict(provenance=dict(n=2, min_match=.9), results=[
        dict(index=0, error='generation failed'), dict(index=1, error=None, match=.8)])))
    result = merge([report])['summary']
    assert result['failures'] == 1
    assert not result['coverage_complete']
    assert result['fraction_below_min_match'] == 1


@pytest.mark.parametrize('mode, fail_job', [('slurm', False), ('bash', False), ('slurm', True)])
def test_readable_workflow_waits_before_merge_and_plot(tmp_path, mode, fail_job):
    import os
    import subprocess
    import sys
    from pathlib import Path
    scripts = Path(__file__).resolve().parents[2]/'examples/template_banks'
    workflow = tmp_path/'verify_bank_slurm.sh'
    source = (scripts/workflow.name).read_text()
    workflow.write_text(source.replace('RUN_MODE="slurm"', f'RUN_MODE="{mode}"')
                       .replace('injection_batches=200', 'injection_batches=3'))
    (tmp_path/'verify_bank_injections.sbatch').write_text((scripts/'verify_bank_injections.sbatch').read_text())
    record = tmp_path/'calls.jsonl'
    stub = r"""import json, os, pathlib, subprocess, sys
name = pathlib.Path(sys.argv[0]).name
args = sys.argv[1:]
with pathlib.Path(os.environ['WORKFLOW_RECORD']).open('a') as f:
    f.write(json.dumps(dict(name=name, args=args)) + '\n')
if name == 'sbatch':
    assert '--wait' in args
    if os.environ['FAIL_JOB'] == '1':
        sys.exit(1)
    script_index = next(i for i, a in enumerate(args) if a.endswith('.sbatch'))
    for i in range(3):
        subprocess.run(['bash', *args[script_index:]],
                       env={**os.environ, 'SLURM_ARRAY_TASK_ID': str(i)}, check=True)
elif name == 'hwilas_verify_bank' and args[0] == 'run':
    pathlib.Path(args[args.index('--output-file') + 1]).write_text('{}')
elif name == 'hwilas_verify_bank' and args[0] == 'merge':
    reports = args[3:]
    assert len(reports) == 3 and all(pathlib.Path(p).exists() for p in reports)
    pathlib.Path(args[2]).write_text('{}')
elif name == 'hwilas_plot_bank_verification':
    assert pathlib.Path(args[2]).exists()
"""
    for executable in ['sbatch', 'hwilas_verify_bank', 'hwilas_plot_bank_verification']:
        command = tmp_path/executable
        command.write_text(f'#!{sys.executable}\n' + stub)
        command.chmod(0o755)
    env = {**os.environ, 'PATH': str(tmp_path) + os.pathsep + os.environ['PATH'],
           'WORKFLOW_RECORD': str(record), 'FAIL_JOB': str(int(fail_job))}
    result = subprocess.run(['bash', str(workflow), 'config.toml', 'bank.hdf', str(tmp_path/'output with spaces')],
                            env=env, capture_output=True, text=True)
    calls = [json.loads(line) for line in record.read_text().splitlines()]
    assert calls[0]['args'][0] == 'prepare'
    if fail_job:
        assert result.returncode != 0
        assert len(calls) == 2
    else:
        assert result.returncode == 0, result.stderr
        assert calls[-2]['args'][0] == 'merge'
        assert calls[-1]['name'] == 'hwilas_plot_bank_verification'
        runs = [c for c in calls if c['name'] == 'hwilas_verify_bank' and c['args'][0] == 'run']
        assert len(runs) == 3
        assert [c['args'][c['args'].index('--shard-index') + 1] for c in runs] == ['0', '1', '2']


@pytest.mark.parametrize('matches', [[.8, .95], []])
def test_plot_report_counts_failures(tmp_path, monkeypatch, matches):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from tools.plotting.plot_bank_verification import plot_report
    path = tmp_path/'shard-0.json'
    path.write_text(json.dumps(dict(provenance=dict(min_match=.9), results=[
        *[dict(error=None, match=m) for m in matches],
        dict(error='waveform failed', match=100.)])))
    # Capture plotted values to check that failed injections are excluded.
    plotted = []
    original = plt.Axes.hist
    def histogram(ax, values, *args, **kwargs):
        plotted.extend(values)
        return original(ax, values, *args, **kwargs)
    monkeypatch.setattr(plt.Axes, 'hist', histogram)
    output = plot_report(path, tmp_path/'plots')
    assert plotted == matches
    assert output.read_bytes().startswith(b'\x89PNG')
    assert plt.get_fignums() == []


@pytest.mark.parametrize('failed', [False, True])
def test_direct_validation_subcommand_preserves_report(tmp_path, monkeypatch, failed):
    from tools.template_banks import verify_bank as cli
    import sys
    from types import SimpleNamespace
    cfg = {'prior': {}, 'bank': {'min_match': .9}}
    monkeypatch.setattr(cli, 'load_config', lambda path: cfg)
    monkeypatch.setattr(cli, 'init_worker', lambda cfg: None)
    monkeypatch.setattr(cli, 'load_bank', lambda path: [{'bank': True}])
    monkeypatch.setattr(cli, 'draw', lambda rng,n,prior: [{'test': i} for i in range(n)])
    class Bank:
        def __init__(self, cfg): self.rows = []
        def add(self, params, wave, sig): self.rows.append(params)
        def best_match(self, params, wave, sig):
            assert self.rows == [{'bank': True}]
            return .8 if params['test']==0 else .95
    class Pool:
        def __enter__(self): return self
        def __exit__(self, *args): pass
        def imap(self, function, params):
            for p in params:
                error = 'waveform failed' if failed and p.get('test')==0 else None
                yield p, {}, {}, error
    monkeypatch.setattr(cli, 'Bank', Bank)
    monkeypatch.setattr(cli.mp, 'get_context', lambda mode: SimpleNamespace(Pool=lambda *a,**k: Pool()))
    output = tmp_path/'validation.json'
    monkeypatch.setattr(sys,'argv',['verify_bank','validate','--config','config.toml',
        '--bank-file','bank.hdf','--output-file',str(output),'--n','2','--verbose'])
    if failed:
        with pytest.raises(RuntimeError, match='1 validation waveforms failed'): cli.main()
    else:
        cli.main()
    report = json.loads(output.read_text())
    assert report['n']==2
    assert report['fitting_factors']==([None,.95] if failed else [.8,.95])
    assert report['fraction_below_min_match']==(0. if failed else .5)
    assert len(report['failures'])==int(failed)
