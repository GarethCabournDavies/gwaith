"""Validate a bank directly, prepare cached recovery, run shards, or merge reports."""
import argparse
import json
import logging
import multiprocessing as mp
from pathlib import Path

import numpy as np
from hwilas.search.config import load_config, draw
from hwilas.template_banks.io import load_bank
from hwilas.template_banks.stochastic import Bank, init_worker, generate
from hwilas.template_banks.verification import prepare, init_recovery, recover, fingerprint


def write_report(path, result):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(result))
    temporary.replace(path)


def merge(paths):
    reports = [json.loads(Path(p).read_text()) for p in paths]
    provenance = reports[0]['provenance']
    rows, seen = [], set()
    for report in reports:
        if report['provenance'] != provenance:
            raise ValueError('Incompatible shard provenance')
        for row in report['results']:
            i = row['index']
            if i in seen or not 0 <= i < provenance['n']:
                raise ValueError('Duplicate or out-of-range injection index')
            seen.add(i)
            rows.append(row)
    if len(seen) != provenance['n']:
        raise ValueError('Missing injections; refusing incomplete coverage report')
    rows.sort(key=lambda r: r['index'])
    good = np.array([r['match'] for r in rows if not r['error']])
    failures = len(rows) - len(good)
    summary = dict(n=len(rows), failures=failures, min_match=provenance['min_match'],
                   fraction_below_min_match=float(np.mean(good < provenance['min_match'])) if len(good) else None,
                   minimum=float(good.min()) if len(good) else None,
                   quantiles=np.quantile(good, [.01, .05, .5, .95]).tolist() if len(good) else [],
                   coverage_complete=failures == 0,
                   evaluated=sum(r.get('evaluated', 0) for r in rows),
                   rejected=sum(r.get('rejected', 0) for r in rows))
    return dict(provenance=provenance, summary=summary, results=rows)


def validate(a):
    """Measure independent prior draws against regenerated bank waveforms."""
    cfg = load_config(a.config)
    init_worker(cfg)
    bank = Bank(cfg)
    tests = draw(np.random.default_rng(a.seed), a.n, cfg['prior'])
    ff, failures = [], []
    with mp.get_context('spawn').Pool(a.nproc, initializer=init_worker, initargs=(cfg,)) as pool:
        for params, W, sig, err in pool.imap(generate, load_bank(a.bank_file)):
            if err:
                raise RuntimeError(f"Bank regeneration failed: {err}")
            bank.add(params, W, sig)
        for i, (params, W, sig, err) in enumerate(pool.imap(generate, tests)):
            if err:
                failures.append(dict(test=i, error=err))
                ff.append(None)
            else:
                ff.append(bank.best_match(params, W, sig))
    good = np.array([x for x in ff if x is not None])
    mm = cfg['bank'].get('min_match', .9)
    result = dict(bank=a.bank_file, config=cfg, seed=a.seed, n=a.n, fitting_factors=ff,
                  tests=tests, failures=failures,
                  fraction_below_min_match=float(np.mean(good < mm)) if len(good) else None,
                  min_match=mm)
    Path(a.output_file).write_text(json.dumps(result, indent=2))
    print(json.dumps({k:v for k,v in result.items() if k not in ('config','tests','fitting_factors')}, indent=2))
    if failures:
        raise RuntimeError(f"{len(failures)} validation waveforms failed; see output report")


def main():
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    p = sub.add_parser('validate', help='Direct independent fitting-factor validation')
    p.add_argument('--config', required=True)
    p.add_argument('--bank-file', required=True)
    p.add_argument('--output-file', required=True)
    p.add_argument('--n', type=int, default=400)
    p.add_argument('--seed', type=int, default=12345)
    p.add_argument('--nproc', type=int, default=4)
    p.add_argument('--verbose', action='store_true')
    p = sub.add_parser('prepare')
    p.add_argument('--config', required=True)
    p.add_argument('--bank-file', required=True)
    p.add_argument('--cache-dir', required=True)
    p = sub.add_parser('run')
    p.add_argument('--cache-dir', required=True)
    p.add_argument('--output-file', required=True)
    p.add_argument('--n', type=int, default=200000)
    p.add_argument('--seed', type=int, default=12345)
    p.add_argument('--shard-index', type=int, default=0)
    p.add_argument('--shard-count', type=int, default=1)
    p.add_argument('--nproc', type=int, default=1)
    p.add_argument('--template-batch', type=int, default=64)
    p.add_argument('--no-rejector', action='store_true')
    p = sub.add_parser('merge')
    p.add_argument('--output-file', required=True)
    p.add_argument('reports', nargs='+')
    a = parser.parse_args()
    if a.command == 'validate':
        if min(a.n, a.nproc) < 1 or a.seed < 0:
            parser.error('n and nproc must be positive; seed must be nonnegative')
        if a.verbose:
            logging.getLogger().setLevel(logging.DEBUG)
        validate(a)
    elif a.command == 'prepare':
        prepare(load_config(a.config), a.bank_file, a.cache_dir)
    elif a.command == 'run':
        if min(a.n, a.shard_count, a.nproc, a.template_batch) < 1 or a.seed < 0 or not 0 <= a.shard_index < a.shard_count:
            parser.error('Invalid injection, worker, seed, batch or shard controls')
        directory = Path(a.cache_dir)
        meta = json.loads((directory / 'metadata.json').read_text())
        for c, digest in meta['psd_sha256'].items():
            if fingerprint(meta['config']['bank']['psd_' + c[-1]]) != digest:
                raise ValueError('PSD changed since cache preparation')
        for c, digest in meta['spectra_sha256'].items():
            if fingerprint(directory / (c + '.npy')) != digest:
                raise ValueError('Cached spectrum checksum mismatch')
        provenance = dict(cache_sha256=fingerprint(directory / 'metadata.json'), n=a.n,
                          seed=a.seed, min_match=meta['config']['bank'].get('min_match', .9))
        # Validate initialization before spawning to avoid repeated worker failures.
        init_recovery(a.cache_dir, a.seed, not a.no_rejector, a.template_batch)
        with mp.get_context('spawn').Pool(a.nproc, initializer=init_recovery,
                 initargs=(a.cache_dir, a.seed, not a.no_rejector, a.template_batch)) as pool:
            indices = range(a.shard_index, a.n, a.shard_count)
            rows = []
            for row in pool.imap(recover, indices):
                rows.append(row)
                if len(rows) % 100 == 0 or len(rows) == len(indices):
                    logging.info('Shard %d: recovered %d/%d injections', a.shard_index, len(rows), len(indices))
        write_report(a.output_file, dict(provenance=provenance, results=rows))
        if any(r['error'] for r in rows):
            raise RuntimeError('Injection generation failures recorded in report')
    else:
        report = merge(a.reports)
        write_report(a.output_file, report)
        print(json.dumps(report['summary'], indent=2))
        if report['summary']['failures']:
            raise RuntimeError('Coverage incomplete: injection generation failures')


if __name__ == '__main__':
    main()
