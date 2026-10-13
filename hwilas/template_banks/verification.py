"""Cached exhaustive bank recovery with a conservative FFT rejector."""
import hashlib
import json
import logging
from pathlib import Path

import numpy as np
from scipy.fft import ifft
from hwilas.search.config import draw
from hwilas.template_banks.io import load_bank
from hwilas.template_banks import stochastic as sb

logger = logging.getLogger(__name__)
_STATE = {}


def fingerprint(path):
    digest = hashlib.sha256()
    with open(path, 'rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def prepare(cfg, bank_file, directory):
    """Write immutable, memory-mappable normalized spectra on shared storage."""
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    if any(directory.iterdir()):
        raise ValueError('Cache directory must be empty; use a new directory')
    sb.init_worker(cfg)
    params = load_bank(bank_file)
    n = round(2 * 86400 / cfg['waveform']['delta_t'])
    arrays = {c: np.lib.format.open_memmap(directory / (c + '.npy'), mode='w+',
              dtype=np.complex64, shape=(len(params), n // 2 + 1)) for c in sb.CHANNELS}
    for i, p in enumerate(params):
        _, W, _, error = sb.generate(p)
        if error:
            raise RuntimeError(f'Template {i} failed: {error}')
        for c in arrays:
            arrays[c][i] = W[c]
        logger.debug('Cached template %d', i)
        if (i + 1) % 100 == 0 or i + 1 == len(params):
            logger.info('Cached %d/%d templates', i + 1, len(params))
    for a in arrays.values():
        a.flush()
    meta = dict(config=cfg, bank_sha256=fingerprint(bank_file), templates=len(params),
                spectra_sha256={c: fingerprint(directory / (c + '.npy')) for c in arrays},
                psd_sha256={c: fingerprint(cfg['bank']['psd_' + c[-1]]) for c in arrays})
    (directory / 'metadata.json').write_text(json.dumps(meta, indent=2))


def init_recovery(directory, seed, rejector, batch):
    directory = Path(directory)
    meta = json.loads((directory / 'metadata.json').read_text())
    sb.init_worker(meta['config'])
    _STATE.update(meta=meta, seed=seed, rejector=rejector, batch=batch,
                  spectra={c: np.load(directory / (c + '.npy'), mmap_mode='r') for c in sb.CHANNELS})
    bank = sb.Bank(meta['config'])
    if not len(bank.lag_idx):
        raise ValueError('Match lag window must include at least one sample')
    _STATE.update(n=bank.n, lag_idx=bank.lag_idx)


def best_match(W, sig, spectra, n, lag_idx, batch=64, rejector=True):
    """Preserve placement's independent channel time/phase maxima.

    The triangle inequality bounds each channel overlap by sum(|h*g|).
    Only templates whose network bound exceeds the incumbent need an FFT.
    """
    weights = np.array([sig[c] ** 2 for c in sb.CHANNELS], float)
    weights /= weights.sum()
    best, winner, evaluated, rejected = 0., -1, 0, 0
    for start in range(0, len(spectra[sb.CHANNELS[0]]), batch):
        products = [spectra[c][start:start + batch] * np.conj(W[c]) for c in sb.CHANNELS]
        size = len(products[0])
        bounds = np.sqrt(sum(w * np.sum(np.abs(p), axis=1, dtype=np.float64) ** 2
                             for w, p in zip(weights, products)))
        # Float32 FFTs can overshoot the mathematical bound by roundoff.
        ids = np.flatnonzero(bounds + 1e-5 >= best) if rejector else np.arange(size)
        rejected += size - len(ids)
        evaluated += len(ids)
        if not len(ids):
            continue
        power = np.zeros(len(ids))
        for w, product in zip(weights, products):
            z = np.zeros((len(ids), n), np.complex64)
            z[:, :product.shape[1]] = product[ids]
            overlap = np.abs(ifft(z, axis=1, workers=1)[:, lag_idx] * n).max(axis=1)
            power += w * overlap ** 2
        matches = np.sqrt(power)
        k = int(np.argmax(matches))
        if winner < 0 or matches[k] > best:
            best, winner = float(matches[k]), start + int(ids[k])
    return best, winner, evaluated, rejected


def recover(index):
    # Per-injection seeds make results independent of shard count and scheduling.
    cfg = _STATE['meta']['config']
    p = draw(np.random.default_rng(np.random.SeedSequence([_STATE['seed'], index])), 1, cfg['prior'])[0]
    p, W, sig, error = sb.generate(p)
    result = dict(index=index, params=p, error=error)
    if not error:
        match, template, evaluated, rejected = best_match(W, sig, _STATE['spectra'],
            _STATE['n'], _STATE['lag_idx'], _STATE['batch'], _STATE['rejector'])
        result.update(match=match, template=template, evaluated=evaluated, rejected=rejected)
    return result
