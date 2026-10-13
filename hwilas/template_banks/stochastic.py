"""Stochastic placement using time/phase-maximised network matches.

Proposals use configurable priors. The stopping rule measures recent acceptance;
it is not a guarantee of minimal-match coverage. Validate with independent draws.
"""
import json
import logging
import multiprocessing as mp
from pathlib import Path
import time
import h5py
import numpy as np
from scipy.fft import ifft
from hwilas.search.config import draw
from hwilas.template_banks.io import load_bank, write_bank
from hwilas.template_banks.frequency import lower_frequency_for_duration, lower_frequency_for_snr, duration_from_frequency
from hwilas.waveforms.short_fastbhb import CHANNELS, short_template, frequency_template

logger = logging.getLogger(__name__)
_STATE = {}


def init_worker(cfg):
    _STATE["cfg"] = cfg
    w, b = cfg["waveform"], cfg["bank"]
    n = round(2*86400/w["delta_t"])
    f = np.fft.rfftfreq(n, w["delta_t"])
    _STATE['frequencies'] = f
    _STATE['integration_frequencies'] = np.geomspace(w['f_lower'], w['f_upper'],
                                                    w.get('snr_frequency_samples', 32768))
    _STATE['integration_psd'] = {}
    _STATE["psd"] = {}
    units = b.get("psd_units", "Hz")
    if units not in ("Hz", "fractional_frequency"):
        raise ValueError("psd_units must be Hz or fractional_frequency")
    scale = w["laser_frequency"]**2 if units == "fractional_frequency" else 1.
    for c in CHANNELS:
        a = np.loadtxt(b["psd_"+c[-1]])
        if a.ndim != 2 or a.shape[1] != 2 or len(a) < 2 or not np.isfinite(a).all():
            raise ValueError("PSD files require finite frequency/PSD columns")
        if np.any(a[:, 0] < 0) or np.any(np.diff(a[:, 0]) <= 0):
            raise ValueError("PSD frequencies must be nonnegative and strictly increasing")
        # DC is outside the placement band; exclude it before log interpolation.
        a = a[a[:, 0] > 0]
        if len(a) < 2 or np.any(a[:, 1] <= 0):
            raise ValueError("PSD files require positive PSDs at positive frequencies")
        if a[0, 0] > w["f_lower"] or a[-1, 0] < w["f_upper"]:
            raise ValueError("PSD files must cover the full placement frequency band")
        s = np.exp(np.interp(np.log(np.maximum(f, 1e-20)), np.log(a[:, 0]), np.log(a[:, 1]))) * scale
        _STATE["psd"][c] = np.where((f >= w["f_lower"]) & (f <= w["f_upper"]), s, np.inf)
        _STATE['integration_psd'][c] = np.exp(np.interp(
            np.log(_STATE['integration_frequencies']), np.log(a[:, 0]), np.log(a[:, 1]))) * scale


def generate(p):
    try:
        w = _STATE["cfg"]["waveform"]
        if 'snr_fraction' in w:
            f = _STATE['integration_frequencies']
            full = frequency_template(p, f, w, modes=[(2, 2)])
            density = sum(abs(full[c])**2/_STATE['integration_psd'][c] for c in CHANNELS)
            cutoff = lower_frequency_for_snr(f, density, w['snr_fraction'])
            p = {**p, 'f_lower': cutoff, 'pre_merger_seconds': duration_from_frequency(p, cutoff)}
            # A common frequency grid permits matching templates of different durations.
            f = _STATE['frequencies']
            mask = (f >= cutoff) & (f <= w['f_upper'])
            selected = frequency_template(p, f[mask], {**w, 'f_lower': cutoff}, modes=[(2, 2)])
            spectra = {c: np.zeros(len(f), complex) for c in CHANNELS}
            for c in CHANNELS:
                spectra[c][mask] = selected[c]
            df = f[1]-f[0]
        else:
            p = {**p, "f_lower": lower_frequency_for_duration(p, w)}
            tm = short_template(p, {**w, "f_lower": p["f_lower"]}, modes=[(2, 2)])
            df = 1/(len(tm[CHANNELS[0]])*w["delta_t"])
            spectra = {c: w['delta_t']*np.fft.rfft(tm[c]) for c in CHANNELS}
        W, sig = {}, {}
        for c in CHANNELS:
            h = spectra[c]/np.sqrt(_STATE["psd"][c])
            # Use the same per-template cutoff bins as matched filtering.
            h[:int(p["f_lower"]/df)] = 0
            power = np.sum(abs(h)**2)
            if not np.isfinite(power) or power <= 0:
                raise ValueError("Nonpositive template norm")
            sig[c] = float(np.sqrt(4*df*power))
            W[c] = (h/np.sqrt(power)).astype(np.complex64)
        return p, W, sig, None
    except Exception as e:
        return p, None, None, repr(e)


class Bank:
    def __init__(self, cfg):
        self.cfg = cfg
        self.n = round(2*86400/cfg["waveform"]["delta_t"])
        lag = np.fft.fftfreq(self.n)*self.n*cfg["waveform"]["delta_t"]
        self.lag_idx = np.flatnonzero(abs(lag) <= cfg["bank"].get("max_lag_seconds", 500.))
        self.params, self.sigmas, self.mc = [], [], []
        self.W = {c: [] for c in CHANNELS}

    def add(self, p, W, sig):
        self.params.append(p)
        self.sigmas.append(sig)
        self.mc.append((p["mass1"]*p["mass2"])**.6/(p["mass1"]+p["mass2"])**.2)
        for c in CHANNELS:
            self.W[c].append(W[c])

    def best_match(self, p, W, sig, threshold=1.):
        mc = (p["mass1"]*p["mass2"])**.6/(p["mass1"]+p["mass2"])**.2
        order = np.argsort(abs(np.log(np.asarray(self.mc)/mc)))
        best = 0.
        for start in range(0, len(order), 64):
            ids = order[start:start+64]
            num = np.zeros(len(ids))
            for c in CHANNELS:
                z = np.zeros((len(ids), self.n), np.complex64)
                z[:, :len(W[c])] = np.asarray([self.W[c][i] for i in ids])*np.conj(W[c])
                overlap = abs(ifft(z, axis=1, workers=1)[:, self.lag_idx]*self.n).max(axis=1)
                num += (overlap*sig[c])**2
            best = max(best, float(np.sqrt(num/sum(s*s for s in sig.values())).max()))
            if best >= threshold:
                break
        return best


def build(cfg, nproc=4):
    b = cfg["bank"]
    mm, tol, window = b.get("min_match", .9), b.get("tolerance", .02), b.get("window", 200)
    limit, batch = b.get("max_proposals", 200000), b.get("batch", 40)
    if not 0 < mm <= 1 or not 0 < tol < 1 or min(window, limit, batch, nproc) < 1:
        raise ValueError("Invalid placement controls")
    init_worker(cfg)  # Fail once, before spawning, for missing/malformed PSD inputs.
    bank = Bank(cfg)
    rng = np.random.default_rng(b.get("seed", 1))
    recent, count, failed, converged = [], 0, 0, False
    output = Path(b["output_file"])
    output.parent.mkdir(parents=True, exist_ok=True)
    t0 = time.monotonic()
    with mp.get_context("spawn").Pool(nproc, initializer=init_worker, initargs=(cfg,)) as pool:
        if b.get("resume"):
            for p, W, sig, err in pool.imap(generate, load_bank(b["resume"])):
                if err:
                    raise RuntimeError(f"Cannot regenerate resume template: {err}")
                bank.add(p, W, sig)
        while count < limit:
            proposals = draw(rng, min(batch, limit-count), cfg["prior"])
            for p, W, sig, err in pool.map(generate, proposals):
                count += 1
                if err:
                    failed += 1
                    logger.warning("Proposal failed: %s", err)
                    continue
                accept = bank.best_match(p, W, sig, mm) < mm
                if accept:
                    bank.add(p, W, sig)
                recent = (recent+[bool(accept)])[-window:]
            logger.info("%d proposals, %d templates, %d failures, %.1fs", count, len(bank.params), failed, time.monotonic()-t0)
            # Completion metadata also feeds the CLI JSON report.
            converged = bool(len(recent) == window and np.mean(recent) < tol)
            write_bank(output, bank.params, cfg["waveform"], config_json=json.dumps(cfg),
                       min_match=mm, n_proposals=count, n_failed=failed, converged=converged,
                       f_lower_method=("per-template network SNR retention" if 'snr_fraction' in cfg['waveform']
                                       else "per-template IMRPhenomD time-to-merger with analysis floor"),
                       snr_fraction=cfg['waveform'].get('snr_fraction', 0.))
            if converged:
                break
    if not bank.params:
        raise RuntimeError("No valid templates generated")
    return dict(templates=len(bank.params), proposals=count, failures=failed, converged=converged,
                seconds=time.monotonic()-t0)
