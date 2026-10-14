"""Compare physical coalescence-phase mappings with all other parameters fixed."""
import logging

import numpy as np
from scipy.optimize import minimize_scalar
from scipy.signal import correlate, correlation_lags

from hwilas.search.residual_diagnostics import waveform_window

logger = logging.getLogger(__name__)


def alignment_metrics(data, model, dt, max_lag_seconds=3600.):
    """Measure detector-grid alignment independently of recorded coalescence times.

    Positive lag means the model would be delayed to match the data. The common
    A/E lag maximizes signed correlation, without fitting phase or amplitude.
    """
    channels = ['LISA_A', 'LISA_E']
    n = len(data[channels[0]])
    if dt <= 0 or max_lag_seconds < 0 or n == 0:
        raise ValueError('Alignment requires a positive cadence and nonempty data')
    if any(len(data[c]) != n or len(model[c]) != n or
           not np.isfinite(data[c]).all() or not np.isfinite(model[c]).all() for c in channels):
        raise ValueError('Alignment requires finite arrays on the same grid')
    data_power = sum(np.asarray(data[c])**2 for c in channels)
    model_power = sum(np.asarray(model[c])**2 for c in channels)
    if not data_power.sum() > 0 or not model_power.sum() > 0:
        raise ValueError('Alignment requires nonzero data and model power')
    cross = sum(correlate(data[c], model[c], mode='full', method='fft') for c in channels)
    lags = correlation_lags(n, n)
    allowed = np.abs(lags*dt) <= max_lag_seconds
    best = np.argmax(np.where(allowed, cross, -np.inf))
    lag = int(lags[best])
    shifted = {c: np.zeros_like(np.asarray(x), dtype=float) for c, x in model.items()}
    for c, x in model.items():
        if lag >= 0:
            shifted[c][lag:] = x[:n-lag]
        else:
            shifted[c][:n+lag] = x[-lag:]
    return dict(best_common_lag_seconds=float(lag*dt),
                lag_at_search_boundary=bool(abs(lag*dt) == max_lag_seconds),
                data_peak_offset_seconds=float(np.argmax(data_power)*dt),
                model_peak_offset_seconds=float(np.argmax(model_power)*dt),
                zero_lag=phase_metrics(data, model), at_best_lag=phase_metrics(data, shifted))


def phase_conventions(phase):
    """Candidate mappings from catalogue reference phase to generator phase."""
    candidates = {'catalogue': phase, 'negative catalogue': -phase,
                  'half catalogue': phase/2, 'negative half catalogue': -phase/2,
                  'double catalogue': 2*phase, 'negative double catalogue': -2*phase,
                  'zero': 0.}
    for sign, label in [(1, 'catalogue'), (-1, 'negative catalogue')]:
        for offset, suffix in [(np.pi/2, '+ pi/2'), (np.pi, '+ pi'), (3*np.pi/2, '+ 3pi/2')]:
            candidates[f'{label} {suffix}'] = sign*phase+offset
    return {label: float(value % (2*np.pi)) for label, value in candidates.items()}


def phase_metrics(data, model):
    """Amplitude-sensitive RMS errors and signed overlaps at fixed time and phase."""
    result = {}
    for channels, label in [(['LISA_A', 'LISA_E'], 'AE'), *[([c], c) for c in data]]:
        a = np.concatenate([np.asarray(data[c]) for c in channels])
        b = np.concatenate([np.asarray(model[c]) for c in channels])
        power, model_power = np.dot(a, a), np.dot(b, b)
        valid = np.isfinite(a).all() and np.isfinite(b).all() and power > 0
        result[f'{label}_relative_rms'] = float(np.sqrt(np.dot(a-b, a-b)/power)) if valid else np.inf
        result[f'{label}_overlap'] = float(np.dot(a, b)/np.sqrt(power*model_power)) if valid and model_power > 0 else np.nan
        result[f'{label}_amplitude_ratio'] = float(np.sqrt(model_power/power)) if valid else np.nan
    return result


def scan_phase(data, params, metadata, epoch, dt, laser_scale=1., phase_count=16,
               waveform_generator=None):
    """Test named mappings and refine a free physical phase; never fit time or amplitude.

    Data should contain the target injection, with a fixed model of overlapping
    sources removed if necessary. The returned free phase is an empirical fit,
    rather than evidence of a universal catalogue convention.
    """
    if phase_count < 4:
        raise ValueError('At least four phases are required')
    n = len(data['LISA_A'])
    if n == 0 or any(len(x) != n or not np.isfinite(x).all() for x in data.values()):
        raise ValueError('Phase scan requires finite, equal-length channel data')
    if sum(float(np.dot(data[c], data[c])) for c in ['LISA_A', 'LISA_E']) <= 0:
        raise ValueError('Phase scan requires nonzero A/E data')
    cache = {}
    generate = waveform_window if waveform_generator is None else waveform_generator

    def evaluate(phase):
        phase = float(phase % (2*np.pi))
        if phase not in cache:
            logger.debug('Generating source %s at physical phase %.9g', params.get('source_id'), phase)
            wave = generate({**params, 'coa_phase': phase}, metadata, list(data), epoch, n, dt,
                                   laser_scale=laser_scale)
            cache[phase] = phase_metrics(data, wave)
        return cache[phase]

    mappings = phase_conventions(params['coa_phase'])
    report = [dict(convention=label, phase=phase, **evaluate(phase)) for label, phase in mappings.items()]
    phases = np.linspace(0., 2*np.pi, phase_count, endpoint=False)
    curve = [dict(phase=float(p), **evaluate(p)) for p in phases]
    seed = min(curve, key=lambda r: r['AE_relative_rms'])['phase']
    step = 2*np.pi/phase_count
    optimum = minimize_scalar(lambda p: evaluate(p)['AE_relative_rms'],
                              bounds=(seed-step, seed+step), method='bounded',
                              options={'xatol': 1e-4, 'maxiter': 30})
    best = min([*report, *curve,
                dict(phase=float(optimum.x % (2*np.pi)), **evaluate(optimum.x))],
               key=lambda r: r['AE_relative_rms'])
    report.append(dict(best, convention='free phase (diagnostic fit)'))
    best_wave = generate({**params, 'coa_phase': best['phase']}, metadata, list(data), epoch, n, dt,
                                laser_scale=laser_scale)
    baseline = generate(params, metadata, list(data), epoch, n, dt, laser_scale=laser_scale)
    return report, curve, baseline, best_wave
