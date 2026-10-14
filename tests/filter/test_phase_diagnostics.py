"""Physical phase scans recover phase errors without absorbing amplitude or time errors."""
import numpy as np
import pytest
from hwilas.search import phase_diagnostics as diagnostic


def test_phase_scan_recovers_physical_multimode_phase(monkeypatch):
    t = np.linspace(0, 2*np.pi, 256, endpoint=False)
    calls = []

    def waveform(params, metadata, channels, epoch, n, dt, laser_scale):
        calls.append(params.copy())
        phase = params['coa_phase']
        # Distinct harmonics require regeneration under a single physical phase.
        x = np.cos(2*t+2*phase)+.4*np.cos(3*t+3*phase)
        return {c: laser_scale*x*(i+1) for i, c in enumerate(channels)}

    monkeypatch.setattr(diagnostic, 'waveform_window', waveform)
    params = dict(coa_phase=.6, tc=100., distance=2000., polarization=.2)
    phase = -params['coa_phase'] % (2*np.pi)
    data = waveform({**params, 'coa_phase': phase}, {}, ['LISA_A', 'LISA_E', 'LISA_T'], 0, 256, 1, 1)
    report, curve, baseline, best = diagnostic.scan_phase(data, params, {}, 0, 1)
    named = {r['convention']: r for r in report}
    assert named['negative catalogue']['AE_relative_rms'] < 1e-12
    assert named['catalogue']['AE_relative_rms'] > .1
    assert report[-1]['AE_relative_rms'] < 1e-12
    assert len(curve) == 16
    assert params['coa_phase'] == .6
    assert all({k:v for k,v in p.items() if k != 'coa_phase'} ==
               {k:v for k,v in params.items() if k != 'coa_phase'} for p in calls)
    for c in data:
        np.testing.assert_allclose(best[c], data[c])


def test_metrics_do_not_hide_amplitude_errors():
    x = np.arange(1., 9.)
    data = {c:x for c in ['LISA_A', 'LISA_E', 'LISA_T']}
    metrics = diagnostic.phase_metrics(data, {c:2*x for c in data})
    assert metrics['AE_relative_rms'] == pytest.approx(1.)
    assert metrics['AE_overlap'] == pytest.approx(1.)
    assert metrics['AE_amplitude_ratio'] == pytest.approx(2.)
    data['LISA_A'][0] = np.nan
    with pytest.raises(ValueError, match='finite'):
        diagnostic.scan_phase(data, dict(coa_phase=0), {}, 0, 1)


def test_alignment_distinguishes_parameter_time_from_sample_lag():
    x = np.zeros(64)
    x[25:29] = [1., 3., -2., 1.]
    delayed = np.zeros_like(x)
    delayed[28:32] = x[25:29]
    data = {c: delayed for c in ['LISA_A', 'LISA_E', 'LISA_T']}
    model = {c: x for c in data}
    report = diagnostic.alignment_metrics(data, model, 4., max_lag_seconds=40.)
    assert report['best_common_lag_seconds'] == 12.
    assert report['at_best_lag']['AE_relative_rms'] == 0.
    assert report['zero_lag']['AE_relative_rms'] > 0.
    assert report['data_peak_offset_seconds']-report['model_peak_offset_seconds'] == 12.
    reverse = diagnostic.alignment_metrics(model, data, 4., max_lag_seconds=40.)
    assert reverse['best_common_lag_seconds'] == -12.
    assert reverse['at_best_lag']['AE_relative_rms'] == 0.
