"""Residual candidate ranking away from fitted merger times."""
import json
import sys

import h5py
import numpy as np
import pytest

from tools.search import check_search_residuals as checker


@pytest.mark.parametrize('window,mergers,expected',[
    (86400., [200000., 500000.], [10., 10., 20.]),
    (86400., [0., 200000., 500000.], [20.]),
    (10., [200000., 500000.], [10., 10., 20., 80., 90.]),
    (1e6, [200000., 500000.], []),
    (86400., [], [10., 10., 20., 80., 90., 100., 110.]),
])
def test_residual_ranking_and_cli_plot(tmp_path, monkeypatch, window, mergers, expected):
    data, signal, residual, parameters, candidates = [tmp_path/f'{name}.hdf'
        for name in ('data', 'signal', 'residual', 'parameters', 'candidates')]
    channels = ['LISA_A', 'LISA_E']
    for path, value in [(data, 3.), (signal, 2.), (residual, 1.)]:
        with h5py.File(path, 'w') as f:
            f.attrs.update(epoch=0., delta_t=50000.)
            for channel in channels:
                f[channel] = np.full(16, value)
    with h5py.File(parameters, 'w') as f:
        f['tc'] = mergers
    with h5py.File(candidates, 'w') as f:
        f.attrs['analysis_intervals'] = [[0., 800000.], [0., 800000.]]
        f['statistic'] = [10., 10., 20., 80., 90., 100., 110.]
        f['template_id'] = np.arange(7)
        for channel, offset in zip(channels, [-2., 2.]):
            f[f'time_{channel}'] = np.array([0., 1., 700000., 113600., 286400., 200000., 500000.])+offset
    output, plot = tmp_path/'report.json', tmp_path/'plots'/'ranking.png'
    monkeypatch.setattr(sys, 'argv', ['check_residuals', '--data-file', str(data),
        '--residual-file', str(residual), '--signal-files', str(signal),
        '--parameter-file', str(parameters), '--candidate-file', str(candidates),
        '--channels', *channels, '--signal-exclusion', str(window),
        '--output-file', str(output), '--output-statistic-plot', str(plot)])
    checker.main()
    report = json.loads(output.read_text())
    assert report['subtraction_consistent']
    assert report['maximum_residual_statistic_outside_signal_windows'] == (max(expected) if expected else None)
    assert report['residual_coincidence_count_outside_signal_windows'] == len(expected)
    survival = report['residual_statistic_survival_outside_signal_windows']
    assert survival['statistic'] == sorted(set(expected))
    assert survival['count_at_or_above'] == [sum(v >= t for v in expected) for t in sorted(set(expected))]
    exposure = report['residual_search_exposure_seconds_outside_signal_windows']
    assert exposure == checker.remaining_exposure([[0., 800000.]], mergers, window)
    assert report['residual_search_exposure_source'] == 'candidate analysis_intervals'
    if exposure > 0:
        np.testing.assert_allclose(survival['rate_hz_at_or_above'],
            np.array(survival['count_at_or_above'])/exposure)
    assert plot.read_bytes().startswith(b'\x89PNG')


@pytest.mark.parametrize('window', [-1., float('nan'), float('inf')])
def test_invalid_signal_window(window):
    with pytest.raises(ValueError, match='Signal exclusion'):
        checker.check_residuals(None, None, [], [], signal_exclusion=window)



def test_exposure_unions_intervals_and_clips_overlapping_windows():
    assert checker.remaining_exposure([[0., 100.], [0., 100.], [80., 150.], [200., 220.]],
                                      [5., 15., 145., 300.], 10.) == 130.
    assert checker.remaining_exposure([[0., 100.]], [50.], 100.) == 0.
    assert checker.remaining_exposure([[0., 100.]], [], 10.) == 100.
