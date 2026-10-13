import sys

import h5py
import numpy as np
import pytest

from tools.plotting import plot_processed_data as plot
from gwaith.types import TimeSeries


@pytest.mark.parametrize('length', [64, 65])
def test_whitening_preserves_grid_and_scales_tone(tmp_path, length):
    psd = tmp_path / 'psd.txt'
    np.savetxt(psd, [[0., 4.], [0.5, 4.]])
    tone = np.sin(2 * np.pi * 5 * np.arange(length) / length)
    ts = TimeSeries(tone, epoch=123., delta_t=1.)
    for power, scale in [(.5, 2.), (1., 4.)]:
        result = plot.whiten_timeseries(ts, psd, power)
        np.testing.assert_allclose(result.data, tone / scale, atol=1e-14)
        np.testing.assert_array_equal(result.sample_times, ts.sample_times)


@pytest.mark.parametrize('edge_exclusion', [0., 4.2])
def test_cli_writes_all_plots(tmp_path, monkeypatch, edge_exclusion):
    data = tmp_path / 'data.hdf'
    with h5py.File(data, 'w') as f:
        f.attrs.update(epoch=123., delta_t=1.)
        f['LISA_A'] = np.sin(2 * np.pi * np.arange(65) / 13)
    psd = tmp_path / 'psd.txt'
    np.savetxt(psd, [[0., 4.], [0.5, 4.]])
    outputs = [tmp_path / 'raw.png', tmp_path / 'white/w.png', tmp_path / 'over/o.png']
    plotted = {}
    times = {}
    original_plot_full = plot.plot_full

    def capture_plot(series, channels, path, **kwargs):
        plotted[str(path)] = series['LISA_A'].data.copy()
        times[str(path)] = series['LISA_A'].sample_times.copy()
        if str(path) != str(outputs[0]):
            fig, ax = plot.make_full_plot(series, channels, **kwargs)
            np.testing.assert_allclose(ax.lines[0].get_xdata(),
                                       (times[str(path)] - 123.) / 86400.)
            plot.plt.close(fig)
        return original_plot_full(series, channels, path, **kwargs)

    monkeypatch.setattr(plot, 'plot_full', capture_plot)
    monkeypatch.setattr(sys, 'argv', ['plot', '--data-file', str(data),
        '--channels', 'LISA_A', '--psd-files', f'LISA_A:{psd}',
        '--output-plot', str(outputs[0]), '--output_whitened_plot', str(outputs[1]),
        '--output-overwhitened-plot', str(outputs[2]), '--decimate', '3',
        '--edge-exclusion', str(edge_exclusion)])
    plot.main()
    assert all(path.read_bytes().startswith(b'\x89PNG') for path in outputs)
    edge = int(np.ceil(edge_exclusion))
    tone = np.sin(2 * np.pi * np.arange(65) / 13)
    for output, scale in zip(outputs[1:], [2., 4.]):
        np.testing.assert_allclose(plotted[str(output)], tone[edge:65-edge:3] / scale, atol=1e-14)
        np.testing.assert_array_equal(times[str(output)], 123. + np.arange(edge, 65-edge, 3))
    np.testing.assert_allclose(plotted[str(outputs[0])], tone[::3])


@pytest.mark.parametrize('psd_args', [[], ['--psd-file', 'unused.txt']])
def test_raw_plot_skips_whitening(tmp_path, monkeypatch, psd_args):
    data = tmp_path / 'data.hdf'
    with h5py.File(data, 'w') as f:
        f.attrs.update(epoch=0., delta_t=1.)
        f['LISA_A'] = np.arange(20.)

    def unexpected_whitening(*args, **kwargs):
        pytest.fail('Raw plotting must not load or apply a PSD')

    monkeypatch.setattr(plot, 'whiten_timeseries', unexpected_whitening)
    monkeypatch.setattr(sys, 'argv', ['plot', '--data-file', str(data),
        '--channels', 'LISA_A', '--output-directory', str(tmp_path / 'plots'), *psd_args])
    plot.main()
    assert sorted(p.name for p in (tmp_path / 'plots').iterdir()) == ['full.png']


@pytest.mark.parametrize('output_option', ['--output-whitened-plot', '--output-overwhitened-plot'])
@pytest.mark.parametrize('psd_args, message', [
    ([], 'require --psd-file or --psd-files'),
    (['--psd-files', 'LISA_E:missing.txt'], 'must supply every selected channel'),
    (['--psd-file', 'missing.txt'], 'PSD file does not exist'),
])
def test_invalid_psd_rejected_before_data_loading(monkeypatch, capsys, output_option, psd_args, message):
    def unexpected_loading(*args, **kwargs):
        pytest.fail('PSD validation must run before loading the data')

    monkeypatch.setattr(plot, 'load_timeseries', unexpected_loading)
    monkeypatch.setattr(sys, 'argv', ['plot', '--data-file', 'unused.hdf',
        '--channels', 'LISA_A', '--output-plot', 'raw.png', output_option, 'filtered.png', *psd_args])
    with pytest.raises(SystemExit) as error:
        plot.main()
    assert error.value.code == 2
    assert message in capsys.readouterr().err


def test_edge_exclusion_rejects_empty_filtered_plot(tmp_path, monkeypatch):
    data = tmp_path / 'data.hdf'
    with h5py.File(data, 'w') as f:
        f.attrs.update(epoch=0., delta_t=1.)
        f['LISA_A'] = np.arange(20.)
    psd = tmp_path / 'psd.txt'
    np.savetxt(psd, [[0., 4.], [0.5, 4.]])
    monkeypatch.setattr(sys, 'argv', ['plot', '--data-file', str(data),
        '--channels', 'LISA_A', '--output-plot', str(tmp_path / 'raw.png'),
        '--output-whitened-plot', str(tmp_path / 'white.png'), '--psd-file', str(psd)])
    with pytest.raises(ValueError, match='Edge exclusion removes all'):
        plot.main()
    assert not (tmp_path / 'white.png').exists()
