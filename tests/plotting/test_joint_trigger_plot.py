"""Merged network plots highlight clustered candidates on a common colour scale."""
import h5py
import numpy as np
from matplotlib import pyplot as plt
from tools.plotting import plot_trigger_merge as module


def test_clustered_overlay_and_truth():
    class Truth:
        source_group = 'MBHB'
        def coalescence_times(self):
            return np.array([100.])
    ids = np.array([1, 4, 8])
    times = np.array([10., 100., 200.])
    snrs = np.array([5., 20., 8.])
    clustered = (ids[1:], times[1:], snrs[1:])
    fig = module.make_plot(ids, times, snrs, 'LISA_A+LISA_E', truth=Truth(),
                           clustered=clustered, network_statistic=True)
    ax = fig.axes[0]
    np.testing.assert_allclose(ax.collections[0].get_offsets()[:, 0], times/86400.)
    np.testing.assert_allclose(ax.collections[1].get_offsets()[:, 1], [20., 8.])
    assert ax.collections[0].norm is ax.collections[1].norm
    assert ax.get_ylabel() == 'Network statistic'
    assert ax.lines[0].get_zorder() < ax.collections[0].get_zorder()
    labels = [text.get_text() for text in ax.get_legend().get_texts()]
    assert 'Clustered candidates' in labels and 'Truth (MBHB)' in labels
    plt.close(fig)


def test_empty_network_plot():
    empty = np.array([])
    fig = module.make_plot(empty, empty, empty, None, clustered=(empty, empty, empty), network_statistic=True)
    assert fig.axes[0].get_ylabel() == 'Network statistic'
    plt.close(fig)


def test_cli_saves_single_combined_plot(tmp_path, monkeypatch):
    raw, clustered, output = [tmp_path/name for name in ('raw.hdf', 'clustered.hdf', 'network.png')]
    for path, indices in ((raw, [0, 1, 2]), (clustered, [1, 2])):
        with h5py.File(path, 'w') as f:
            for key, data in dict(template_id=[1, 4, 8], end_time=[10., 100., 200.], snr=[5., 20., 8.]).items():
                f[key] = np.array(data)[indices]
    monkeypatch.setattr('sys.argv', ['plot', '--trigger-file', str(raw),
        '--clustered-trigger-file', str(clustered), '--network-snr', '--output', str(output)])
    module.main()
    assert output.is_file() and output.stat().st_size > 1000
