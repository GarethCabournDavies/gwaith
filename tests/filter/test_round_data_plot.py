from types import SimpleNamespace
import h5py
import numpy as np
from matplotlib import pyplot as plt
from tools.plotting.plot_processed_data import load_timeseries, make_full_plot


def test_round_plot_cadence_and_background_markers(tmp_path):
    path = tmp_path/'data.hdf'
    with h5py.File(path,'w') as f:
        f.attrs.update(epoch=100.,delta_t=4.)
        f['LISA_A'] = np.arange(8.)
    ts = load_timeseries(path,'LISA_A',decimate=2)
    np.testing.assert_array_equal(ts.sample_times,[100.,108.,116.,124.])
    truth = SimpleNamespace(source_group='mbhb',coalescence_times=lambda:np.array([124.]))
    fig,ax = make_full_plot({'LISA_A':ts},['LISA_A'],truth=truth,removed_times=[116.])
    data,truth_line,removed_line = ax.lines
    np.testing.assert_allclose(data.get_xdata(),np.array([0.,8.,16.,24.])/86400.)
    np.testing.assert_allclose(truth_line.get_xdata(),[24/86400.]*2)
    np.testing.assert_allclose(removed_line.get_xdata(),[16/86400.]*2)
    for line in [truth_line,removed_line]:
        np.testing.assert_array_equal(line.get_ydata(),[0,1])
        assert line.get_zorder() < data.get_zorder()
    assert truth_line.get_linestyle() == '--'
    assert removed_line.get_linestyle() == ':'
    assert {'LISA_A','Truth (mbhb)','Removed signals'} == set(ax.get_legend_handles_labels()[1])
    plt.close(fig)
