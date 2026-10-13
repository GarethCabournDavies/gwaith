"""Same-time A/E statistics retain channel information and feed event optimisation."""
from types import SimpleNamespace
import h5py
import numpy as np
from gwaith.types import TimeSeries
from hwilas.template_banks.io import write_bank
from tools.search import matched_filter_search as search
from tools.search import trigger_merge
from tools.search.optimise_snr_peaks import get_events_to_optimise


def test_joint_filter_combines_before_peak_selection(tmp_path, monkeypatch):
    dt, n = 4., 345600
    data, bank, a_psd, e_psd, output = [tmp_path/name for name in
        ('data.hdf', 'bank.hdf', 'a.txt', 'e.txt', 'joint.hdf')]
    with h5py.File(data, 'w') as f:
        f.attrs.update(epoch=1000., delta_t=dt, divide_by_laser_frequency=True)
        for c in ('LISA_A', 'LISA_E'):
            f[c] = np.zeros(n)
    params = dict(mass1=1e6, mass2=5e5, spin1z=0., spin2z=0., eclipticlatitude=0.,
        eclipticlongitude=0., inclination=1., polarization=.2, coa_phase=0., distance=1e4)
    write_bank(bank, [params], dict(f_lower=3e-5, f_upper=.09, pre_merger_days=1.))
    freq = np.linspace(0., .125, 1000)
    for path, value in ((a_psd, 8.), (e_psd, 32.)):
        np.savetxt(path, np.column_stack([freq, np.full(len(freq), value)]))
    monkeypatch.setattr(search, 'duration_from_frequency', lambda *a: 86400.)
    calls = []
    def waveform(params, settings, modes):
        calls.append(params)
        length, _ = search.template_geometry(params, settings)
        return {c: np.zeros(length) for c in ('LISA_A', 'LISA_E')}
    monkeypatch.setattr(search, 'short_template', waveform)
    def matched_filter(wave, data_fs, psd, n_samples, epoch, **kwargs):
        z = np.zeros(n_samples, complex)
        if np.isclose(psd.data[10], 8.):
            z[round(4*86400/dt)] = 10.
            z[round(8*86400/dt)] = 7.
            z[round(10*86400/dt)] = 7.
        else:
            z[round(4*86400/dt)] = 20j
            z[round(8*86400/dt)] = 7j
            z[round(11*86400/dt)] = 7j
        z[round(1.75*86400/dt)] = 1e6
        return TimeSeries(z, epoch=epoch, delta_t=dt)
    monkeypatch.setattr(search, 'matched_filter_frequencyseries', matched_filter)
    monkeypatch.setattr('sys.argv', ['filter', '--input-data-file', str(data), '--bank-file', str(bank),
        '--channels', 'LISA_A', 'LISA_E', '--psd-files', f'LISA_A:{a_psd}', f'LISA_E:{e_psd}',
        '--snr-threshold', '8', '--output-file', str(output)])
    search.main()
    assert len(calls) == 1
    with h5py.File(output) as f:
        np.testing.assert_allclose(f['end_time'][:], 1000+np.array([4., 8.])*86400)
        np.testing.assert_allclose(f['statistic'][:], [np.hypot(10, 20), np.hypot(7, 7)])
        np.testing.assert_array_equal(f['complex_snr_LISA_E'][:], [20j, 7j])
        np.testing.assert_array_equal(f['time_LISA_A'][:], f['time_LISA_E'][:])
    merged, raw = tmp_path/'merged.hdf', tmp_path/'raw.hdf'
    monkeypatch.setattr('sys.argv', ['merge', '--trigger-files', str(output), '--output-file', str(merged),
        '--unclustered-output-file', str(raw), '--cluster-across-templates', '--cluster-window', '86400'])
    trigger_merge.main()
    args = SimpleNamespace(input_file=merged, optimisation_channels=['LISA_A', 'LISA_E'],
                           statistic_threshold=0., minimum_peak_separation=86400.)
    stats, times, _, _ = get_events_to_optimise(args)
    np.testing.assert_allclose(stats, [np.hypot(10, 20), np.hypot(7, 7)])
    np.testing.assert_allclose(times, 1000+np.array([4., 8.])*86400)
    with h5py.File(merged) as f:
        np.testing.assert_array_equal(f['complex_snr_LISA_E'][:], [20j, 7j])


def test_network_clustering_across_templates_preserves_columns():
    rows = dict(template_id=np.array([0, 1, 2]), end_time=np.array([0., 100., 200000.]),
        snr=np.array([10., 20., 8.]), complex_snr_LISA_A=np.array([1j, 2j, 3j]))
    result = trigger_merge.cluster_rows(rows, 86400., across_templates=True)
    np.testing.assert_array_equal(result['template_id'], [1, 2])
    np.testing.assert_array_equal(result['complex_snr_LISA_A'], [2j, 3j])
    assert len(trigger_merge.cluster_rows(rows, 86400.)['statistic']) == 3
