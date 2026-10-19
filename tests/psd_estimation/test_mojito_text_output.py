"""The estimated PSD must be readable by the search's text PSD reader."""
import h5py
import numpy as np
from hwilas.psd.read import from_txt
from tools.psd_estimation.estimate_psd import main


def test_estimated_psd_roundtrip(tmp_path, monkeypatch):
    data = tmp_path/'data.hdf'
    output = tmp_path/'psd.txt'
    unsmoothed = tmp_path/'raw.txt'
    with h5py.File(data, 'w') as f:
        f.attrs.update(delta_t=4., epoch=0.)
        f['LISA_A'] = np.random.default_rng(9).normal(size=2048)
    monkeypatch.setattr('sys.argv', ['estimate_psd', '--data-file', str(data),
                        '--output-file', f'LISA_A:{output}', '--unsmoothed-output-file',
                        f'LISA_A:{unsmoothed}', '--segment-duration', '256',
                        '--smooth-regions', '.03,.07'])
    main()
    raw_welch = np.loadtxt(unsmoothed)
    from gwaith.types import TimeSeries
    with h5py.File(data) as f:
        expected = TimeSeries(f["LISA_A"][:], epoch=0., delta_t=4.).psd(256)
    from hwilas.psd.interpolate import interpolate
    expected = interpolate(expected, 1 / (2048 * 4.))
    np.testing.assert_allclose(raw_welch[:,1], expected.data)
    assert not h5py.is_hdf5(output)
    raw = np.loadtxt(output)
    np.testing.assert_array_equal(raw[:, 0], raw_welch[:, 0])
    assert set(tmp_path.iterdir()) == {data, output, unsmoothed}
    assert np.any(raw[:, 1] != raw_welch[:, 1])
    assert raw.shape[1] == 2 and np.all(np.diff(raw[:, 0]) > 0)
    psd = from_txt(output, 128, 1/2048, 1/2048, is_asd_file=False)
    assert np.all(np.isfinite(psd.data[1:])) and np.all(psd.data[1:] > 0)


def test_multiple_processed_files_and_missing_output_directory(tmp_path, monkeypatch):
    import pytest
    from gwaith.types import TimeSeries
    from hwilas.psd.interpolate import interpolate

    chunks = np.random.default_rng(12).normal(size=(2, 1024))
    inputs = []
    for i, chunk in enumerate(chunks):
        path = tmp_path / f'chunk_{i}.hdf'
        with h5py.File(path, 'w') as f:
            f.attrs.update(delta_t=4., epoch=i * 4096.)
            f['LISA_A'] = chunk
        inputs.append(str(path))
    output = tmp_path / 'psd.txt'
    argv = ['estimate_psd', '--data-file', *inputs, '--output-file',
            f'LISA_A:{output}', '--segment-duration', '256']
    monkeypatch.setattr('sys.argv', argv)
    main()
    expected = interpolate(TimeSeries(chunks.ravel(), delta_t=4., epoch=4096.).psd(256),
                           1 / (2048 * 4.))
    np.testing.assert_allclose(np.loadtxt(output)[:, 1], expected.data)

    missing_output = tmp_path / 'missing' / 'psd.txt'
    monkeypatch.setattr('sys.argv', [*argv[:], '--output-file', f'LISA_A:{missing_output}'])
    with pytest.raises(FileNotFoundError):
        main()
    assert not missing_output.parent.exists()


def test_independent_unsmoothed_channels(tmp_path, monkeypatch):
    from gwaith.types import TimeSeries
    from hwilas.psd.interpolate import interpolate

    data = tmp_path / 'data.hdf'
    output = tmp_path / 'A_psd.txt'
    unsmoothed = tmp_path / 'T_psd.txt'
    samples = np.random.default_rng(23).normal(size=2048)
    with h5py.File(data, 'w') as f:
        f.attrs.update(delta_t=4., epoch=0.)
        f['LISA_A'] = samples
        f['LISA_T'] = samples
    base = ['estimate_psd', '--data-file', str(data), '--segment-duration', '256',
            '--smooth-regions', '.03,.07', '--unsmoothed-output-file',
            f'LISA_T:{unsmoothed}']
    monkeypatch.setattr('sys.argv', [*base, '--output-file', f'LISA_A:{output}'])
    main()
    expected = interpolate(TimeSeries(samples, delta_t=4., epoch=0.).psd(256),
                           1 / (2048 * 4.))
    raw = np.loadtxt(unsmoothed)
    smoothed = np.loadtxt(output)
    np.testing.assert_allclose(raw[:, 1], expected.data)
    np.testing.assert_array_equal(raw[:, 0], smoothed[:, 0])
    assert np.any(raw[:, 1] != smoothed[:, 1])

    # An unsmoothed-only invocation needs no smoothed output filenames.
    monkeypatch.setattr('sys.argv', base)
    main()
    np.testing.assert_array_equal(np.loadtxt(unsmoothed), raw)


def test_requires_an_output(tmp_path, monkeypatch):
    import pytest
    monkeypatch.setattr('sys.argv', ['estimate_psd', '--data-file', str(tmp_path / 'data.hdf')])
    with pytest.raises(SystemExit) as exc:
        main()
    assert exc.value.code == 2
