"""Ranking adapters prefer canonical scores and retain raw complex measurements."""
import h5py
import numpy as np
import pytest
from hwilas.search.statistics import canonical_statistics, read_statistic


def test_canonical_fields_win_and_raw_measurements_survive():
    rows = dict(snr=np.array([5.]), statistic=np.array([20.]),
                snr_A=np.array([3.]), complex_snr_A=np.array([3j]))
    result = canonical_statistics(rows)
    assert 'snr' not in result and 'snr_A' not in result
    assert result['statistic'][0] == 20.
    assert result['statistic_A'][0] == 3.
    assert result['complex_snr_A'][0] == 3j
    assert 'snr' in rows


def test_stored_statistic_takes_precedence_over_quadrature(tmp_path):
    with h5py.File(tmp_path/'scores.hdf', 'w') as f:
        f['statistic'] = [100.]
        f['snr'] = [5.]
        f['statistic_A'] = [3.]
        f['statistic_E'] = [4.]
        assert read_statistic(f)[0] == 100.
        assert read_statistic(f, 'A')[0] == 3.


def test_legacy_component_fallback_and_missing_error(tmp_path):
    with h5py.File(tmp_path/'legacy.hdf', 'w') as f:
        f['snr_A'] = [3.]
        f['snr_E'] = [4.]
        assert read_statistic(f)[0] == 5.
        with pytest.raises(KeyError):
            read_statistic(f, 'T')
