import numpy as np
import pytest

pycbc = pytest.importorskip("pycbc")
pycbc_psd = pytest.importorskip("pycbc.psd")

from hwilas.psd import from_txt as hwilas_from_txt
from hwilas.psd import interpolate as hwilas_interpolate


def test_from_txt_matches_pycbc_for_psd_input(tmp_path):
    freq_data = np.array([1.0e-4, 2.0e-4, 4.0e-4, 8.0e-4, 1.6e-3])
    psd_data = np.array([9.0e-42, 5.0e-42, 3.0e-42, 2.0e-42, 1.5e-42])
    table = np.column_stack([freq_data, psd_data])

    psd_file = tmp_path / "psd.txt"
    np.savetxt(psd_file, table)

    length = 9
    delta_f = 1.0e-4
    low_freq_cutoff = 1.0e-4

    expected = pycbc_psd.from_txt(
        str(psd_file),
        length,
        delta_f,
        low_freq_cutoff,
        is_asd_file=False,
    )
    actual = hwilas_from_txt(
        str(psd_file),
        length,
        delta_f,
        low_freq_cutoff,
        is_asd_file=False,
    )

    assert actual.delta_f == expected.delta_f
    np.testing.assert_allclose(np.asarray(actual), np.asarray(expected), rtol=1e-12, atol=0.0)


def test_from_txt_matches_pycbc_for_asd_input(tmp_path):
    freq_data = np.array([1.0e-4, 2.0e-4, 4.0e-4, 8.0e-4, 1.6e-3])
    asd_data = np.array([3.0e-21, 2.2e-21, 1.7e-21, 1.4e-21, 1.2e-21])
    table = np.column_stack([freq_data, asd_data])

    psd_file = tmp_path / "asd.txt"
    np.savetxt(psd_file, table)

    length = 9
    delta_f = 1.0e-4
    low_freq_cutoff = 1.0e-4

    expected = pycbc_psd.from_txt(
        str(psd_file),
        length,
        delta_f,
        low_freq_cutoff,
        is_asd_file=True,
    )
    actual = hwilas_from_txt(
        str(psd_file),
        length,
        delta_f,
        low_freq_cutoff,
        is_asd_file=True,
    )

    assert actual.delta_f == expected.delta_f
    np.testing.assert_allclose(np.asarray(actual), np.asarray(expected), rtol=1e-12, atol=0.0)


def test_interpolate_matches_pycbc():
    from pycbc.types import FrequencySeries

    base = FrequencySeries(
        np.array([1.0e-42, 2.0e-42, 3.5e-42, 6.0e-42, 1.0e-41], dtype=np.float64),
        delta_f=0.25,
    )

    expected = pycbc_psd.interpolate(base, delta_f=0.1)
    actual = hwilas_interpolate(base, delta_f=0.1)

    assert actual.delta_f == expected.delta_f
    assert len(actual) == len(expected)
    np.testing.assert_allclose(np.asarray(actual), np.asarray(expected), rtol=1e-12, atol=0.0)
