import numpy as np
import pytest

pytest.importorskip("gwaith.types")
pycbc_filter = pytest.importorskip("pycbc.filter")
pytest.importorskip("pycbc.types")

from pycbc.types import FrequencySeries, TimeSeries

from hwilas.filter import matched_filter_frequencyseries, sigmasq_frequencyseries


def _build_frequency_inputs(n_samples=64, delta_t=1.0):
    rng = np.random.default_rng(20260805)

    template_td = TimeSeries(rng.normal(size=n_samples), delta_t=delta_t)
    data_td = TimeSeries(rng.normal(size=n_samples), delta_t=delta_t)

    template_fd = template_td.to_frequencyseries()
    data_fd = data_td.to_frequencyseries()

    psd_vals = 1.0 + np.linspace(0.0, 0.2, len(template_fd))
    psd = FrequencySeries(psd_vals, delta_f=template_fd.delta_f)

    return template_fd, data_fd, psd


def test_sigmasq_matches_pycbc():
    template_fd, _, psd = _build_frequency_inputs()

    expected = pycbc_filter.sigmasq(template_fd, psd=psd)
    actual = sigmasq_frequencyseries(template_fd, psd)

    assert np.isclose(actual, expected, rtol=1e-12, atol=0.0)


def test_sigmasq_matches_pycbc_without_psd():
    template_fd, _, _ = _build_frequency_inputs()

    expected = pycbc_filter.sigmasq(template_fd, psd=None)
    actual = sigmasq_frequencyseries(template_fd, psd=None)

    assert np.isclose(actual, expected, rtol=1e-12, atol=0.0)


def test_matched_filter_matches_pycbc():
    template_fd, data_fd, psd = _build_frequency_inputs()

    expected = pycbc_filter.matched_filter(template_fd, data_fd, psd=psd)
    actual = matched_filter_frequencyseries(
        template_fd,
        data_fd,
        psd=psd,
        n_samples=(len(template_fd) - 1) * 2,
        epoch=float(data_fd.epoch),
    )

    np.testing.assert_allclose(
        np.asarray(actual),
        np.asarray(expected),
        rtol=1e-10,
        atol=1e-12,
    )

    assert np.isclose(actual.delta_t, expected.delta_t)
    assert len(actual) == len(expected)


def test_matched_filter_matches_pycbc_without_psd():
    template_fd, data_fd, _ = _build_frequency_inputs()

    expected = pycbc_filter.matched_filter(template_fd, data_fd, psd=None)
    actual = matched_filter_frequencyseries(
        template_fd,
        data_fd,
        psd=None,
        n_samples=(len(template_fd) - 1) * 2,
        epoch=float(data_fd.epoch),
    )

    np.testing.assert_allclose(
        np.asarray(actual),
        np.asarray(expected),
        rtol=1e-10,
        atol=1e-12,
    )

    assert np.isclose(actual.delta_t, expected.delta_t)
    assert len(actual) == len(expected)
