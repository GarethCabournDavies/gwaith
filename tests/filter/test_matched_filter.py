"""Unit tests for the matched_filter module."""

import numpy as np
import pytest
from gwaith.types import TimeSeries, FrequencySeries

from hwilas.filter.matched_filter import (
    _get_cutoff_indices,
    matched_filter_frequencyseries,
    sigmasq_frequencyseries,
)


@pytest.fixture
def mock_fs():
    """A mock frequency series class for testing."""

    class MockFrequencySeries:
        def __init__(self, data, delta_f):
            self.data = np.asarray(data)
            self.delta_f = delta_f

        def __len__(self):
            return len(self.data)

        def __array__(self):
            return self.data

    return MockFrequencySeries


def test_get_cutoff_indices():
    """Test the frequency cutoff index calculation."""
    # Test with flow and fhigh
    kmin, kmax = _get_cutoff_indices(10, 100, 1.0, 256)
    assert kmin == 10
    assert kmax == 100

    # Test without flow
    kmin, kmax = _get_cutoff_indices(None, 100, 1.0, 256)
    assert kmin == 1
    assert kmax == 100

    # Test without fhigh
    kmin, kmax = _get_cutoff_indices(10, None, 1.0, 256)
    assert kmin == 10
    assert kmax == 128

    # Test with invalid flow
    with pytest.raises(ValueError):
        _get_cutoff_indices(-1, 100, 1.0, 256)

    # Test with kmax <= kmin
    with pytest.raises(ValueError):
        _get_cutoff_indices(100, 10, 1.0, 256)


def test_sigmasq_frequencyseries(mock_fs):
    """Test the sigma^2 calculation."""
    delta_f = 0.1
    n_samples = 2048
    flen = n_samples // 2 + 1
    template = mock_fs(np.ones(flen, dtype=np.complex128), delta_f=delta_f)
    psd = mock_fs(np.ones(flen) * 2, delta_f=delta_f)

    # With PSD
    sigmasq = sigmasq_frequencyseries(template, psd)
    # integral of |H|^2/S_h = 1/2 from 0 to f_nyquist
    # 4 * delta_f * sum(|H_k|^2 / S_k)
    # 4 * 0.1 * sum(1^2 / 2) from k=1 to k=1024
    # 4 * 0.1 * (1024 * 0.5) = 0.4 * 512 = 204.8
    # kmin is 1, kmax is 1024. So we sum 1023 times.
    # 0.4 * 1023 * 0.5 = 204.6
    assert np.isclose(sigmasq, 204.6)


    # Without PSD
    sigmasq_no_psd = sigmasq_frequencyseries(template)
    # 4 * 0.1 * sum(1^2) = 0.4 * 1023 = 409.2
    assert np.isclose(sigmasq_no_psd, 409.2)

    # With cutoffs
    sigmasq_cut = sigmasq_frequencyseries(
        template, psd, low_frequency_cutoff=10, high_frequency_cutoff=20
    )
    # kmin=100, kmax=200. Sum 100 times.
    # 0.4 * 100 * 0.5 = 20.0
    assert np.isclose(sigmasq_cut, 20.0)


def test_matched_filter_frequencyseries(mock_fs):
    """Test the matched filter function."""
    delta_f = 0.1
    n_samples = 2048
    epoch = 1000.0
    flen = n_samples // 2 + 1

    # Simple signal: a single-bin spike
    template_data = np.zeros(flen, dtype=np.complex128)
    template_data[100] = 1.0
    template = mock_fs(template_data, delta_f)

    # Data is just the template
    data = mock_fs(template_data, delta_f)
    psd = mock_fs(np.ones(flen), delta_f)

    snr_ts = matched_filter_frequencyseries(
        template, data, psd=psd, n_samples=n_samples, epoch=epoch
    )

    assert isinstance(snr_ts, TimeSeries)
    assert snr_ts.epoch == epoch
    assert len(snr_ts) == n_samples

    # Check that SNR has a peak
    # For this simple case, the peak should be related to sigmasq
    sigmasq = sigmasq_frequencyseries(template, psd)
    # sigmasq = 4 * 0.1 * (1^2/1) = 0.4
    # norm = 4 * 0.1 / sqrt(0.4) = 0.4 / 0.632...
    # Peak should be close to sqrt(sigmasq)
    assert np.isclose(np.max(np.abs(snr_ts.data)), np.sqrt(sigmasq))

    # Test value errors
    with pytest.raises(ValueError):
        # Mismatched delta_f
        wrong_df_data = mock_fs(template_data, delta_f=0.2)
        matched_filter_frequencyseries(
            template, wrong_df_data, psd=psd, n_samples=n_samples, epoch=epoch
        )
    with pytest.raises(ValueError):
        # Mismatched length
        wrong_len_data = mock_fs(template_data[:-1], delta_f)
        matched_filter_frequencyseries(
            template, wrong_len_data, psd=psd, n_samples=n_samples, epoch=epoch
        )
