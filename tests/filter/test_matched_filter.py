"""Unit tests for the matched_filter module."""

import numpy as np
import pytest
from gwaith.types import TimeSeries, FrequencySeries

from hwilas.filter.matched_filter import (
    _get_cutoff_indices,
    matched_filter_frequencyseries,
    matched_filter_timeseries,
    sigmasq_frequencyseries,
)


@pytest.fixture
def mock_ts():
    """A mock time series class for testing."""

    class MockTimeSeries:
        def __init__(self, data, delta_t, epoch=0.0):
            self.data = np.asarray(data)
            self.delta_t = delta_t
            self.epoch = epoch

        def __len__(self):
            return len(self.data)

        def __array__(self, dtype=None, copy=None):
            if dtype is None:
                return self.data
            return np.asarray(self.data, dtype=dtype)

        def to_frequencyseries(self):
            # Match gwaith's TimeSeries.to_frequencyseries: an rfft on the
            # (zero-padded) time series gives a one-sided spectrum of length
            # n // 2 + 1 with delta_f = 1 / (n * delta_t).
            n = len(self.data)
            df = 1.0 / (n * self.delta_t)
            return FrequencySeries(np.fft.rfft(self.data, n=n), delta_f=df)

    return MockTimeSeries


@pytest.fixture
def mock_fs():
    """A mock frequency series class for testing."""

    class MockFrequencySeries:
        def __init__(self, data, delta_f):
            self.data = np.asarray(data)
            self.delta_f = delta_f

        def __len__(self):
            return len(self.data)

        def __array__(self, dtype=None, copy=None):
            if dtype is None:
                return self.data
            return np.asarray(self.data, dtype=dtype)

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
    # norm = 2 * 0.1 / sqrt(0.4). hwilas' irfft-based correlation is 2x PyCBC's
    # real part, so the normalisation constant is halved (2*df instead of 4*df)
    # to match PyCBC's SNR convention. Hence the peak is sqrt(sigmasq)/2.
    assert np.isclose(np.max(np.abs(snr_ts.data)), np.sqrt(sigmasq) / 2.0)

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


def test_matched_filter_timeseries(mock_ts):
    """Test the time-domain matched filter function."""
    delta_t = 0.1
    n_samples = 2048
    epoch = 1000.0

    # A single-bin spike in the time domain (a unit impulse).
    template_data = np.zeros(n_samples, dtype=np.float64)
    template_data[100] = 1.0
    template = mock_ts(template_data, delta_t=delta_t, epoch=epoch)

    # Data is just the template.
    data = mock_ts(template_data.copy(), delta_t=delta_t, epoch=epoch)

    snr_ts = matched_filter_timeseries(template, data, n_samples=n_samples, epoch=epoch)

    assert isinstance(snr_ts, TimeSeries)
    assert snr_ts.epoch == epoch
    assert len(snr_ts) == n_samples

    # For a unit impulse template with PSD-free normalisation:
    # sigma_sq = (4 / delta_t) * sum(h^2) = (4 / 0.1) * 1 = 40
    # The peak of the cross-correlation is at lag 0 with value sum(h*s) = 1,
    # so the peak SNR should be norm * 1 where norm = (2/delta_t)/sqrt(sigma_sq).
    # hwilas' irfft-based correlation is 2x PyCBC's real part, so the constant
    # is halved (2 instead of 4) to match PyCBC; hence peak = sqrt(sigma_sq)/2.
    sigma_sq = (4.0 / delta_t) * np.sum(template_data ** 2)
    assert np.isclose(np.max(np.abs(snr_ts.data)), np.sqrt(sigma_sq) / 2.0)

    # The peak should be located at lag 0 (index 0), i.e. where the impulse in
    # the template aligns with itself. (The impulse sits at index 100 in both
    # series, so the zero-lag alignment is at index 0 of the correlation.)
    peak_idx = int(np.argmax(np.abs(snr_ts.data)))
    assert peak_idx == 0


def test_matched_filter_timeseries_equivalence(mock_fs, mock_ts):
    """The time-domain and frequency-domain matched filters should agree.

    For a real template and data with a flat PSD, the normalised SNR from both
    implementations should be identical (up to FFT round-off).
    """
    delta_t = 0.1
    n_samples = 2048
    epoch = 1000.0
    flen = n_samples // 2 + 1
    delta_f = 1.0 / (n_samples * delta_t)

    # A smooth real template in the time domain.
    rng = np.random.default_rng(42)
    h_td = rng.standard_normal(n_samples)
    s_td = h_td.copy() + 0.5 * rng.standard_normal(n_samples)

    template_ts = mock_ts(h_td, delta_t=delta_t, epoch=epoch)
    data_ts = mock_ts(s_td, delta_t=delta_t, epoch=epoch)

    # Frequency-domain equivalents with a flat PSD of unity.
    h_fd = np.fft.rfft(h_td, n=n_samples)
    s_fd = np.fft.rfft(s_td, n=n_samples)
    psd = mock_fs(np.ones(flen), delta_f=delta_f)

    snr_td = matched_filter_timeseries(
        template_ts, data_ts, psd=psd, n_samples=n_samples, epoch=epoch
    )
    snr_fd = matched_filter_frequencyseries(
        mock_fs(h_fd, delta_f), mock_fs(s_fd, delta_f),
        psd=psd, n_samples=n_samples, epoch=epoch,
    )

    # The two SNR time series should match closely.
    assert np.allclose(snr_td.data, snr_fd.data, rtol=1e-8, atol=1e-8)


def test_matched_filter_timeseries_equivalence_nonflat_psd(mock_fs, mock_ts):
    """The TD and FD filters must agree for a *non-flat* PSD.

    PSD-weighted normalisation uses the unwhitened-template sigma^2,
    so the time-domain and frequency-domain implementations agree up to
    FFT round-off for frequency-dependent noise.
    """
    delta_t = 0.1
    n_samples = 2048
    epoch = 1000.0
    flen = n_samples // 2 + 1
    delta_f = 1.0 / (n_samples * delta_t)

    rng = np.random.default_rng(7)
    h_td = rng.standard_normal(n_samples)
    s_td = h_td.copy() + 0.5 * rng.standard_normal(n_samples)

    # A strongly non-flat, strictly positive PSD (1/f-ish).
    freqs = delta_f * np.arange(flen)
    psd_vals = (1e-20 / (freqs + 1e-4)) ** 1.5 + 1e-30
    psd = mock_fs(psd_vals, delta_f=delta_f)

    # Pre-whitened time-domain inputs: divide the FD spectra by the PSD and
    # transform back, exactly as a caller would do before matched filtering.
    h_fd = np.fft.rfft(h_td, n=n_samples)
    s_fd = np.fft.rfft(s_td, n=n_samples)
    h_whitened_td = np.fft.irfft(h_fd / psd_vals, n=n_samples)
    s_whitened_td = np.fft.irfft(s_fd / psd_vals, n=n_samples)

    template_ts = mock_ts(h_whitened_td, delta_t=delta_t, epoch=epoch)
    data_ts = mock_ts(s_whitened_td, delta_t=delta_t, epoch=epoch)

    snr_td = matched_filter_timeseries(
        template_ts, data_ts, psd=psd, n_samples=n_samples, epoch=epoch
    )
    snr_fd = matched_filter_frequencyseries(
        mock_fs(h_fd, delta_f), mock_fs(s_fd, delta_f),
        psd=psd, n_samples=n_samples, epoch=epoch,
    )

    # The normalised SNR time series should match closely despite the non-flat PSD.
    assert np.allclose(snr_td.data, snr_fd.data, rtol=1e-8, atol=1e-8)


def test_matched_filter_timeseries_errors(mock_ts):
    """Test the value errors of the time-domain matched filter."""
    delta_t = 0.1
    n_samples = 2048
    epoch = 1000.0

    template_data = np.zeros(n_samples, dtype=np.float64)
    template_data[100] = 1.0
    template = mock_ts(template_data, delta_t=delta_t, epoch=epoch)
    data = mock_ts(template_data.copy(), delta_t=delta_t, epoch=epoch)

    with pytest.raises(ValueError):
        # Mismatched delta_t
        wrong_dt_data = mock_ts(template_data, delta_t=0.2, epoch=epoch)
        matched_filter_timeseries(template, wrong_dt_data, n_samples=n_samples, epoch=epoch)

    with pytest.raises(ValueError):
        # n_samples shorter than the inputs
        matched_filter_timeseries(template, data, n_samples=n_samples // 2, epoch=epoch)

    with pytest.raises(ValueError):
        # Non-positive n_samples
        matched_filter_timeseries(template, data, n_samples=0, epoch=epoch)


def test_matched_filter_timeseries_defaults(mock_ts):
    """Test the default n_samples and epoch handling."""
    delta_t = 0.1
    epoch = 1234.0

    template_data = np.zeros(512, dtype=np.float64)
    template_data[50] = 1.0
    data_data = np.zeros(768, dtype=np.float64)
    data_data[50] = 1.0

    template = mock_ts(template_data, delta_t=delta_t, epoch=epoch)
    data = mock_ts(data_data, delta_t=delta_t, epoch=epoch)

    # Default n_samples is the longer of the two inputs; default epoch is the
    # data epoch.
    snr_ts = matched_filter_timeseries(template, data)
    assert len(snr_ts) == 768
    assert snr_ts.epoch == epoch
