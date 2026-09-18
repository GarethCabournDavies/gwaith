"""Utilities for matched-filter operations used across search scripts."""

from __future__ import annotations

from typing import Any

import numpy as np
from numpy.typing import NDArray
from gwaith.types import TimeSeries


def _get_cutoff_indices(
    flow: float | None,
    fhigh: float | None,
    delta_f: float,
    n_samples: int,
) -> tuple[int, int]:
    """Replicate PyCBC cutoff-bin selection semantics."""
    if flow:
        kmin = int(flow / delta_f)
        if kmin < 0:
            raise ValueError(f"Start frequency cannot be negative: {flow}")
    else:
        kmin = 1

    if fhigh:
        kmax = int(fhigh / delta_f)
        if kmax > int((n_samples + 1) / 2.0):
            kmax = int((n_samples + 1) / 2.0)
    else:
        kmax = int((n_samples + 1) / 2.0)

    if kmax <= kmin:
        raise ValueError(
            f"kmax must be > kmin, got {kmax} <= {kmin} for flow={flow}, fhigh={fhigh}"
        )

    return kmin, kmax


def sigmasq_frequencyseries(
    template_fs: Any,
    psd: Any | None = None,
    low_frequency_cutoff: float | None = None,
    high_frequency_cutoff: float | None = None,
) -> float:
    """Compute PyCBC-equivalent sigma^2 for a frequency-domain template."""
    if psd is not None and len(template_fs) != len(psd):
        raise ValueError(
            "Template and PSD lengths differ: "
            f"{len(template_fs)} vs {len(psd)}"
        )

    delta_f = float(template_fs.delta_f)
    if psd is not None and not np.isclose(delta_f, float(psd.delta_f)):
        raise ValueError(
            "Template and PSD delta_f differ: "
            f"{template_fs.delta_f} vs {psd.delta_f}"
        )

    n_samples = (len(template_fs) - 1) * 2
    kmin, kmax = _get_cutoff_indices(
        low_frequency_cutoff,
        high_frequency_cutoff,
        delta_f,
        n_samples,
    )

    template_data: NDArray[np.complex128] = np.asarray(template_fs, dtype=np.complex128)[kmin:kmax]
    if psd is None:
        return float(4.0 * delta_f * np.sum(np.abs(template_data) ** 2))

    psd_data: NDArray[np.float64] = np.asarray(psd, dtype=np.float64)[kmin:kmax]
    valid: NDArray[np.bool_] = np.isfinite(psd_data) & (psd_data > 0)
    return float(4.0 * delta_f * np.sum((np.abs(template_data[valid]) ** 2) / psd_data[valid]))


def matched_filter_frequencyseries(
    template_fs: Any,
    data_fs: Any,
    *,
    psd: Any | None = None,
    n_samples: int,
    epoch: float,
    low_frequency_cutoff: float | None = None,
    high_frequency_cutoff: float | None = None,
) -> TimeSeries:
    """Compute a normalized matched-filter SNR time series from frequency series.

    Parameters
    ----------
    template_fs : FrequencySeries
        Frequency-domain template waveform.
    data_fs : FrequencySeries
        Frequency-domain detector data.
    psd : FrequencySeries, optional
        One-sided noise PSD. If ``None``, whitening is skipped.
    n_samples : int
        Number of output time-domain samples.
    epoch : float
        Epoch to attach to the output SNR time series.
    """
    if n_samples <= 0:
        raise ValueError(f"n_samples must be positive, got {n_samples}")

    if template_fs.delta_f != data_fs.delta_f:
        raise ValueError(
            "template/data delta_f mismatch: "
            f"{template_fs.delta_f} / {data_fs.delta_f}"
        )
    if psd is not None and template_fs.delta_f != psd.delta_f:
        raise ValueError(
            "template/data/psd delta_f mismatch: "
            f"{template_fs.delta_f} / {data_fs.delta_f} / {psd.delta_f}"
        )

    flen_expected = n_samples // 2 + 1
    if len(template_fs) != flen_expected or len(data_fs) != flen_expected:
        raise ValueError(
            "template/data length mismatch: "
            f"expected {flen_expected}, got {len(template_fs)} / "
            f"{len(data_fs)}"
        )
    if psd is not None and len(psd) != flen_expected:
        raise ValueError(
            "template/data/psd length mismatch: "
            f"expected {flen_expected}, got {len(template_fs)} / "
            f"{len(data_fs)} / {len(psd)}"
        )

    delta_f = float(template_fs.delta_f)
    kmin, kmax = _get_cutoff_indices(
        low_frequency_cutoff,
        high_frequency_cutoff,
        delta_f,
        n_samples,
    )

    htilde: NDArray[np.complex128] = np.asarray(template_fs.data, dtype=np.complex128)
    stilde: NDArray[np.complex128] = np.asarray(data_fs.data, dtype=np.complex128)
    psd_data: NDArray[np.float64] | None = np.asarray(psd.data, dtype=np.float64) if psd is not None else None

    qtilde: NDArray[np.complex128] = np.zeros(flen_expected, dtype=np.complex128)
    qtilde_slice: NDArray[np.complex128] = np.conjugate(htilde[kmin:kmax]) * stilde[kmin:kmax]
    if psd_data is not None:
        valid: NDArray[np.bool_] = np.isfinite(psd_data[kmin:kmax]) & (psd_data[kmin:kmax] > 0)
        whitened: NDArray[np.complex128] = np.zeros(kmax - kmin, dtype=np.complex128)
        whitened[valid] = qtilde_slice[valid] / psd_data[kmin:kmax][valid]
        qtilde_slice = whitened
    qtilde[kmin:kmax] = qtilde_slice

    # With a real signal, the SNR time series should also be real.
    # We use irfft which is designed for this purpose, producing a
    # real time series of length n_samples.
    # NOTE: hwilas applies an extra factor of n_samples to the unscaled
    # irfft result (see below). This makes ``corr`` exactly 2x the real part of
    # PyCBC's complex ``ifft(conj(h)*s/psd)`` output, so the normalisation
    # constant is halved relative to PyCBC (2*delta_f instead of 4*delta_f) for
    # the SNR to match PyCBC's convention. Without this the noise-floor SNR has
    # the wrong variance (std ~0.5 instead of ~1.0).
    corr: NDArray[np.float64] = np.fft.irfft(qtilde, n=n_samples) * n_samples

    sigmasq = sigmasq_frequencyseries(
        template_fs,
        psd,
        low_frequency_cutoff=low_frequency_cutoff,
        high_frequency_cutoff=high_frequency_cutoff,
    )

    if not np.isfinite(sigmasq) or sigmasq <= 0:
        snr_data: NDArray[np.float64] = np.zeros(n_samples, dtype=np.float64)
    else:
        # See note above: hwilas' irfft-based correlation is 2x PyCBC's real
        # part, so the normalisation constant is halved (2.0 instead of 4.0) to
        # match PyCBC's SNR convention.
        norm = (2.0 * delta_f) / np.sqrt(sigmasq)
        snr_data = corr * norm

    delta_t = 1.0 / (n_samples * template_fs.delta_f)
    return TimeSeries(snr_data, epoch=float(epoch), delta_t=delta_t)


def matched_filter_timeseries(
    template_ts: Any,
    data_ts: Any,
    *,
    psd: Any | None = None,
    n_samples: int | None = None,
    epoch: float | None = None,
) -> TimeSeries:
    """Compute a normalized matched-filter SNR time series from time series.

    This is the time-domain counterpart of :func:`matched_filter_frequencyseries`.
    The template and data are (optionally pre-whitened / overwhitened) time
    series; the SNR is computed as the cross-correlation between them,
    normalised so that a perfect match gives an SNR equal to sqrt(sigma^2).

    Normalisation
    -------------
    * If ``psd`` is provided, the inputs are assumed to be **pre-whitened**
      (i.e. already divided by the PSD in the frequency domain) and ``sigma^2``
      is the PSD-weighted inner product of the *unwhitened* template with
      itself, computed exactly as in :func:`sigmasq_frequencyseries`. This makes
      a perfect match yield an SNR of sqrt(sigma^2), matching
      :func:`matched_filter_frequencyseries` for any (non-flat) PSD.
    * If ``psd`` is ``None``, the inputs are assumed to be raw (unwhitened) and
      ``sigma^2`` is the unwhitened time-domain inner product of the template
      with itself.

    Parameters
    ----------
    template_ts : TimeSeries
        Time-domain template waveform (pre-whitened if ``psd`` is given).
    data_ts : TimeSeries
        Time-domain detector data (or overwhitened data).
    psd : FrequencySeries, optional
        One-sided noise PSD used to normalise the SNR. Must have the same
        frequency resolution as the inputs (length ``n_samples // 2 + 1`` for
        the output length). If ``None``, whitening is assumed to have been done
        externally and the unwhitened template norm is used instead.
    n_samples : int, optional
        Number of output time-domain samples. Defaults to the length of the
        longer of ``template_ts`` and ``data_ts``. If provided, it must be at
        least as long as both inputs.
    epoch : float, optional
        Epoch to attach to the output SNR time series. Defaults to the data
        epoch.

    Returns
    -------
    TimeSeries
        The normalised matched-filter SNR time series.
    """
    if template_ts.delta_t != data_ts.delta_t:
        raise ValueError(
            "template/data delta_t mismatch: "
            f"{template_ts.delta_t} / {data_ts.delta_t}"
        )

    n_template = len(template_ts)
    n_data = len(data_ts)
    if n_samples is None:
        n_samples = max(n_template, n_data)
    if n_samples <= 0:
        raise ValueError(f"n_samples must be positive, got {n_samples}")
    if n_samples < n_template or n_samples < n_data:
        raise ValueError(
            "n_samples must be at least as long as both the template and the data: "
            f"n_samples={n_samples}, template={n_template}, data={n_data}"
        )

    delta_t = float(template_ts.delta_t)
    if epoch is None:
        epoch = float(data_ts.epoch)

    # If a PSD is supplied it must match the frequency resolution implied by the
    # output length (one-sided grid of n_samples // 2 + 1 bins).
    if psd is not None:
        flen_expected = n_samples // 2 + 1
        if len(psd) != flen_expected:
            raise ValueError(
                "psd length mismatch: expected "
                f"{flen_expected} (n_samples // 2 + 1), got {len(psd)}"
            )

    # Zero-pad the template and data to n_samples so that the FFT-based
    # cross-correlation is linear (non-circular).
    h_pad: NDArray[np.float64] = np.zeros(n_samples, dtype=np.float64)
    s_pad: NDArray[np.float64] = np.zeros(n_samples, dtype=np.float64)
    h_pad[:n_template] = np.asarray(template_ts.data, dtype=np.float64)[:n_template]
    s_pad[:n_data] = np.asarray(data_ts.data, dtype=np.float64)[:n_data]

    # Linear cross-correlation via FFT. The result is the cyclic correlation of
    # the zero-padded sequences, which equals the linear correlation for lags
    # 0 .. n_samples - 1 (the negative lags wrap around to the end).
    htilde = np.fft.rfft(h_pad)
    stilde = np.fft.rfft(s_pad)
    corr: NDArray[np.float64] = np.fft.irfft(np.conjugate(htilde) * stilde, n=n_samples)

    # Normalise so that a perfect match yields an SNR of sqrt(sigma^2), matching
    # matched_filter_frequencyseries.
    #
    # The FD path computes corr_fd = irfft(conj(h)*s, n) * n, which equals the
    # linear cross-correlation sum_j h[j]*s[j+k] (by Parseval). This TD path
    # computes corr_td = irfft(conj(htilde)*stilde, n) *without* the factor of n,
    # so corr_fd == n_samples * corr_td. To produce an identical SNR series to
    # the FD path, the normalisation constant here must therefore be
    # n_samples times larger than in matched_filter_frequencyseries.
    #
    # If a PSD is given the inputs are pre-whitened, so sigma^2 must be the
    # PSD-weighted inner product of the *unwhitened* template with itself --
    # computed exactly as in sigmasq_frequencyseries. Using the norm of the
    # (already whitened) h_pad here would normalise by the wrong quantity and
    # give an SNR that depends on the PSD shape rather than being a true
    # matched-filter SNR.
    if psd is not None:
        template_fs = template_ts.to_frequencyseries()
        sigma_sq = sigmasq_frequencyseries(template_fs, psd)
        norm = (2.0 * n_samples * float(psd.delta_f)) / np.sqrt(sigma_sq)
    else:
        sigma_sq = float(4.0 / delta_t * np.sum(h_pad ** 2))
        norm = (2.0 * n_samples / delta_t) / np.sqrt(sigma_sq)

    if not np.isfinite(sigma_sq) or sigma_sq <= 0:
        snr_data: NDArray[np.float64] = np.zeros(n_samples, dtype=np.float64)
    else:
        snr_data = corr * norm

    return TimeSeries(snr_data, epoch=float(epoch), delta_t=delta_t)
