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
    # The result of irfft is unscaled, so we apply a factor of n_samples
    # to be consistent with the unscaled ifft.
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
        norm = (4.0 * delta_f) / np.sqrt(sigmasq)
        snr_data = corr * norm

    delta_t = 1.0 / (n_samples * template_fs.delta_f)
    return TimeSeries(snr_data, epoch=float(epoch), delta_t=delta_t)
