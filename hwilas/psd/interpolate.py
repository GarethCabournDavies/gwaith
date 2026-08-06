"""Interpolate frequency-domain PSD series to a target delta_f."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

try:
    from pycbc.types import FrequencySeries as PycbcFrequencySeries
except Exception:  # pragma: no cover - optional at import time
    PycbcFrequencySeries = None

try:
    from gwaith.types import FrequencySeries as GwaithFrequencySeries
except Exception:  # pragma: no cover - optional at import time
    GwaithFrequencySeries = None


def _to_numpy(values) -> NDArray[np.float64]:
    if hasattr(values, "numpy"):
        return np.asarray(values.numpy(), dtype=float)
    return np.asarray(values, dtype=float)


def interpolate(series, delta_f, length=None):
    """Return a new PSD interpolated to the desired ``delta_f``.

    This mirrors the behavior used by ``pycbc.psd.interpolate``.
    """
    if length is None:
        new_n = (len(series) - 1) * float(series.delta_f) / float(delta_f) + 1
    else:
        new_n = int(length)

    n_samples = int(np.rint(new_n))
    samples = np.arange(0, n_samples, dtype=float) * float(delta_f)

    freqs = _to_numpy(series.sample_frequencies)
    data = _to_numpy(series)
    interpolated_data = np.interp(samples, freqs, data)

    epoch = getattr(series, "epoch", None)

    if PycbcFrequencySeries is not None and isinstance(series, PycbcFrequencySeries):
        return PycbcFrequencySeries(
            interpolated_data,
            epoch=epoch,
            delta_f=float(delta_f),
            dtype=series.dtype,
        )

    if GwaithFrequencySeries is not None and isinstance(series, GwaithFrequencySeries):
        return GwaithFrequencySeries(
            interpolated_data,
            delta_f=float(delta_f),
            f0=getattr(series, "f0", 0.0),
            epoch=epoch,
        )

    # Fallback for compatible FrequencySeries-like classes.
    return series.__class__(interpolated_data, delta_f=float(delta_f), epoch=epoch)
