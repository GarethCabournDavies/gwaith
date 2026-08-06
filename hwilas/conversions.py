"""Mass-parameter conversion helpers used across search workflows."""

from __future__ import annotations

from numbers import Real
from typing import TypeAlias

import numpy as np
from numpy.typing import ArrayLike, NDArray

FloatArray: TypeAlias = NDArray[np.float64]
ScalarNumeric: TypeAlias = Real | np.number
NumericInput: TypeAlias = ScalarNumeric | ArrayLike


def _all_scalar(*values: NumericInput) -> bool:
    """True only when every original input was scalar-like."""
    return all(np.isscalar(v) for v in values)


def mass1_mass2_from_mchirp_eta(
    mchirp: NumericInput,
    eta: NumericInput,
) -> tuple[float | FloatArray, float | FloatArray]:
    """
    Convert (mchirp, eta) to (mass1, mass2).
    Works with scalar or arrays for mchirp and eta, as well as mixed types
    """
    scalar_output = _all_scalar(mchirp, eta)
    mchirp_arr = np.asarray(mchirp, dtype=float)
    eta_arr = np.asarray(eta, dtype=float)
    mchirp_arr, eta_arr = np.broadcast_arrays(mchirp_arr, eta_arr)

    if np.any(mchirp_arr <= 0):
        raise ValueError("mchirp must be positive")
    if np.any((eta_arr <= 0.0) | (eta_arr > 0.25)):
        raise ValueError("eta must be in (0, 0.25]")

    mtotal = mchirp_arr / (eta_arr ** (3.0 / 5.0))
    disc = np.sqrt(np.maximum(0.0, 1.0 - 4.0 * eta_arr))
    mass1 = 0.5 * mtotal * (1.0 + disc)
    mass2 = 0.5 * mtotal * (1.0 - disc)

    if scalar_output:
        return float(mass1), float(mass2)
    return mass1, mass2


def mchirp_eta_from_mass1_mass2(
    mass1: NumericInput,
    mass2: NumericInput,
) -> tuple[float | FloatArray, float | FloatArray]:
    """
    Convert (mass1, mass2) to (mchirp, eta).
    Works with scalar or arrays for mass1 and mass2, as well as mixed types
    """
    scalar_output = _all_scalar(mass1, mass2)
    mass1_arr = np.asarray(mass1, dtype=float)
    mass2_arr = np.asarray(mass2, dtype=float)
    mass1_arr, mass2_arr = np.broadcast_arrays(mass1_arr, mass2_arr)

    if np.any(mass1_arr <= 0) or np.any(mass2_arr <= 0):
        raise ValueError("mass1 and mass2 must be positive")

    mtotal = mass1_arr + mass2_arr
    eta = (mass1_arr * mass2_arr) / (mtotal * mtotal)
    mchirp = mtotal * (eta ** (3.0 / 5.0))

    if scalar_output:
        return float(mchirp), float(eta)
    return mchirp, eta


def chi_eff_from_component_spins(
    mass1: NumericInput,
    mass2: NumericInput,
    spin1z: NumericInput,
    spin2z: NumericInput,
) -> float | FloatArray:
    """Convert (mass1, mass2, spin1z, spin2z) to chi_eff."""
    scalar_output = _all_scalar(mass1, mass2, spin1z, spin2z)
    mass1_arr = np.asarray(mass1, dtype=float)
    mass2_arr = np.asarray(mass2, dtype=float)
    spin1z_arr = np.asarray(spin1z, dtype=float)
    spin2z_arr = np.asarray(spin2z, dtype=float)
    mass1_arr, mass2_arr, spin1z_arr, spin2z_arr = np.broadcast_arrays(
        mass1_arr,
        mass2_arr,
        spin1z_arr,
        spin2z_arr,
    )

    if np.any(mass1_arr <= 0) or np.any(mass2_arr <= 0):
        raise ValueError("mass1 and mass2 must be positive")

    chi_eff = (mass1_arr * spin1z_arr + mass2_arr * spin2z_arr) / (mass1_arr + mass2_arr)
    if scalar_output:
        return float(chi_eff)
    return chi_eff
