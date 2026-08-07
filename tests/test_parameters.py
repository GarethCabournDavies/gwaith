"""Tests for physical parameter conversion helpers."""

import numpy as np
import pytest

from hwilas.parameters import (
    chi_eff_from_component_spins,
    mass1_mass2_from_mchirp_eta,
    mchirp_eta_from_mass1_mass2,
)

# Generate some self-consistent test data
# Note that m1 is always the larger mass
MASS_PAIRS = [(10.0, 10.0), (15.0, 10.0), (20.0, 5.0)]
MCHIRP_ETA_PAIRS = [mchirp_eta_from_mass1_mass2(m1, m2) for m1, m2 in MASS_PAIRS]


@pytest.mark.parametrize(
    ("mchirp", "eta", "m1", "m2"),
    [
        (mc_eta[0], mc_eta[1], m[0], m[1])
        for mc_eta, m in zip(MCHIRP_ETA_PAIRS, MASS_PAIRS)
    ],
)
def test_mass1_mass2_from_mchirp_eta_scalar(
    mchirp: float,
    eta: float,
    m1: float,
    m2: float,
) -> None:
    """Test scalar (mchirp, eta) -> (m1, m2) conversion."""
    new_m1, new_m2 = mass1_mass2_from_mchirp_eta(mchirp, eta)
    assert isinstance(new_m1, float)
    assert isinstance(new_m2, float)
    # The function returns m1 as the larger mass
    assert np.isclose(new_m1, m1, rtol=1e-5)
    assert np.isclose(new_m2, m2, rtol=1e-5)


def test_mass1_mass2_from_mchirp_eta_array() -> None:
    """Test array (mchirp, eta) -> (m1, m2) conversion."""
    mchirps = np.array([p[0] for p in MCHIRP_ETA_PAIRS])
    etas = np.array([p[1] for p in MCHIRP_ETA_PAIRS])
    expected_m1s = np.array([p[0] for p in MASS_PAIRS])
    expected_m2s = np.array([p[1] for p in MASS_PAIRS])

    m1s, m2s = mass1_mass2_from_mchirp_eta(mchirps, etas)
    assert isinstance(m1s, np.ndarray)
    assert isinstance(m2s, np.ndarray)
    assert m1s.shape == mchirps.shape
    assert m2s.shape == mchirps.shape
    assert np.allclose(m1s, expected_m1s, rtol=1e-5)
    assert np.allclose(m2s, expected_m2s, rtol=1e-5)


@pytest.mark.parametrize("mchirp", [-1.0, 0.0, np.array([-1.0, 1.0])])
def test_mass1_mass2_from_mchirp_eta_invalid_mchirp(
    mchirp: float | np.ndarray,
) -> None:
    """Test that invalid mchirp values raise a ValueError."""
    with pytest.raises(ValueError, match="mchirp must be positive"):
        mass1_mass2_from_mchirp_eta(mchirp, 0.25)


@pytest.mark.parametrize("eta", [-1.0, 0.0, 0.251, np.array([0.25, -0.1])])
def test_mass1_mass2_from_mchirp_eta_invalid_eta(
    eta: float | np.ndarray,
) -> None:
    """Test that invalid eta values raise a ValueError."""
    with pytest.raises(ValueError, match="eta must be in .*"):
        mass1_mass2_from_mchirp_eta(10.0, eta)


@pytest.mark.parametrize(
    ("m1", "m2", "mchirp", "eta"),
    [
        (m[0], m[1], mc_eta[0], mc_eta[1])
        for m, mc_eta in zip(MASS_PAIRS, MCHIRP_ETA_PAIRS)
    ],
)
def test_mchirp_eta_from_mass1_mass2_scalar(
    m1: float,
    m2: float,
    mchirp: float,
    eta: float,
) -> None:
    """Test scalar (m1, m2) -> (mchirp, eta) conversion."""
    new_mchirp, new_eta = mchirp_eta_from_mass1_mass2(m1, m2)
    assert isinstance(new_mchirp, float)
    assert isinstance(new_eta, float)
    assert np.isclose(new_mchirp, mchirp, rtol=1e-5)
    assert np.isclose(new_eta, eta, rtol=1e-5)


def test_mchirp_eta_from_mass1_mass2_array() -> None:
    """Test array (m1, m2) -> (mchirp, eta) conversion."""
    m1s = np.array([p[0] for p in MASS_PAIRS])
    m2s = np.array([p[1] for p in MASS_PAIRS])
    expected_mchirps = np.array([p[0] for p in MCHIRP_ETA_PAIRS])
    expected_etas = np.array([p[1] for p in MCHIRP_ETA_PAIRS])

    mchirps, etas = mchirp_eta_from_mass1_mass2(m1s, m2s)
    assert isinstance(mchirps, np.ndarray)
    assert isinstance(etas, np.ndarray)
    assert mchirps.shape == m1s.shape
    assert etas.shape == m1s.shape
    assert np.allclose(mchirps, expected_mchirps, rtol=1e-5)
    assert np.allclose(etas, expected_etas, rtol=1e-5)


@pytest.mark.parametrize(
    ("m1", "m2"),
    [(-1.0, 10.0), (10.0, -1.0), (0.0, 10.0), (10.0, 0.0)],
)
def test_mchirp_eta_from_mass1_mass2_invalid_masses(m1: float, m2: float) -> None:
    """Test that invalid masses raise a ValueError."""
    with pytest.raises(ValueError, match="mass1 and mass2 must be positive"):
        mchirp_eta_from_mass1_mass2(m1, m2)


def test_mass_parameter_roundtrip() -> None:
    """Test roundtrip conversion between mass parameter spaces."""
    rng = np.random.default_rng(0)
    size = (100,)
    mchirps = rng.uniform(5, 100, size=size)
    etas = rng.uniform(0.01, 0.25, size=size)

    m1s, m2s = mass1_mass2_from_mchirp_eta(mchirps, etas)
    new_mchirps, new_etas = mchirp_eta_from_mass1_mass2(m1s, m2s)

    assert np.allclose(new_mchirps, mchirps)
    assert np.allclose(new_etas, etas)


@pytest.mark.parametrize(
    ("m1", "m2", "spin1z", "spin2z", "chi_eff"),
    [
        (10.0, 10.0, 0.5, -0.5, 0.0),
        (20.0, 10.0, 0.5, 0.2, 0.4),
        (10.0, 20.0, 0.2, 0.5, 0.4),
        (10.0, 10.0, 0.8, 0.8, 0.8),
    ],
)
def test_chi_eff_from_component_spins_scalar(
    m1: float,
    m2: float,
    spin1z: float,
    spin2z: float,
    chi_eff: float,
) -> None:
    """Test scalar chi_eff calculation."""
    new_chi_eff = chi_eff_from_component_spins(m1, m2, spin1z, spin2z)
    assert isinstance(new_chi_eff, float)
    assert np.isclose(new_chi_eff, chi_eff)


def test_chi_eff_from_component_spins_array() -> None:
    """Test array chi_eff calculation."""
    m1s = np.array([10.0, 20.0, 10.0, 10.0])
    m2s = np.array([10.0, 10.0, 20.0, 10.0])
    s1zs = np.array([0.5, 0.5, 0.2, 0.8])
    s2zs = np.array([-0.5, 0.2, 0.5, 0.8])
    expected_chi_effs = np.array([0.0, 0.4, 0.4, 0.8])

    chi_effs = chi_eff_from_component_spins(m1s, m2s, s1zs, s2zs)
    assert isinstance(chi_effs, np.ndarray)
    assert chi_effs.shape == m1s.shape
    assert np.allclose(chi_effs, expected_chi_effs)


@pytest.mark.parametrize(
    ("m1", "m2"),
    [(-1.0, 10.0), (10.0, -1.0), (0.0, 10.0), (10.0, 0.0)],
)
def test_chi_eff_from_component_spins_invalid_masses(
    m1: float,
    m2: float,
) -> None:
    """Test that invalid masses raise a ValueError."""
    with pytest.raises(ValueError, match="mass1 and mass2 must be positive"):
        chi_eff_from_component_spins(m1, m2, 0.0, 0.0)
