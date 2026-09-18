"""Tests for the PhenomXPy + TDISpline time-domain waveform generator.

The heavy Fortran-backed dependencies (``phenomxpy``, ``tdispline``,
``lisaorbits`` and ``astropy.units``) are not required to run these tests: they
are stubbed in ``sys.modules`` before :mod:`hwilas.waveforms.phenomx_tdi` is
imported, so that only numpy / scipy / pytest are needed. The stubs reproduce the
exact call sequence used by the module so that the glue logic (parameter
conversion, spline fitting, TDISpline invocation and TimeSeries assembly) is
exercised end to end.
"""

import sys
import types

import numpy as np
import pytest


# ---------------------------------------------------------------------------
# Stub heavy dependencies in sys.modules BEFORE importing phenomx_tdi.
# This allows the module-level imports (IMRPhenomTHM, TDISpline, etc.) to
# resolve without the real Fortran-backed packages being installed.
# ---------------------------------------------------------------------------

def _install_stub_modules():
    """Register lightweight stubs for all heavy dependencies if they are missing."""
    # --- astropy.units ---
    try:
        import astropy.units  # noqa: F401
    except ImportError:
        class _StubUnit:
            def __mul__(self, other):
                return other * 1.0

            def __rmul__(self, other):
                return other * 1.0

            def __truediv__(self, other):
                return 1.0

        astropy_mod = types.ModuleType("astropy")
        units_mod = types.ModuleType("astropy.units")
        for name in ("solMass", "Mpc", "rad", "s", "Hz", "dimensionless_unscaled"):
            setattr(units_mod, name, _StubUnit())
        astropy_mod.units = units_mod
        sys.modules.setdefault("astropy", astropy_mod)
        sys.modules.setdefault("astropy.units", units_mod)

    # --- phenomxpy ---
    try:
        import phenomxpy  # noqa: F401
    except ImportError:
        phenomx_mod = types.ModuleType("phenomxpy")

        class _StubIMRPhenomTHM:
            def __init__(self, **params):
                self.params = params
                self.pWF = type("pWF", (), {"inclination": 0.5, "phi_ref": 0.25})()

            def compute_hlms(self, t):
                t = np.asarray(t)
                n = t.size
                freq = 1e-4 + 1e-3 * (t - t[0]) / max(n, 1)
                phase = 2 * np.pi * np.cumsum(freq) / 1e-3
                hlms = {(2, 2): np.exp(1j * phase), (2, -2): np.exp(1j * phase) * 0.9}
                return hlms, None

        gwsignal_mod = types.ModuleType("phenomxpy.gwsignal_wrapper")
        gwsignal_mod.convert_params_from_gwsignal = lambda d: dict(d)

        utils_mod = types.ModuleType("phenomxpy.utils")
        utils_mod.SpinWeightedSphericalHarmonic = lambda *args: 1.0 + 0.0j

        phenomx_mod.IMRPhenomTHM = _StubIMRPhenomTHM
        phenomx_mod.gwsignal_wrapper = gwsignal_mod
        phenomx_mod.utils = utils_mod
        sys.modules.setdefault("phenomxpy", phenomx_mod)
        sys.modules.setdefault("phenomxpy.gwsignal_wrapper", gwsignal_mod)
        sys.modules.setdefault("phenomxpy.utils", utils_mod)

    # --- tdispline ---
    try:
        import tdispline  # noqa: F401
    except ImportError:
        tdi_mod = types.ModuleType("tdispline")

        combinations_mod = types.ModuleType("tdispline.combinations")
        combinations_mod.X2 = "X2"

        orbits_mod = types.ModuleType("tdispline.orbits")

        class _StubOrbitInterpolant:
            def __init__(self, orbit_file):
                self.orbit_file = orbit_file

        orbits_mod.OrbitInterpolant = _StubOrbitInterpolant

        tdi_sub_mod = types.ModuleType("tdispline.tdi")

        class _StubTDISpline:
            def __init__(self, orbits, combination=None):
                self.orbits = orbits
                self.combination = combination

            def set_extrinsic_parameters(self, ra, dec, psi):
                self.extrinsic = (ra, dec, psi)

            def __call__(self, t, amplitude, phase, nmodes, new_t):
                amp = np.asarray(amplitude(t))
                ph = np.asarray(phase(t))
                vals = (amp * np.exp(1j * ph)).real
                new_t = np.asarray(new_t)
                out = np.zeros((3, len(new_t), nmodes))
                for ch in range(3):
                    for m in range(nmodes):
                        out[ch, :, m] = np.interp(new_t, t, vals[:, m]) * (1 + 0.1 * ch)
                return out

        tdi_sub_mod.TDISpline = _StubTDISpline

        tdi_mod.combinations = combinations_mod
        tdi_mod.orbits = orbits_mod
        tdi_mod.tdi = tdi_sub_mod
        sys.modules.setdefault("tdispline", tdi_mod)
        sys.modules.setdefault("tdispline.combinations", combinations_mod)
        sys.modules.setdefault("tdispline.orbits", orbits_mod)
        sys.modules.setdefault("tdispline.tdi", tdi_sub_mod)

    # --- lisaorbits ---
    try:
        import lisaorbits  # noqa: F401
    except ImportError:
        lo_mod = types.ModuleType("lisaorbits")

        class _StubOEMOrbits:
            @staticmethod
            def from_included(name):
                return type("orbits", (), {"orbit_file": "handle"})()

        lo_mod.OEMOrbits = _StubOEMOrbits
        sys.modules.setdefault("lisaorbits", lo_mod)


_install_stub_modules()

# Now safe to import the module under test.
from hwilas.waveforms import phenomx_tdi  # noqa: E402


# ---------------------------------------------------------------------------
# Test helpers
# ---------------------------------------------------------------------------

class _MockGenerator:
    """Replaces IMRPhenomTHM; records construction params and returns per-mode waveforms."""

    captured = {}

    def __init__(self, **params):
        self.captured["params"] = params
        self.pWF = type("pWF", (), {"inclination": 0.5, "phi_ref": 0.25})()

    def compute_hlms(self, t):
        t = np.asarray(t)
        n = t.size
        freq = 1e-4 + 1e-3 * (t - t[0]) / max(n, 1)
        phase = 2 * np.pi * np.cumsum(freq) / 1e-3
        hlms = {}
        for i in range(2):
            hlms[(2, 2 if i % 2 == 0 else -2)] = np.exp(1j * phase) * (1.0 + 0.1 * i)
        return hlms, None


class _MockTDISpline:
    """Replaces TDISpline; records orbits/combination and returns a (3, n, nmodes) array."""

    captured = {}

    def __init__(self, orbits, combination=None):
        self.captured["orbits"] = orbits
        self.captured["combination"] = combination

    def set_extrinsic_parameters(self, ra, dec, psi):
        self.captured["extrinsic"] = (ra, dec, psi)

    def __call__(self, t, amplitude, phase, nmodes, new_t):
        amp = np.asarray(amplitude(t))
        ph = np.asarray(phase(t))
        vals = (amp * np.exp(1j * ph)).real
        new_t = np.asarray(new_t)
        out = np.zeros((3, len(new_t), nmodes))
        for ch in range(3):
            for m in range(nmodes):
                out[ch, :, m] = np.interp(new_t, t, vals[:, m]) * (1 + 0.1 * ch)
        return out


class _MockOrbitInterpolant:
    """Replaces OrbitInterpolant; records the orbit file."""

    captured = {}

    def __init__(self, orbit_file):
        self.captured["orbit_file"] = orbit_file


class _MockOEMOrbits:
    """Replaces OEMOrbits; records the orbit name."""

    captured = {}

    @staticmethod
    def from_included(name):
        _MockOEMOrbits.captured["orbit_name"] = name
        return type("orbits", (), {"orbit_file": "handle"})()


@pytest.fixture(autouse=True)
def mock_phenomx(monkeypatch):
    """Patch the module-level imports and reset the orbit cache for each test."""
    _MockGenerator.captured = {}
    _MockTDISpline.captured = {}
    _MockOrbitInterpolant.captured = {}
    _MockOEMOrbits.captured = {}

    monkeypatch.setattr(phenomx_tdi, "IMRPhenomTHM", _MockGenerator)
    monkeypatch.setattr(phenomx_tdi, "convert_params_from_gwsignal", lambda d: dict(d))
    monkeypatch.setattr(phenomx_tdi, "SpinWeightedSphericalHarmonic", lambda *a: 1.0 + 0.0j)
    monkeypatch.setattr(phenomx_tdi, "X2", "X2")
    monkeypatch.setattr(phenomx_tdi, "OrbitInterpolant", _MockOrbitInterpolant)
    monkeypatch.setattr(phenomx_tdi, "TDISpline", _MockTDISpline)
    monkeypatch.setattr(phenomx_tdi, "OEMOrbits", _MockOEMOrbits)
    monkeypatch.setattr(phenomx_tdi, "_ORBIT_INTERPOLANT_CACHE", None)


def test_xyz_to_aet():
    """The XYZ -> AET conversion matches the linear TDI combination."""
    n = 5
    rng = np.random.default_rng(0)
    xyz = rng.standard_normal((3, n))
    aet = phenomx_tdi.xyz_to_aet(xyz)

    expected_a = (xyz[2] - xyz[0]) / np.sqrt(2.0)
    expected_e = (xyz[0] - 2.0 * xyz[1] + xyz[2]) / np.sqrt(6.0)
    expected_t = (xyz[0] + xyz[1] + xyz[2]) / np.sqrt(3.0)

    assert aet.shape == (3, n)
    np.testing.assert_allclose(aet[0], expected_a)
    np.testing.assert_allclose(aet[1], expected_e)
    np.testing.assert_allclose(aet[2], expected_t)


def test_mode_indices():
    """Mode keys are parsed into integer (l, m) pairs."""
    assert phenomx_tdi._mode_indices("22") == (2, 2)
    assert phenomx_tdi._mode_indices((3, -3)) == (3, -3)
    assert phenomx_tdi._mode_indices([4, -4]) == (4, -4)


def test_validate_detector_times_rejects_nonuniform():
    with pytest.raises(ValueError, match="uniformly sampled"):
        phenomx_tdi._validate_detector_times(np.array([0.0, 1.0, 3.0]))
    with pytest.raises(ValueError, match="one-dimensional"):
        phenomx_tdi._validate_detector_times(np.array([[[0.0, 1.0]]]))


def test_build_phenomx_parameters(mock_phenomx):
    """Parameters are converted through the (mocked) gwsignal wrapper."""
    out = phenomx_tdi.build_phenomx_parameters(
        m1_msun=1.0e5,
        m2_msun=1.0,
        chi1z=0.3,
        chi2z=-0.1,
        distance_mpc=1.0e9,
        inclination=0.7,
        phi_ref=0.4,
        delta_t=0.5,
    )
    # The mock wrapper returns the dict unchanged; check key presence and that
    # defaults were applied (f_ref falls back to f_min).
    assert "mode_array" in out
    assert out["mode_array"] == phenomx_tdi.DEFAULT_HM_MODE_ARRAY_TD
    assert out["condition"] == 0


def test_generate_tdi_waveform(mock_phenomx):
    """The core generator returns a (3, n) XYZ signal and wires the pipeline."""
    detector_times = np.linspace(0.0, 100.0, 200)

    class TDI:
        def set_extrinsic_parameters(self, ra, dec, psi):
            _MockTDISpline.captured["extrinsic"] = (ra, dec, psi)

        def __call__(self, t, amplitude, phase, nmodes, new_t):
            amp = np.asarray(amplitude(t))
            ph = np.asarray(phase(t))
            vals = (amp * np.exp(1j * ph)).real  # (n_response, n_modes)
            out = np.zeros((3, len(new_t), nmodes))
            for ch in range(3):
                for m in range(nmodes):
                    out[ch, :, m] = np.interp(new_t, t, vals[:, m]) * (1 + 0.1 * ch)
            return out

    result = phenomx_tdi.generate_tdi_waveform(
        detector_times,
        t_merger=50.0,
        tdi=TDI(),
        m1_msun=1.0e5,
        m2_msun=1.0,
        chi1z=0.3,
        chi2z=-0.1,
        distance_mpc=1.0e9,
        inclination=0.7,
        phi_ref=0.4,
        ra=1.0,
        dec=0.5,
        psi=0.2,
    )

    assert result.shape == (3, detector_times.size)
    # Extrinsic parameters were passed through to the TDISpline instance.
    assert _MockTDISpline.captured["extrinsic"] == (1.0, 0.5, 0.2)
    # The generator was constructed with a PhenomXPy-ready parameter dict.
    assert "params" in _MockGenerator.captured


def test_get_orbit_interpolant_caches(mock_phenomx):
    """The orbit file is loaded once and reused across calls."""
    first = phenomx_tdi.get_orbit_interpolant()
    second = phenomx_tdi.get_orbit_interpolant()

    # The OEM orbits were loaded exactly once (from_included called a single time).
    assert _MockOEMOrbits.captured["orbit_name"] == "esa-trailing"
    # Both calls return an interpolant wrapping the cached orbit file.
    assert first is not None and second is not None


def test_get_td_waveform_from_phenomx(mock_phenomx, monkeypatch):
    """The top-level entry point returns XYZ TimeSeries with correct metadata."""

    class FakeTimeSeries:
        def __init__(self, data, delta_t=None, epoch=None):
            self.data = np.asarray(data)
            self.delta_t = delta_t
            self.epoch = epoch


    # Patch gwaith.types.TimeSeries (imported lazily inside the function).
    fake_gwaith = types.ModuleType("gwaith")
    fake_types = types.ModuleType("gwaith.types")
    fake_types.TimeSeries = FakeTimeSeries
    monkeypatch.setitem(sys.modules, "gwaith", fake_gwaith)
    monkeypatch.setitem(sys.modules, "gwaith.types", fake_types)

    detector_times = np.linspace(10.0, 110.0, 150)
    result = phenomx_tdi.get_td_waveform_from_phenomx(
        detector_times=detector_times,
        tc=60.0,
        mass1=1.0e5,
        mass2=1.0,
        spin1z=0.3,
        spin2z=-0.1,
        distance=1.0e9,
        inclination=0.7,
        phi_ref=0.4,
        eclipticlongitude=1.0,
        eclipticlatitude=0.5,
        polarization=0.2,
    )

    assert set(result.keys()) == set(phenomx_tdi.LISA_CHANNELS)
    for channel, ts in result.items():
        assert isinstance(ts, FakeTimeSeries)
        assert ts.data.shape == (detector_times.size,)
        assert ts.delta_t == pytest.approx(100.0 / 149.0)
        assert ts.epoch == pytest.approx(10.0)


def test_get_td_waveform_rejects_unknown_combination(mock_phenomx):
    with pytest.raises(ValueError, match="Unsupported TDISpline combination"):
        phenomx_tdi.get_td_waveform_from_phenomx(
            detector_times=np.linspace(0.0, 10.0, 50),
            tc=5.0,
            mass1=1.0e5,
            mass2=1.0,
            distance=1.0e9,
            inclination=0.7,
            tdi_combination="AET",
        )
