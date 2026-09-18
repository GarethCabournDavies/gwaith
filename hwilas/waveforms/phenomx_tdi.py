"""Time-domain MBHB waveform generation using PhenomXPy and TDISpline.


1. A PhenomXPy ``IMRPhenomTHM`` generator produces complex per-mode waveforms
   ``h_lm(t)`` on a source-time grid (time relative to merger).
2. Each mode is multiplied by its spin-weighted spherical harmonic (which encodes
   the inclination and polarization angle).
3. The amplitude ``|h|`` and unwrapped phase ``angle(h)`` are fit separately with
   cubic splines so that the slowly-varying LISA orbit response can be evaluated
   sparsely and interpolated onto the data grid.
4. ``tdispline.TDISpline`` (with LISA orbits from ``lisaorbits``) evaluates the
   detector-frame XYZ response, which is summed over modes.

The heavy dependencies (``phenomxpy``, ``tdispline``, ``lisaorbits`` and
``astropy.units``) are regular project dependencies, so this module imports them
directly at import time. Install them with::

    pip install hwilas

The generator returns waveforms in the XYZ TDI frame; use
:func:`hwilas.data_processing.tdi.linear_AET_conversion` to convert to AET if
needed.
"""

import logging

import astropy.units as u
import numpy as np
from lisaorbits import KeplerianOrbits, OEMOrbits
from phenomxpy import IMRPhenomTHM
from phenomxpy.gwsignal_wrapper import convert_params_from_gwsignal
from phenomxpy.utils import SpinWeightedSphericalHarmonic

from tdispline import TDISpline
from tdispline.numpy.combinations import X2
from tdispline.numpy.orbits import OrbitInterpolant

logger = logging.getLogger(__name__)

# Default set of (l, m) modes used by the PhenomXPy + TDISpline generator. This
# matches the mode array used in the reference ``mbhbs`` injection script.
DEFAULT_HM_MODE_ARRAY_TD = [
    [2, 2],
    [2, -2],
    [2, 1],
    [2, -1],
    [3, 3],
    [3, -3],
    [4, 4],
    [4, -4],
]

# LISA TDI channel names (XYZ frame).
LISA_CHANNELS = ("LISA_X", "LISA_Y", "LISA_Z")

# LISA TDI channel names (AET frame), obtained from XYZ via :func:`xyz_to_aet`.
AET_CHANNELS = ("LISA_A", "LISA_E", "LISA_T")


def xyz_to_aet(xyz):
    """Convert XYZ TDI channels to the AET frame.

    Parameters
    ----------
    xyz : array_like, shape (3, n) or (n,)
        The three XYZ TDI channels.

    Returns
    -------
    numpy.ndarray
        The corresponding A, E and T channels with the same shape as ``xyz``.
    """
    xyz = np.asarray(xyz, dtype=float)
    a = (xyz[2] - xyz[0]) / np.sqrt(2.0)
    e = (xyz[0] - 2.0 * xyz[1] + xyz[2]) / np.sqrt(6.0)
    t = (xyz[0] + xyz[1] + xyz[2]) / np.sqrt(3.0)
    return np.stack([a, e, t], axis=0)


# Cache the (expensive) LISA orbit interpolant so it is built only once per
# process. ``TDISpline`` instances are created per call because they are mutable
# and should not be shared between concurrent likelihood evaluations.
_ORBIT_INTERPOLANT_CACHE = None


def get_orbit_interpolant(orbit_file=None, orbit_name="esa-trailing"):
    """Return a :class:`tdispline.orbits.OrbitInterpolant` for LISA.

    Parameters
    ----------
    orbit_file : optional
        An explicit orbit object (e.g. an ``lisaorbits.OEMOrbits`` instance). If
        ``None``, the orbits are loaded from the bundled ``esa-trailing`` OEM
        ephemeris and cached for reuse.
    orbit_name : str, optional
        Name of the bundled OEM orbit set to load when ``orbit_file`` is ``None``.

    Returns
    -------
    tdispline.orbits.OrbitInterpolant
    """
    global _ORBIT_INTERPOLANT_CACHE

    if orbit_file is None:
        if _ORBIT_INTERPOLANT_CACHE is None:
            logger.debug("Loading LISA OEM orbits '%s'", orbit_name)
            orbit_file = OEMOrbits.from_included(orbit_name)
            # Detach the underlying file handle so the interpolant can be pickled /
            # shared safely (mirrors the reference implementation).
            orbit_file.orbit_file = None
        else:
            orbit_file = _ORBIT_INTERPOLANT_CACHE

    interpolant = OrbitInterpolant(orbit_file)
    if _ORBIT_INTERPOLANT_CACHE is None:
        _ORBIT_INTERPOLANT_CACHE = orbit_file
    return interpolant


def get_orbit_time_domain(orbit_interpolant=None, orbit_name="esa-trailing"):
    """Return the valid (start, end) time domain of a LISA orbit interpolant.

    ``TDISpline`` evaluates the slowly-varying TDI response on the supplied
    detector times using the orbit ephemeris. The bundled OEM orbits only cover a
    finite interval, so any detector time outside it is extrapolated and produces
    non-finite responses (which then break the internal ``CubicSpline`` fit). This
    helper exposes that valid domain so callers can check their time range before
    calling :func:`generate_tdi_waveform`.

    Parameters
    ----------
    orbit_interpolant : tdispline.orbits.OrbitInterpolant, optional
        An explicit interpolant to query. If ``None``, the cached bundled orbits are
        used (see :func:`get_orbit_interpolant`).
    orbit_name : str, optional
        Name of the bundled OEM orbit set, only used when ``orbit_interpolant`` is
        ``None``.

    Returns
    -------
    tuple of float
        ``(t_min, t_max)`` in seconds: the inclusive time interval covered by the
        orbit ephemeris.
    """
    if orbit_interpolant is None:
        orbit_interpolant = get_orbit_interpolant(orbit_name=orbit_name)
    t_interp = np.asarray(orbit_interpolant.orbit_file.t_interp, dtype=float)
    return float(t_interp[0]), float(t_interp[-1])


class AnalyticOrbitAdapter:
    """Wrap an analytic LISA orbit so it can be used by TDISpline at any time.

    The bundled OEM ephemeris (and ``lisaorbits.InterpolatedOrbits``) only cover a
    finite interval; :class:`tdispline.orbits.OrbitInterpolant` builds its internal
    linear interpolants over exactly that interval and returns NaN for times outside
    it. The analytic orbits in ``lisaorbits`` (e.g. :class:`KeplerianOrbits`) have no
    such limitation: their ``compute_position`` / ``compute_ltt`` are closed-form and
    valid at any time, but they do not expose the ``t_interp`` grid that
    ``OrbitInterpolant.__init__`` requires to build its sampling grid.

    This adapter bridges the two by delegating position/light-travel-time evaluation
    to the underlying analytic orbit while exposing a ``t_interp`` grid that spans an
    arbitrary (user supplied) time range. Because TDISpline only ever queries the
    interpolant through ``get_positions`` / ``get_ltts`` -- which evaluate the *linear*
    coefficients built on ``t_interp`` -- the underlying analytic values are sampled
    densely enough along that grid for the linear interpolation to be accurate, and no
    out-of-domain extrapolation (hence no NaN) can occur.

    Parameters
    ----------
    orbit : lisaorbits.Orbits
        An analytic orbit object exposing ``compute_position`` and ``compute_ltt``
        (e.g. a :class:`KeplerianOrbits` instance).
    t_min, t_max : float
        The inclusive time interval (seconds) that the internal sampling grid should
        span. Choose this to cover the detector times you intend to evaluate.
    n_grid : int, optional
        Number of nodes on the internal sampling grid. Default ``200_001`` (spacing of
        ~35 s for a week-long window), which is far finer than the light-travel-time
        scale (~8.3 ms) so linear interpolation error is negligible.
    """

    def __init__(self, orbit, t_min, t_max, n_grid=200_001):
        self._orbit = orbit
        self.t_interp = np.linspace(float(t_min), float(t_max), int(n_grid))

    def compute_position(self, t):
        return self._orbit.compute_position(np.asarray(t, dtype=float))

    def compute_ltt(self, t):
        return self._orbit.compute_ltt(np.asarray(t, dtype=float))


def get_analytic_orbit_interpolant(
    detector_times=None,
    *,
    t_min=None,
    t_max=None,
    n_grid=200_001,
    kepler_order=2,
):
    """Return an :class:`OrbitInterpolant` backed by analytic (unbounded) LISA orbits.

    Unlike the bundled OEM ephemeris, this interpolant is valid at *any* detector
    time, so it can be used for data that falls outside the finite orbit-ephemeris
    domain (see :func:`get_orbit_time_domain`). The analytic orbit is a
    :class:`KeplerianOrbits` model of LISA's nearly circular constellation.

    Parameters
    ----------
    detector_times : array_like, optional
        If given, the sampling grid is automatically chosen to span these times (with
        no extra margin). Provide ``t_min``/``t_max`` instead to control the range
        explicitly.
    t_min, t_max : float, optional
        Explicit inclusive time range (seconds) for the internal sampling grid.
    n_grid : int, optional
        Number of nodes on the internal sampling grid. Default ``200_001``.
    kepler_order : int, optional
        Order of the Keplerian expansion used by :class:`KeplerianOrbits`. Default 2.

    Returns
    -------
    tdispline.orbits.OrbitInterpolant
        An interpolant whose positions and light-travel-times are evaluated analytically.
    """
    if detector_times is not None:
        dt = np.asarray(detector_times, dtype=float)
        lo, hi = float(dt.min()), float(dt.max())
    else:
        if t_min is None or t_max is None:
            raise ValueError("Provide either detector_times or both t_min and t_max.")
        lo, hi = float(t_min), float(t_max)

    # A small margin guards against the linear-interpolation edges.
    span = max(hi - lo, 1.0)
    lo -= 0.5 * span
    hi += 0.5 * span

    orbit = KeplerianOrbits(kepler_order=kepler_order)
    adapter = AnalyticOrbitAdapter(orbit, lo, hi, n_grid=n_grid)
    return OrbitInterpolant(adapter)


def _validate_detector_times(detector_times):
    """Validate that detector times are one-dimensional and uniformly sampled."""
    detector_times = np.asarray(detector_times, dtype=float)
    if detector_times.ndim != 1 or detector_times.size < 2:
        raise ValueError("detector_times must be a one-dimensional array")

    sample_dt = detector_times[1] - detector_times[0]
    if not np.allclose(np.diff(detector_times), sample_dt):
        raise ValueError("detector_times must be uniformly sampled")
    return detector_times, sample_dt


def _mode_indices(mode):
    """Return the integer (l, m) indices for a mode key.

    Handles sequence keys (e.g. ``(2, 2)``) and string keys such as ``"22"`` or
    ``"2-2"`` (the latter used by some PhenomXPy output conventions).
    """
    if isinstance(mode, str):
        return int(mode[0]), int(mode[1:])
    return int(mode[0]), int(mode[1])


def build_phenomx_parameters(
    *,
    m1_msun,
    m2_msun,
    chi1z,
    chi2z,
    distance_mpc,
    inclination,
    phi_ref,
    delta_t,
    f_min_hz=1.0e-4,
    f_ref_hz=None,
    mode_array=None,
):
    """Build the PhenomXPy parameter dictionary from plain float parameters.

    This wraps :func:`phenomxpy.gwsignal_wrapper.convert_params_from_gwsignal` so
    that callers can pass unitless floats (matching the ``hwilas`` convention of
    masses in solar masses and distance in Mpc).

    Parameters
    ----------
    m1_msun, m2_msun : float
        Component masses in solar masses.
    chi1z, chi2z : float
        Dimensionless spin components along the line of sight.
    distance_mpc : float
        Luminosity distance in Mpc.
    inclination : float
        Inclination angle in radians.
    phi_ref : float
        Phase at the reference frequency in radians.
    delta_t : float
        Time sampling interval in seconds.
    f_min_hz : float, optional
        Minimum frequency in Hz. Default is ``1e-4``.
    f_ref_hz : float or None, optional
        Reference frequency in Hz. Defaults to ``f_min_hz`` if not given.
    mode_array : list of [int, int], optional
        The (l, m) modes to generate. Defaults to :data:`DEFAULT_HM_MODE_ARRAY_TD`.

    Returns
    -------
    dict
        A PhenomXPy-ready parameter dictionary with astropy units attached.
    """
    if mode_array is None:
        mode_array = DEFAULT_HM_MODE_ARRAY_TD
    if f_ref_hz is None:
        f_ref_hz = f_min_hz

    m1 = m1_msun * u.solMass
    m2 = m2_msun * u.solMass
    mass_ratio = m1 / m2

    return convert_params_from_gwsignal(
        {
            "eta": mass_ratio / (1.0 + mass_ratio) ** 2,
            "total_mass": m1 + m2,
            "mass_ratio": mass_ratio,
            "s1": [0.0 * u.dimensionless_unscaled] * 2
            + [chi1z * u.dimensionless_unscaled],
            "s2": [0.0 * u.dimensionless_unscaled] * 2
            + [chi2z * u.dimensionless_unscaled],
            "distance": distance_mpc * u.Mpc,
            "inclination": inclination * u.rad,
            "phi_ref": phi_ref * u.rad,
            "delta_t": delta_t * u.s,
            "condition": 0,
            "f_min": f_min_hz * u.Hz,
            "f_ref": f_ref_hz * u.Hz,
            "mode_array": mode_array,
        }
    )



def generate_tdi_waveform(
    detector_times,
    t_merger,
    tdi,
    *,
    m1_msun,
    m2_msun,
    chi1z,
    chi2z,
    distance_mpc,
    inclination,
    phi_ref,
    ra,
    dec,
    psi,
    f_min_hz=1.0e-4,
    f_ref_hz=None,
    mode_array=None,
    waveform_nodes=8_192,
    response_nodes=4_096,
    padding=1_000.0,
):
    """Generate a PhenomXPy waveform and its XYZ TDISpline response.

    All sampled parameters are plain floats. The supplied :class:`TDISpline` instance
    and its cached orbit interpolation are reused between calls. Because
    ``TDISpline`` is mutable, one instance should not be shared by concurrent
    likelihood calls.

    Parameters
    ----------
    detector_times : array_like, shape (n,)
        Uniformly sampled detector-frame times (seconds). These must lie within the
        time domain covered by the LISA orbit ephemeris (see
        :func:`get_orbit_time_domain`); otherwise the TDI response is extrapolated
        and non-finite values are produced.
    t_merger : float
        Detector-frame merger time (seconds). PhenomXPy evaluates the waveform on
        time relative to this value.
    tdi : tdispline.tdi.TDISpline
        A configured ``TDISpline`` instance (orbits + combination) used to apply
        the LISA orbit response.
    m1_msun, m2_msun : float
        Component masses in solar masses.
    chi1z, chi2z : float
        Dimensionless spin components along the line of sight.
    distance_mpc : float
        Luminosity distance in Mpc.
    inclination : float
        Inclination angle in radians.
    phi_ref : float
        Phase at the reference frequency in radians.
    ra, dec : float
        Right ascension and declination (radians) used for the TDISpline response.
    psi : float
        Polarization angle (radians).
    f_min_hz : float, optional
        Minimum frequency in Hz. Default ``1e-4``.
    f_ref_hz : float or None, optional
        Reference frequency in Hz. Defaults to ``f_min_hz``.
    mode_array : list of [int, int], optional
        The (l, m) modes to generate. Defaults to :data:`DEFAULT_HM_MODE_ARRAY_TD`.
    waveform_nodes : int, optional
        Number of source-time nodes on which the PhenomXPy waveform is evaluated.
    response_nodes : int, optional
        Number of detector-time nodes on which the TDISpline response is evaluated
        before interpolating onto ``detector_times``.
    padding : float, optional
        Extra time (seconds) padded at either end of the waveform evaluation grid.

    Returns
    -------
    numpy.ndarray, shape (3, n)
        The XYZ detector-frame waveform sampled on ``detector_times``.
    """
    from scipy.interpolate import CubicSpline

    detector_times, sample_dt = _validate_detector_times(detector_times)

    # TDISpline evaluates the TDI response on ``detector_times`` using the orbit
    # ephemeris. The bundled OEM orbits only cover a finite interval; evaluating
    # outside it extrapolates to non-finite values and later breaks the internal
    # ``CubicSpline`` fit with "y must contain only finite values". Fail fast with
    # an actionable message instead. Analytic orbits (see
    # :func:`get_analytic_orbit_interpolant`) are valid at any time, so the check is
    # skipped for them.
    if not isinstance(tdi.orbits.orbit_file, AnalyticOrbitAdapter):
        orbit_t_min, orbit_t_max = get_orbit_time_domain(tdi.orbits)
        if detector_times[0] < orbit_t_min or detector_times[-1] > orbit_t_max:
            raise ValueError(
                "detector_times are outside the valid LISA orbit domain. "
                f"Requested range [{detector_times[0]:.3f}, {detector_times[-1]:.3f}] s "
                f"but the orbits only cover [{orbit_t_min:.3f}, {orbit_t_max:.3f}] s "
                f"(~{(orbit_t_max - orbit_t_min) / 86400.0:.1f} days). "
                "Choose a detector time range that lies entirely within the orbit "
                "domain, or pass orbit='analytic' to use an unbounded analytic orbit."
            )

    if mode_array is None:
        mode_array = DEFAULT_HM_MODE_ARRAY_TD
    if f_ref_hz is None:
        f_ref_hz = f_min_hz

    waveform_parameters = build_phenomx_parameters(
        m1_msun=m1_msun,
        m2_msun=m2_msun,
        chi1z=chi1z,
        chi2z=chi2z,
        distance_mpc=distance_mpc,
        inclination=inclination,
        phi_ref=phi_ref,
        delta_t=sample_dt,
        f_min_hz=f_min_hz,
        f_ref_hz=f_ref_hz,
        mode_array=mode_array,
    )
    waveform_generator = IMRPhenomTHM(**waveform_parameters)

    # PhenomXPy uses time relative to merger. A coarse source grid is sufficient
    # because amplitude and unwrapped phase are interpolated separately.
    waveform_times = np.linspace(
        detector_times[0] - t_merger - padding,
        detector_times[-1] - t_merger + padding,
        waveform_nodes,
    )
    hlms, _ = waveform_generator.compute_hlms(waveform_times)
    mode_keys = list(hlms)

    harmonics = np.array(
        [
            SpinWeightedSphericalHarmonic(
                waveform_generator.pWF.inclination,
                np.pi / 2 - waveform_generator.pWF.phi_ref,
                *_mode_indices(mode),
            )
            for mode in mode_keys
        ]
    )
    mode_waveforms = (
        np.column_stack([hlms[mode] for mode in mode_keys]) * harmonics
    )

    source_times = waveform_times + t_merger
    amplitude = CubicSpline(source_times, np.abs(mode_waveforms), axis=0)
    phase = CubicSpline(
        source_times,
        np.unwrap(np.angle(mode_waveforms), axis=0),
        axis=0,
    )

    # NOTE: the installed TDISpline (uk-lisa-gs main) only accepts (ra, dec);
    # its polarisation convention is fixed. The ``psi`` argument is accepted for
    # API compatibility but has no effect on the generated waveform.
    tdi.set_extrinsic_parameters(ra, dec)
    response_times = np.linspace(detector_times[0], detector_times[-1], response_nodes)
    tdi_modes = tdi(
        response_times,
        amplitude,
        phase,
        nmodes=len(mode_keys),
        new_t=detector_times,
    )
    return tdi_modes.sum(axis=-1)



def get_td_waveform_from_phenomx(
    *,
    detector_times,
    tc,
    mass1,
    mass2,
    spin1z=0.0,
    spin2z=0.0,
    distance,
    inclination,
    phi_ref=0.0,
    eclipticlongitude=0.0,
    eclipticlatitude=0.0,
    polarization=0.0,
    f_lower=1.0e-4,
    f_ref=None,
    mode_array=None,
    ifos=None,
    waveform_nodes=8_192,
    response_nodes=4_096,
    padding=1_000.0,
    tdi_combination="X2",
    orbit="oem",
    **_extra,
):
    """Generate a time-domain XYZ detector waveform from PhenomXPy + TDISpline.

    This is the top-level entry point for the time-domain MBHB waveform path. It
    builds the LISA orbit interpolant and ``TDISpline`` instance, evaluates the
    PhenomXPy waveform through the orbit response, and returns the result as a
    dictionary of :class:`gwaith.types.TimeSeries` objects in the XYZ TDI frame.

    Parameter names follow the ``hwilas`` convention (see
    :func:`hwilas.template_banks.helpers.waveform_from_bank`) so that this can be
    driven directly from a template bank.

    Parameters
    ----------
    detector_times : array_like, shape (n,)
        Uniformly sampled detector-frame times (seconds). The returned TimeSeries
        use ``detector_times[0]`` as their epoch and the sampling interval as
        ``delta_t``.
    tc : float
        Detector-frame merger time (seconds).
    mass1, mass2 : float
        Component masses in solar masses.
    spin1z, spin2z : float, optional
        Dimensionless spin components along the line of sight. Default ``0.0``.
    distance : float
        Luminosity distance in Mpc.
    inclination : float
        Inclination angle in radians.
    phi_ref : float, optional
        Phase at the reference frequency in radians. Default ``0.0``.
    eclipticlongitude : float, optional
        Ecliptic longitude (radians), mapped to right ascension for TDISpline.
    eclipticlatitude : float, optional
        Ecliptic latitude (radians), mapped to declination for TDISpline.
    polarization : float, optional
        Polarization angle (radians). Default ``0.0``.
    f_lower : float, optional
        Minimum frequency in Hz. Default ``1e-4``.
    f_ref : float or None, optional
        Reference frequency in Hz. Defaults to ``f_lower`` if not given.
    mode_array : list of [int, int], optional
        The (l, m) modes to generate. Defaults to :data:`DEFAULT_HM_MODE_ARRAY_TD`.
    waveform_nodes : int, optional
        Number of source-time nodes for the PhenomXPy evaluation grid.
    response_nodes : int, optional
        Number of detector-time nodes for the TDISpline response evaluation.
    padding : float, optional
        Extra time (seconds) padded at either end of the waveform grid.
    tdi_combination : str, optional
        Name of the TDISpline combination to use. Currently only ``"X2"`` is
        supported. Default ``"X2"``.
    orbit : {'oem', 'analytic'}, optional
        Source of the LISA orbits used for the TDI response.

        * ``'oem'`` (default) uses the bundled finite OEM ephemeris; detector times
          must then lie within its domain (see :func:`get_orbit_time_domain`).
        * ``'analytic'`` uses an analytic :class:`KeplerianOrbits` model that is valid
          at any time, so it can be used for data outside the ephemeris domain. The
          internal sampling grid is chosen automatically to span ``detector_times``.

        Default ``'oem'``.
    ifos : list of str, optional
        Which channels to return. XYZ channels (``'LISA_X'``, ``'LISA_Y'``,
        ``'LISA_Z'``) are returned directly from the TDISpline evaluation; AET
        channels (``'LISA_A'``, ``'LISA_E'``, ``'LISA_T'``) are obtained by
        rotating the XYZ frame via :func:`xyz_to_aet`. If ``None``, all three
        XYZ channels (:data:`LISA_CHANNELS`) are returned. This mirrors the
        ``ifos`` argument of the BBHx generator so that both can be driven from
        the same matched-filter search loop.
    **_extra : dict
        Any additional shared waveform parameters (e.g. ``delta_f``, ``length``,
        ``f_final``, ``t_offset``) passed alongside this call are accepted and
        ignored, matching the permissive behaviour of the BBHx generator.

    Returns
    -------
    dict
        Dictionary mapping the requested channel names (a subset of ``'LISA_X'``,
        ``'LISA_Y'`` and ``'LISA_Z'``) to :class:`gwaith.types.TimeSeries` objects.
    """
    detector_times = np.asarray(detector_times, dtype=float)

    if tdi_combination != "X2":
        raise ValueError(f"Unsupported TDISpline combination: {tdi_combination}")

    if ifos is None:
        channels = list(LISA_CHANNELS)
    else:
        unknown = [ifo for ifo in ifos if ifo not in LISA_CHANNELS + AET_CHANNELS]
        if unknown:
            raise ValueError(
                f"Unknown PhenomX TD channel(s) {unknown}; "
                f"expected a subset of {list(LISA_CHANNELS)} or {list(AET_CHANNELS)}."
            )
        channels = list(ifos)

    if orbit == "analytic":
        orbits = get_analytic_orbit_interpolant(detector_times)
    elif orbit == "oem":
        orbits = get_orbit_interpolant()
    else:
        raise ValueError(f"Unsupported orbit source: {orbit!r} (expected 'oem' or 'analytic')")
    tdi = TDISpline(orbits, combination=X2)

    # AET channels require the full XYZ frame so that the rotation below is
    # well defined; in that case all three XYZ signals must be generated.
    need_full_xyz = any(channel not in LISA_CHANNELS for channel in channels)

    xyz_signal = generate_tdi_waveform(
        detector_times,
        tc,
        tdi,
        m1_msun=mass1,
        m2_msun=mass2,
        chi1z=spin1z,
        chi2z=spin2z,
        distance_mpc=distance,
        inclination=inclination,
        phi_ref=phi_ref,
        ra=eclipticlongitude,
        dec=eclipticlatitude,
        psi=polarization,
        f_min_hz=f_lower,
        f_ref_hz=f_ref,
        mode_array=mode_array,
        waveform_nodes=waveform_nodes,
        response_nodes=response_nodes,
        padding=padding,
    )

    if need_full_xyz:
        aet_signal = xyz_to_aet(xyz_signal)
        signal_by_channel = {
            **{channel: xyz_signal[LISA_CHANNELS.index(channel)] for channel in LISA_CHANNELS},
            **{channel: aet_signal[AET_CHANNELS.index(channel)] for channel in AET_CHANNELS},
        }
    else:
        signal_by_channel = {
            channel: xyz_signal[LISA_CHANNELS.index(channel)] for channel in channels
        }

    from gwaith.types import TimeSeries

    delta_t = detector_times[1] - detector_times[0]
    epoch = float(detector_times[0])
    return {
        channel: TimeSeries(signal_by_channel[channel], delta_t=delta_t, epoch=epoch)
        for channel in channels
    }


__all__ = [
    "AET_CHANNELS",
    "AnalyticOrbitAdapter",
    "DEFAULT_HM_MODE_ARRAY_TD",
    "LISA_CHANNELS",
    "build_phenomx_parameters",
    "generate_tdi_waveform",
    "get_analytic_orbit_interpolant",
    "get_orbit_interpolant",
    "get_orbit_time_domain",
    "get_td_waveform_from_phenomx",
    "xyz_to_aet",
]

