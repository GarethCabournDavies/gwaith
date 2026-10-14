"""PhenomXPy IMRPhenomTHM modes with the NumPy TDISpline response."""
import logging

import numpy as np
from scipy.interpolate import CubicSpline

from hwilas.data_processing.tdi import linear_AET_conversion, to_timeseries
from .sampling import detector_grid, requested_channels, tdi_version

logger = logging.getLogger(__name__)
DEFAULT_MODES = [(2, 2), (2, -2), (2, 1), (2, -1),
                 (3, 3), (3, -3), (4, 4), (4, -4)]


class AnalyticOrbits:
    """Expose LISAOrbits' ICRS Keplerian positions and retarded link times."""
    def __init__(self):
        from lisaorbits import KeplerianOrbits
        self.orbits = KeplerianOrbits()

    def get_positions(self, times):
        return self.orbits.compute_position(times)

    def get_ltts(self, times):
        return self.orbits.compute_ltt(times)


def equatorial_angles(longitude, latitude, polarization):
    """Rotate both the ecliptic source direction and its polarization basis."""
    eps = np.deg2rad(23.439291111)
    rotation = np.array([[1., 0., 0.], [0., np.cos(eps), -np.sin(eps)],
                         [0., np.sin(eps), np.cos(eps)]])
    direction = rotation @ np.array([np.cos(latitude)*np.cos(longitude),
                                    np.cos(latitude)*np.sin(longitude), np.sin(latitude)])
    ra = np.arctan2(direction[1], direction[0]) % (2*np.pi)
    dec = np.arcsin(np.clip(direction[2], -1., 1.))
    u = rotation @ np.array([np.sin(longitude), -np.cos(longitude), 0.])
    u_eq = np.array([np.sin(ra), -np.cos(ra), 0.])
    v_eq = np.array([-np.sin(dec)*np.cos(ra), -np.sin(dec)*np.sin(ra), np.cos(dec)])
    return ra, dec, polarization+np.arctan2(u @ v_eq, u @ u_eq)


def get_td_waveform_from_phenomthm(approximant, tdi='2', **params):
    """Generate XYZ or AET TimeSeries; masses in solar masses, distance in Mpc.

    Angles are physical SSB inclination/polarization and orbital phase at
    ``f_ref``. ``data_time_origin`` maps detector seconds onto orbit seconds.
    Supply an equatorial orbit object via ``orbit``, analytic or ESA OEM orbits.
    """
    times, dt = detector_grid(params)
    channels = requested_channels(params)
    version = tdi_version(tdi)
    from phenomxpy import IMRPhenomTHM
    from phenomxpy.utils import SpinWeightedSphericalHarmonic
    from tdispline.numpy.combinations import X1, X2
    from tdispline.numpy.tdi import TDISpline

    f_min = float(params.get('f_lower', 1e-4))
    f_ref = float(params.get('f_ref', f_min))
    if not np.isfinite(f_min) or not np.isfinite(f_ref) or min(f_min, f_ref) <= 0:
        raise ValueError('f_lower and f_ref must be finite and positive')
    padding = float(params.get('padding', 1000.))
    oversampling = int(params.get('source_oversampling', 4))
    if padding < 1000 or oversampling < 1:
        raise ValueError('Require padding >= 1000 seconds and source_oversampling >= 1')
    mass1, mass2 = float(params['mass1']), float(params['mass2'])
    phase_ref = float(params.get('coa_phase', params.get('phi_ref', 0.)))
    source_dt = dt/oversampling
    phenom = IMRPhenomTHM(
        eta=mass1*mass2/(mass1+mass2)**2, total_mass=mass1+mass2,
        s1=[0., 0., params.get('spin1z', 0.)], s2=[0., 0., params.get('spin2z', 0.)],
        distance=params['distance'], inclination=params['inclination'], phi_ref=phase_ref,
        delta_t=source_dt, f_min=f_min, f_ref=f_ref, condition=0,
        mode_array=params.get('mode_array', DEFAULT_MODES))
    origin = float(params.get('data_time_origin', 0.))
    orbit_times = times+origin
    source_times = np.arange(orbit_times[0]-padding, orbit_times[-1]+padding+source_dt, source_dt)
    modes, _ = phenom.compute_hlms(source_times-origin-float(params['tc']))
    ra, dec, psi = equatorial_angles(params['eclipticlongitude'], params['eclipticlatitude'],
                                     params['polarization'])
    waves = []
    for mode, wave in modes.items():
        l, m = (int(mode[0]), int(mode[1:])) if isinstance(mode, str) else mode
        waves.append(wave*SpinWeightedSphericalHarmonic(
            phenom.pWF.inclination, np.pi/2-phenom.pWF.phi_ref, l, m))
    h = np.column_stack(waves)*np.exp(-2j*psi)
    amplitude = CubicSpline(source_times, np.abs(h), axis=0, extrapolate=False)
    # TDISpline reconstructs complex strain with exp(-i phase).
    phase = CubicSpline(source_times, -np.unwrap(np.angle(h), axis=0), axis=0, extrapolate=False)
    orbit = params.get('orbit', 'analytic')
    if isinstance(orbit, str):
        if orbit == 'analytic':
            orbit = AnalyticOrbits()
        elif orbit in ('oem', 'esa'):
            from lisaorbits import OEMOrbits
            from tdispline.numpy.orbits import OrbitInterpolant
            orbit = OrbitInterpolant(OEMOrbits.from_included('esa-trailing'), npoints=100000)
        else:
            raise ValueError('orbit must be analytic, oem, or an equatorial orbit object')
    if hasattr(orbit, 't') and (orbit_times[0]-padding < orbit.t[0] or
                              orbit_times[-1]+padding > orbit.t[-1]):
        raise ValueError('Orbit coverage excludes the padded detector window; set data_time_origin')
    response = TDISpline(orbit, combination=X2 if version == '2' else X1)
    response.set_extrinsic_parameters(ra, dec)
    nodes = int(params.get('response_nodes', 4096))
    if nodes < 2:
        raise ValueError('response_nodes must be at least two')
    logger.debug('PhenomTHM: %d source samples, %d response nodes, epoch=%g, f_ref=%g',
                 len(source_times), nodes, times[0], f_ref)
    xyz = response(np.linspace(orbit_times[0], orbit_times[-1], min(nodes, len(times))),
                   amplitude, phase, nmodes=len(modes), new_t=orbit_times).sum(axis=-1)
    data = dict(zip(('LISA_X', 'LISA_Y', 'LISA_Z'), xyz))
    data.update(linear_AET_conversion(*xyz))
    if not all(np.isfinite(data[c]).all() for c in channels):
        raise ValueError('TDI response exceeds source padding or orbit coverage')
    return to_timeseries({c: data[c] for c in channels}, dt, epoch=times[0])
