"""Reference-frequency PhenomHM reconstructions with analytic and ESA responses."""
import logging

import h5py
import numpy as np
from scipy.constants import c
from scipy.interpolate import CubicSpline
from scipy.special import factorial

from hwilas.data_processing.truth import OBLIQUITY_RAD
from hwilas.search.residual_diagnostics import load_catalogue
from hwilas.waveforms.short_fastbhb import MODES

logger = logging.getLogger(__name__)
EQ_TO_ECL = np.array([[1., 0., 0.], [0., np.cos(OBLIQUITY_RAD), np.sin(OBLIQUITY_RAD)],
                     [0., -np.sin(OBLIQUITY_RAD), np.cos(OBLIQUITY_RAD)]])


def load_reference_catalogue(path, time_offset=800.):
    """Retain equatorial angles and the original frequency/time phase anchors."""
    rows = load_catalogue(path, time_offset)
    names = ['ID', 'RightAscension', 'Declination', 'GW22FrequencySSBFrame', 'TimeReferenceSSBFrame']
    with h5py.File(path, 'r') as f:
        group = f['Binaries'] if 'Binaries' in f else f
        missing = [k for k in names if k not in group]
        if missing:
            raise ValueError(f'Catalogue lacks reference parameters: {missing}')
        ids = np.asarray(group['ID'])
        if len(set(ids.tolist())) != len(ids):
            raise ValueError('Catalogue source IDs must be unique')
        original = {int(sid): {k: float(group[k][i]) for k in names} for i, sid in enumerate(ids)}
    for row in rows:
        raw = original[row['source_id']]
        row.update(right_ascension=raw['RightAscension'], declination=raw['Declination'],
                   phase_reference_frequency=raw['GW22FrequencySSBFrame'],
                   data_time_origin=raw['TimeReferenceSSBFrame']+time_offset)
    return rows


def ecliptic_polarization(row):
    """Rotate the polarization basis together with the equatorial sky direction."""
    ra, lam, beta = row['right_ascension'], row['eclipticlongitude'], row['eclipticlatitude']
    u_eq = np.array([np.sin(ra), -np.cos(ra), 0.])
    u_ecl = np.array([np.sin(lam), -np.cos(lam), 0.])
    v_ecl = np.array([-np.sin(beta)*np.cos(lam), -np.sin(beta)*np.sin(lam), np.cos(beta)])
    rotated = EQ_TO_ECL @ u_eq
    angle = np.arctan2(rotated @ v_ecl, rotated @ u_ecl)
    return float((row['polarization']+angle) % np.pi)


def spherical_harmonic(l, m, theta, phi):
    """Spin-weight minus two harmonics from the finite Wigner small-d sum."""
    mp = 2
    prefactor = np.sqrt(factorial(l+m)*factorial(l-m)*factorial(l+mp)*factorial(l-mp))
    value = 0.
    for k in range(max(0, m-mp), min(l+m, l-mp)+1):
        value += ((-1.)**k*prefactor /
                  (factorial(l+m-k)*factorial(k)*factorial(mp-m+k)*factorial(l-mp-k)) *
                  np.cos(theta/2)**(2*l+m-mp-2*k)*np.sin(theta/2)**(mp-m+2*k))
    return np.sqrt((2*l+1)/(4*np.pi))*value*np.exp(1j*m*phi)


class AnalyticEquatorialOrbits:
    """Expose the native FastBHB cartwheel in the equatorial TDI spline basis."""
    def __init__(self, config, data_origin):
        from lisabeta.lisa.pyresponse import PythonOrbit
        self.orbit = PythonOrbit(config)
        self.data_origin = data_origin

    def get_positions(self, times):
        orbit = self.orbit.eval_orbit(np.asarray(times)-self.data_origin)
        return np.stack([orbit['p'][i] @ EQ_TO_ECL for i in [1, 2, 3]], axis=1)

    def get_ltts(self, times):
        return np.full((len(times), 6), self.orbit.get_L()/c)


class FastBHBCatalogueComparison:
    """Keep PhenomHM intrinsic modes fixed while comparing orbit/response methods."""
    def __init__(self, f_lower=1e-5, f_upper=.09, esa_orbits=None, source_oversampling=4):
        self.f_lower, self.f_upper = f_lower, f_upper
        self.esa_orbits = esa_orbits
        if source_oversampling < 1 or int(source_oversampling) != source_oversampling:
            raise ValueError('Source oversampling must be a positive integer')
        self.source_oversampling = int(source_oversampling)

    def _generator(self, row, epoch, dt):
        import astropy.units as u
        from ldc.lisa import orbits
        from ldc.waveform.lisabeta import FastBHB
        from ldc.waveform.lisabeta.fast_bhb import ASTRONOMICAL_YEAR
        from scipy.fft import next_fast_len
        start = epoch+np.floor((row['tc']-2*86400.-epoch)/dt)*dt
        size = next_fast_len(round(4*86400./dt))
        if size % 2:
            size = next_fast_len(size+1)
        position = -140.-360*start/ASTRONOMICAL_YEAR
        orbit = orbits.Orbits.type(dict(nominal_arm_length=2.5e9*u.m, initial_rotation=0*u.rad,
                                  initial_position=np.radians(position)*u.rad, orbit_type='analytic'))
        generator = FastBHB('MBHB', approx='IMRPhenomHM', T=size*dt, delta_t=dt,
                            orbits=orbit, modes=MODES)
        generator.wvf_pars.update(minf=self.f_lower, maxf=min(self.f_upper, .5/dt),
                                  timetomerger_max=None, tmin=None, tmax=None,
                                  fref_for_phiref=row['phase_reference_frequency'], phiref=0.)
        # FastBHB's phi is observer azimuth; the catalogue reports reference orbital phase.
        parameters = dict(Mass1=row['mass1'], Mass2=row['mass2'], Spin1=row['spin1z'], Spin2=row['spin2z'],
                          Distance=row['distance'], Inclination=row['inclination'],
                          EclipticLongitude=row['eclipticlongitude'], EclipticLatitude=row['eclipticlatitude'],
                          Polarization=ecliptic_polarization(row), PhaseAtCoalescence=np.pi/2-row['coa_phase'],
                          CoalescenceTime=row['tc']-start, Cadence=dt)
        logger.debug('FastBHB source %s: f_ref=%g, observer_phi=%g, ecliptic_psi=%g',
                     row['source_id'], row['phase_reference_frequency'], parameters['PhaseAtCoalescence'],
                     parameters['Polarization'])
        return generator, parameters, start, size

    def native(self, row, epoch, n, dt, laser_scale=1.):
        generator, parameters, start, size = self._generator(row, epoch, dt)
        fd = generator.get_fd_tdiaet(template=parameters, freqs=np.fft.rfftfreq(size, dt), tdi2=True)
        offset = round((epoch-start)/dt)
        if offset < 0 or offset+n > size:
            raise ValueError('Requested window is outside the four-day generation grid')
        return {channel: np.fft.irfft(np.asarray(x), n=size)[offset:offset+n]/dt*laser_scale
                for channel, x in zip(['LISA_A', 'LISA_E', 'LISA_T'], fd)}

    def responses(self, row, epoch, n, dt, laser_scale=1.):
        from tdispline.numpy.combinations import X2
        from tdispline.numpy.tdi import TDISpline
        # Resolve retarded-time interpolation more finely than the displayed detector grid.
        source_dt = dt/self.source_oversampling
        generator, parameters, start, size = self._generator(row, epoch, source_dt)
        generator.wvf_pars['output_response'] = False
        source = generator.get_full_waveform(template=parameters)
        frequencies = np.fft.rfftfreq(size, source_dt)
        waves = []
        for l, m in source['modes']:
            mode = source[(l, m)]
            valid = (frequencies >= mode['freq'][0]) & (frequencies <= mode['freq'][-1])
            spectrum = np.zeros(size, dtype=complex)
            spectrum[:len(frequencies)][valid] = (CubicSpline(mode['freq'], mode['amp'])(frequencies[valid]) *
                                                  np.exp(1j*CubicSpline(mode['freq'], mode['phase'])(frequencies[valid])))
            # Lisabeta uses the conjugate Fourier convention; its positive modes have negative TD phase.
            mode_td = np.fft.fft(spectrum)/size/source_dt
            azimuth = np.pi/2-row['coa_phase']
            waves.extend([mode_td*spherical_harmonic(l, m, row['inclination'], azimuth),
                          (-1)**l*np.conj(mode_td)*spherical_harmonic(l, -m, row['inclination'], azimuth)])
        h = np.column_stack(waves)*np.exp(-2j*row['polarization'])
        source_times = row['data_time_origin']+start+np.arange(size)*source_dt
        amplitude = CubicSpline(source_times, abs(h), axis=0)
        phase = CubicSpline(source_times, -np.unwrap(np.angle(h), axis=0), axis=0)
        times = row['data_time_origin']+epoch+np.arange(n)*dt
        if times[0]-1000 < source_times[0] or times[-1]+1000 > source_times[-1]:
            raise ValueError('Orbit comparison needs 1000 seconds of source-grid padding at each end')
        if self.esa_orbits is None:
            from lisaorbits import OEMOrbits
            from tdispline.numpy.orbits import OrbitInterpolant
            self.esa_orbits = OrbitInterpolant(OEMOrbits.from_included('esa-trailing'), npoints=100000)
        config = dict(generator.wvf_pars['LISAconst'])
        config['Orbitt0'] += start
        orbit_models = {'Analytic + TDI spline': AnalyticEquatorialOrbits(config, row['data_time_origin']),
                        'ESA + TDI spline': self.esa_orbits}
        result = {}
        for label, orbit in orbit_models.items():
            tdi = TDISpline(orbit, combination=X2)
            tdi.set_extrinsic_parameters(row['right_ascension'], row['declination'])
            xyz = tdi(times, amplitude, phase, nmodes=h.shape[1], new_t=times).sum(-1)
            result[label] = {channel: x*laser_scale for channel, x in zip(['LISA_A', 'LISA_E', 'LISA_T'],
                             [(xyz[2]-xyz[0])/np.sqrt(2), (xyz[0]-2*xyz[1]+xyz[2])/np.sqrt(6),
                              (xyz[0]+xyz[1]+xyz[2])/np.sqrt(3)])}
        return result
