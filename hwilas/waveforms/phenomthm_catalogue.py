"""Reference-frequency PhenomTHM checks using the same TDI spline orbit models."""
import logging

import numpy as np
from scipy.interpolate import CubicSpline

from hwilas.waveforms.fastbhb_catalogue import (
    AnalyticEquatorialOrbits, FastBHBCatalogueComparison, spherical_harmonic,
)

logger = logging.getLogger(__name__)
MODES = [[2, 2], [2, -2], [2, 1], [2, -1], [3, 3], [3, -3], [4, 4], [4, -4]]


class PhenomTHMCatalogueComparison:
    """Hold time, distance, phase and orientation fixed while changing the response orbit."""
    def __init__(self, reference_generator=None, waveform_nodes=8192, response_nodes=4096, padding=1000.):
        self.reference_generator = reference_generator or FastBHBCatalogueComparison()
        self.waveform_nodes = waveform_nodes
        self.response_nodes = response_nodes
        self.padding = padding

    def responses(self, row, epoch, n, dt, laser_scale=1., reference_frequency=None):
        try:
            from phenomxpy import IMRPhenomTHM
        except ImportError as error:
            raise ImportError('PhenomTHM checks require phenomxpy==2.0.3; install the '
                              'project dependencies in the active Python environment.') from error
        from tdispline.numpy.combinations import X2
        from tdispline.numpy.tdi import TDISpline
        if n < 2 or not np.isfinite(dt) or dt <= 0:
            raise ValueError('PhenomTHM requires at least two samples and a positive cadence')
        f_ref = row['phase_reference_frequency'] if reference_frequency is None else reference_frequency
        if not np.isfinite(f_ref) or f_ref <= 0:
            raise ValueError('Reference frequency must be finite and positive')
        total_mass = row['mass1']+row['mass2']
        eta = row['mass1']*row['mass2']/total_mass**2
        logger.debug('PhenomTHM source %s: f_ref=%g Hz, physical phase=%g', row['source_id'], f_ref, row['coa_phase'])
        phen = IMRPhenomTHM(eta=eta, total_mass=total_mass,
                           s1=[0., 0., row['spin1z']], s2=[0., 0., row['spin2z']],
                           distance=row['distance'], inclination=row['inclination'], phi_ref=row['coa_phase'],
                           f_min=min(1e-5, f_ref), f_ref=f_ref, delta_t=dt, condition=0, mode_array=MODES)
        times = row['data_time_origin']+epoch+np.arange(n)*dt
        merger = row['data_time_origin']+row['tc']
        source_times = np.linspace(times[0]-self.padding, times[-1]+self.padding, self.waveform_nodes)
        modes, _ = phen.compute_hlms(source_times-merger)
        waves = []
        for key, wave in modes.items():
            l, m = (int(key[0]), int(key[1:])) if isinstance(key, str) else key
            waves.append(wave*spherical_harmonic(l, m, phen.pWF.inclination, np.pi/2-phen.pWF.phi_ref))
        h = np.column_stack(waves)*np.exp(-2j*row['polarization'])
        amplitude = CubicSpline(source_times, abs(h), axis=0)
        # TDI spline reconstructs a complex mode with exp(-i phase).
        phase = CubicSpline(source_times, -np.unwrap(np.angle(h), axis=0), axis=0)
        response_times = np.linspace(times[0], times[-1], self.response_nodes)
        reference = self.reference_generator
        if reference.esa_orbits is None:
            from lisaorbits import OEMOrbits
            from tdispline.numpy.orbits import OrbitInterpolant
            reference.esa_orbits = OrbitInterpolant(OEMOrbits.from_included('esa-trailing'), npoints=100000)
        gen, _, start, _ = reference._generator(row, epoch, dt)
        config = dict(gen.wvf_pars['LISAconst'])
        config['Orbitt0'] += start
        orbit_models = {'PhenomTHM analytic + TDI spline': AnalyticEquatorialOrbits(config, row['data_time_origin']),
                        'PhenomTHM ESA + TDI spline': reference.esa_orbits}
        result = {}
        for label, orbit in orbit_models.items():
            tdi = TDISpline(orbit, combination=X2)
            tdi.set_extrinsic_parameters(row['right_ascension'], row['declination'])
            xyz = tdi(response_times, amplitude, phase, nmodes=len(modes), new_t=times).sum(-1)
            result[label] = {channel: x*laser_scale for channel, x in zip(['LISA_A', 'LISA_E', 'LISA_T'],
                             [(xyz[2]-xyz[0])/np.sqrt(2), (xyz[0]-2*xyz[1]+xyz[2])/np.sqrt(6),
                              (xyz[0]+xyz[1]+xyz[2])/np.sqrt(3)])}
        return result
