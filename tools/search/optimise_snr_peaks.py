"""
Optimise the physical-fit statistic for selected search candidates.
"""

import os
# Set before numerical imports initialise the OpenMP runtime or spawn workers.
os.environ.setdefault("KMP_WARNINGS", "FALSE")

import argparse
import copy
import hashlib
import json
from pathlib import Path
from functools import partial
from functools import lru_cache
from concurrent.futures import ProcessPoolExecutor
from multiprocessing import get_context
import logging
from hwilas.search.statistics import read_statistic
import h5py
import numpy as np
import pyswarms as ps
from tqdm.auto import tqdm

from gwaith.types import TimeSeries, FrequencySeries
from scipy.fft import next_fast_len
from scipy.optimize import minimize, differential_evolution
from scipy.signal import detrend
from scipy.signal.windows import tukey
from hwilas.waveforms.short_fastbhb import template
from hwilas.template_banks.frequency import lower_frequency_for_duration

from hwilas import add_hwilas_common_options, init_logging
from hwilas.conversions import (mchirp_eta_from_mass1_mass2, mass1_mass2_from_mchirp_eta,
                                chi_eff_from_component_spins)
from hwilas.filter import matched_filter_frequencyseries, sigmasq_frequencyseries
from hwilas.psd.read import from_txt as psd_from_txt
# Higher-mode (l, m) array used by the follow-up optimisation to build its
# higher-mode template.
DEFAULT_HM_MODE_ARRAY = [(2, 2), (2, 1), (3, 3), (3, 2), (4, 4), (4, 3)]
DEFAULT_THM_MODE_ARRAY = [(2, 2), (2, -2), (2, 1), (2, -1), (3, 3), (3, -3), (4, 4), (4, -4)]
from hwilas.types import KeyValueAction
from hwilas.search.rounds import previous_window_flags
from hwilas.search.sky_modes import sky_mode_vectors
from hwilas.search.partitions import event_partition, parse_fraction

logger = logging.getLogger(__name__)


PARAMETERS = ('mchirp', 'eta', 'spin1z', 'spin2z', 'eclipticlongitude',
              'eclipticlatitude', 'inclination', 'tc', 'polarization', 'coa_phase')


from hwilas.waveforms.cli import parse_approximant
from hwilas.waveforms.plugins import external_generator, reference_frequency, prepare_generator


def final_waveform_model(args):
    return getattr(args, 'refinement_waveform_model', None) or args.waveform_model


def stage_settings(args, refinement=False):
    """Resolve each refinement setting independently from the optimisation stage."""
    settings = dict(waveform_model=args.waveform_model, orbit=getattr(args, 'orbit', 'analytic'),
                    f_ref=getattr(args, 'f_ref', args.f_lower),
                    generator_keywords=dict(getattr(args, 'generator_keywords', {})))
    if refinement:
        for key, attribute in [('waveform_model','refinement_waveform_model'),
                               ('orbit','refinement_orbit'), ('f_ref','refinement_f_ref')]:
            value = getattr(args, attribute, None)
            if value is not None:
                settings[key] = value
        settings['generator_keywords'].update(getattr(args, 'refinement_generator_keywords', None) or {})
    prepared = getattr(args, 'prepared_generator_keywords', {})
    settings['generator_keywords'].update(prepared.get('refinement' if refinement else 'optimisation', {}))
    return settings


def reseed_refinement(candidates, kwargs, lower, upper, phase_count):
    """Compare transferred seeds using only the final model's physical statistic."""
    best_stat, best = -np.inf, None
    physical_kwargs = {**kwargs, 'logarithmic_mass': False}
    phases = np.linspace(lower[9], upper[9], phase_count, endpoint=False)
    for seed in candidates:
        for phase in [seed[9], *phases]:
            candidate = np.asarray(seed).copy()
            candidate[9] = phase
            statistic, physical = profile_physical_fit(candidate, **physical_kwargs)
            if statistic > best_stat:
                best_stat, best = statistic, physical
    if best is None or not np.isfinite(best_stat) or best_stat <= 0:
        raise ValueError('Transferred seeds have no positive final-model fit')
    return best_stat, best


def generator_keyword_value(value):
    """Accept JSON numbers, booleans and arrays, or an unquoted string."""
    try:
        return json.loads(value)
    except json.JSONDecodeError:
        return value


def _de_objective(position, objective_kwargs):
    """Scalar physical-fit objective, pickleable for spawned DE workers."""
    return -compute_statistic_core(position, **objective_kwargs)


class CircularStar(ps.backend.topology.Star):
    """Global-best PSO with circular angles and reflecting physical bounds."""

    def __init__(self, parameters, lower, upper):
        super().__init__()
        periods = {'eclipticlongitude': 2*np.pi, 'polarization': np.pi,
                   'coa_phase': 2*np.pi}
        # A restricted angular prior is an interval rather than a full circle.
        self.circular = np.array([name in periods and np.isclose(
            upper[i]-lower[i], periods[name], rtol=1e-9, atol=1e-10)
            for i, name in enumerate(parameters)])

    def displacement(self, target, position, bounds):
        difference = np.asarray(target)-position
        width = bounds[1]-bounds[0]
        difference[..., self.circular] = (
            difference[..., self.circular]+width[self.circular]/2
            ) % width[self.circular]-width[self.circular]/2
        return difference

    def compute_velocity(self, swarm, clamp=None, vh=None, bounds=None):
        cognitive = self.displacement(swarm.pbest_pos, swarm.position, bounds)
        social = self.displacement(swarm.best_pos, swarm.position, bounds)
        velocity = (swarm.options['w']*swarm.velocity +
            swarm.options['c1']*np.random.uniform(size=swarm.position.shape)*cognitive +
            swarm.options['c2']*np.random.uniform(size=swarm.position.shape)*social)
        return vh(velocity, clamp, position=swarm.position, bounds=bounds)

    def compute_position(self, swarm, bounds=None, bh=None):
        lower, upper = bounds
        width = upper-lower
        position = swarm.position+swarm.velocity
        # Reflect nonperiodic coordinates even when a step crosses several widths.
        folded = (position-lower) % (2*width)
        result = lower+np.where(folded <= width, folded, 2*width-folded)
        result[..., self.circular] = lower[self.circular]+(
            position[..., self.circular]-lower[self.circular]) % width[self.circular]
        return result


class CircularDEStrategy:
    """Differential mutation in physical coordinates with periodic angular seams."""

    def __init__(self, parameters, lower, upper, strategy, mutation, recombination):
        self.geometry = CircularStar(parameters, lower, upper)
        self.bounds = (np.asarray(lower), np.asarray(upper))
        self.strategy = strategy
        self.mutation = mutation
        self.recombination = recombination
        self.scale = None

    def __call__(self, candidate, population, rng=None):
        # SciPy calls candidates in order; use one dithered scale per generation.
        if candidate == 0 or self.scale is None:
            self.scale = rng.uniform(*self.mutation)
        indices = np.delete(np.arange(len(population)), candidate)
        rng.shuffle(indices)
        r = population[indices[:5]]
        current, best = population[candidate], population[0]
        difference = lambda a, b: self.geometry.displacement(a, b, self.bounds)
        kind = self.strategy[:-3]
        if kind == 'best1':
            mutant = best+self.scale*difference(r[0], r[1])
        elif kind == 'rand1':
            mutant = r[0]+self.scale*difference(r[1], r[2])
        elif kind == 'best2':
            mutant = best+self.scale*(difference(r[0], r[2])+difference(r[1], r[3]))
        elif kind == 'rand2':
            mutant = r[0]+self.scale*(difference(r[1], r[3])+difference(r[2], r[4]))
        elif kind == 'randtobest1':
            mutant = r[0]+self.scale*(difference(best, r[0])+difference(r[1], r[2]))
        elif kind == 'currenttobest1':
            mutant = current+self.scale*(difference(best, current)+difference(r[0], r[1]))
        else:
            raise ValueError(f'Unsupported DE strategy: {self.strategy}')
        dimensions = population.shape[1]
        fill = int(rng.integers(dimensions) if hasattr(rng, 'integers') else rng.randint(dimensions))
        crossovers = rng.uniform(size=dimensions) < self.recombination
        if self.strategy.endswith('bin'):
            crossovers[fill] = True
            trial = np.where(crossovers, mutant, current)
        else:
            trial = current.copy()
            crossovers[0] = True
            for offset in range(dimensions):
                if not crossovers[offset]:
                    break
                index = (fill+offset) % dimensions
                trial[index] = mutant[index]
        lower, upper = self.bounds
        circular = self.geometry.circular
        trial[circular] = lower[circular]+(trial[circular]-lower[circular]) % (upper-lower)[circular]
        # SciPy retains its usual random repair for out-of-bounds nonangular values.
        return trial


def optimise_population(args, population, lower, upper, kwargs, event_index):
    """Run the selected population optimiser with the shared physical objective."""
    sweep_every = getattr(args, 'sky_sweep_every', 0)
    def sweep(position, cost, pool=None):
        branches = sky_mode_vectors(position, kwargs['tc_reference'])
        valid = np.all((branches >= lower) & (branches <= upper), axis=1)
        branches = branches[valid]
        objective = partial(_de_objective, objective_kwargs=kwargs)
        costs = np.asarray(list(map(objective, branches)) if pool is None else pool.map(objective, branches))
        winner = int(np.argmin(costs)) if len(costs) else None
        if winner is not None and np.isfinite(costs[winner]) and costs[winner] < cost:
            logger.info('Event %d sky-mode sweep improved statistic %.6g -> %.6g',
                        event_index,-cost,-costs[winner])
            return float(costs[winner]), branches[winner].copy()
        return cost, np.asarray(position).copy()
    if sweep_every:
        branches = sky_mode_vectors(population[0], kwargs['tc_reference'])
        branches = branches[np.all((branches >= lower) & (branches <= upper), axis=1)]
        count = min(len(branches), len(population))
        if count:
            selected = np.linspace(0, len(branches)-1, count, dtype=int)
            population[:count] = branches[selected]
    if args.optimizer == 'pso':
        np.random.seed(args.seed+event_index)
        optimiser = ps.single.GlobalBestPSO(n_particles=args.n_particles, dimensions=len(lower),
            options={'c1':.5, 'c2':.3, 'w':.9}, bounds=(lower, upper), init_pos=population)
        parameters = [name for name in PARAMETERS if name != 'tc']
        optimiser.top = CircularStar(parameters, lower, upper)
        logger.debug('PSO circular coordinates: %s',
                     [name for name, circular in zip(parameters, optimiser.top.circular) if circular])
        optimiser.name = f'Event {event_index} PSO'
        if sweep_every:
            original_hook = optimiser.rep.hook
            iteration = 0
            def sky_hook(**values):
                nonlocal iteration
                iteration += 1
                if iteration % sweep_every == 0 or iteration == args.n_iterations:
                    swarm = optimiser.swarm
                    cost, position = sweep(swarm.best_pos, swarm.best_cost)
                    if cost < swarm.best_cost:
                        index = int(np.argmin(swarm.pbest_cost))
                        swarm.pbest_pos[index], swarm.pbest_cost[index] = position, cost
                        swarm.position[index], swarm.current_cost[index] = position, cost
                        swarm.velocity[index] = 0.
                        swarm.best_pos, swarm.best_cost = position.copy(), np.float64(cost)
                    values['best_cost'] = swarm.best_cost
                return original_hook(**values)
            optimiser.rep.hook = sky_hook
        return optimiser.optimize(compute_minus_statistic, iters=args.n_iterations,
            n_processes=args.n_processes if args.n_processes>1 else None, verbose=True, **kwargs)
    objective = partial(_de_objective, objective_kwargs=kwargs)
    options = dict(bounds=list(zip(lower, upper)), init=population,
        maxiter=args.n_iterations, seed=args.seed+event_index, strategy=CircularDEStrategy(
            [name for name in PARAMETERS if name != 'tc'][:len(lower)], lower, upper,
            args.de_strategy, args.de_mutation, args.de_recombination),
        mutation=tuple(args.de_mutation), recombination=args.de_recombination,
        tol=args.de_tol, atol=args.de_atol, polish=False, updating='deferred')
    # Both serial and parallel DE update once per generation for repeatability.
    with tqdm(total=args.n_iterations, desc=f'Event {event_index} DE', unit='generation') as progress:
        def update_progress(intermediate_result):
            progress.update(1)
            progress.set_postfix(statistic=-intermediate_result.fun)
        options['callback'] = update_progress
        def run_de(pool=None):
            remaining, iteration, evaluations = args.n_iterations, 0, 0
            settings = options.copy()
            # Retain the population between sweeps; share one RNG stream across blocks.
            settings['seed'] = np.random.default_rng(args.seed+event_index)
            while remaining:
                settings['maxiter'] = min(sweep_every, remaining) if sweep_every else remaining
                result = differential_evolution(objective, workers=pool.map if pool else 1, **settings)
                evaluations += result.nfev
                remaining -= result.nit
                iteration += result.nit
                if sweep_every:
                    cost, position = sweep(result.x, result.fun, pool)
                    improved = cost < result.fun
                    result.fun, result.x = cost, position
                    settings['init'] = result.population.copy()
                    settings['init'][np.argmax(result.population_energies)] = position
                else:
                    improved = False
                if result.nit == 0 or (result.success and not improved):
                    break
            result.nit = iteration
            result.nfev = evaluations
            return result
        if args.n_processes > 1:
            with get_context('spawn').Pool(args.n_processes) as pool:
                result = run_de(pool)
        else:
            result = run_de()
    logger.debug('DE event %d: generations=%d evaluations=%d converged=%s; %s',
                 event_index, result.nit, result.nfev, result.success, result.message)
    return result.fun, result.x


def _data_frequencyseries(data):
    """Continuous Fourier convention, independent of installed GWAITH version."""
    return {c: FrequencySeries(ts.delta_t*np.fft.rfft(ts.data),
                              delta_f=1/(len(ts)*ts.delta_t), epoch=ts.epoch)
            for c, ts in data.items()}


class OEMInterpolationNotice(logging.Filter):
    """Keep the expected OEM spline notice available at debug verbosity."""

    def filter(self, record):
        if record.getMessage() == (
            'OEM preferred interpolation method ignored, using spline '
            'interpolation (see InterpolatedOrbits for details)'):
            record.levelno, record.levelname = logging.DEBUG, 'DEBUG'
            return logging.getLogger(record.name).isEnabledFor(logging.DEBUG)
        return True


@lru_cache(maxsize=2)
def _thm_orbit(kind):
    """Reuse the immutable equatorial orbit interpolation in each worker process."""
    if kind == 'analytic':
        from hwilas.waveforms.phenomthm import AnalyticOrbits
        return AnalyticOrbits()
    from lisaorbits import OEMOrbits
    from tdispline.numpy.orbits import OrbitInterpolant
    orbit_logger = logging.getLogger('lisaorbits.oem')
    if not any(isinstance(item, OEMInterpolationNotice) for item in orbit_logger.filters):
        orbit_logger.addFilter(OEMInterpolationNotice())
    return OrbitInterpolant(OEMOrbits.from_included('esa-trailing'), npoints=100000)


def build_aligned_waveform(v, data, waveform_params_shared, raise_err=False, waveform_channels=None):
    """Generate the physical short waveform aligned to the event data grid."""
    first = next(iter(data.values()))
    channels = waveform_channels or list(data)
    try:
        m1, m2 = mass1_mass2_from_mchirp_eta(v[0], v[1])
        p = dict(mass1=m1, mass2=m2, spin1z=v[2], spin2z=v[3],
                 eclipticlongitude=v[4], eclipticlatitude=v[5], inclination=v[6],
                 tc=v[7], polarization=v[8],
                 coa_phase=v[9] if len(v)>9 else waveform_params_shared.get('coa_phase',0.),
                 distance=waveform_params_shared.get('distance',1e4))
        if not first.epoch < p['tc'] < first.epoch+len(first)*first.delta_t:
            raise ValueError("Merger lies outside the event window")
        settings = dict(delta_t=first.delta_t, initial_position_deg=-140.,
                        laser_frequency=waveform_params_shared.get('template_scale',1.),
                        f_lower=waveform_params_shared['f_lower'],
                        f_upper=waveform_params_shared.get('f_upper',.09),
                        pre_merger_days=1., taper_seconds=3600.,
                        tdi=waveform_params_shared.get('tdi','2'), channels=channels,
                        reference_tc=p['tc']-float(first.epoch),
                        f_ref=waveform_params_shared.get('f_ref'))
        model = waveform_params_shared.get('waveform_model', 'fastbhb').lower()
        if model == 'phenomthm':
            from hwilas.waveforms.phenomthm import get_td_waveform_from_phenomthm
            from scipy.signal import resample_poly
            cutoff = float(settings['f_lower'])
            # Generate only the local support, then restore the fixed event FFT grid.
            dt = first.delta_t
            left = max(0, int(np.floor((p['tc']-86400.-float(first.epoch))/dt)))
            right = min(len(first), int(np.ceil((p['tc']+86400.-float(first.epoch))/dt)))
            oversampling = int(waveform_params_shared.get('thm_oversampling', 4))
            origin = float(waveform_params_shared.get('data_time_origin', 0.))
            generation = dict(p, tdi=settings['tdi'],
                start_time=float(first.epoch)+left*dt, delta_t=dt/oversampling,
                length=(right-left)*oversampling, ifos=channels,
                f_lower=cutoff, f_ref=float(waveform_params_shared.get('f_ref', cutoff)),
                orbit=_thm_orbit(waveform_params_shared.get('orbit', 'esa')),
                data_time_origin=origin, source_oversampling=1,
                response_nodes=int(waveform_params_shared.get('thm_response_nodes', 16384)),
                mode_array=DEFAULT_THM_MODE_ARRAY)
            generation.update(waveform_params_shared.get('generator_keywords', {}))
            logger.debug('THM generator keywords: %s', waveform_params_shared.get('generator_keywords', {}))
            raw = get_td_waveform_from_phenomthm('PhenomTHM', **generation)
            tm = {c: np.zeros(len(first)) for c in channels}
            for c in channels:
                # Anti-alias before returning to the detector cadence.
                tm[c][left:right] = resample_poly(raw[c].data, 1, oversampling)*settings['laser_frequency']
        elif model == 'fastbhb':
            cutoff = lower_frequency_for_duration(p,settings)
            settings['f_lower'] = cutoff
            # Translate orbit and time together; every candidate uses the same FFT grid.
            settings['generator_keywords'] = waveform_params_shared.get('generator_keywords', {})
            tm = template(p,float(first.epoch),len(first),settings,modes=DEFAULT_HM_MODE_ARRAY)
        else:
            cutoff = settings['f_lower']
            generation = dict(p, delta_t=first.delta_t, start_time=float(first.epoch),
                length=len(first), ifos=channels, f_lower=cutoff, f_upper=settings['f_upper'],
                f_ref=waveform_params_shared.get('f_ref', 0.), orbit=waveform_params_shared.get('orbit','analytic'),
                data_time_origin=waveform_params_shared.get('data_time_origin'))
            generation.update(waveform_params_shared.get('generator_keywords', {}))
            raw = external_generator(model)(approximant=model, tdi=settings['tdi'], **generation)
            tm = {c:np.asarray(raw[c].data)*settings['laser_frequency'] for c in channels}
            if any(len(raw[c]) != len(first) or not np.isclose(raw[c].delta_t, first.delta_t)
                   or not np.isclose(float(raw[c].epoch),float(first.epoch),rtol=0.,atol=1e-6) for c in channels):
                raise ValueError('External waveform grid differs from the event grid')
        t = float(first.epoch)+np.arange(len(first))*first.delta_t
        ramp = .5-.5*np.cos(np.pi*np.clip((t-(p['tc']-86400.))/3600.,0,1))
        result = {}
        for c in channels:
            h = first.delta_t*np.fft.rfft(tm[c]*ramp)
            h[:int(cutoff/(1/(len(first)*first.delta_t)))] = 0
            if model != 'fastbhb':
                h[np.fft.rfftfreq(len(first), first.delta_t) > settings['f_upper']] = 0
            result[c] = FrequencySeries(h,delta_f=1/(len(first)*first.delta_t),epoch=first.epoch)
        return result
    except (ValueError, RuntimeError, SystemError, FloatingPointError):
        if raise_err:
            raise
        return None


def coherent_fit(wf, data_fs, psd, channels):
    """Solve a positive real amplitude, equivalent to a physical distance."""
    numerator, norm = 0., 0.
    for c in channels:
        H, D = wf[c].data, data_fs[c].data
        S = psd[c].data if isinstance(psd,dict) else psd.data
        if len(H)!=len(D) or len(H)!=len(S):
            raise ValueError('Waveform, data and PSD grids must agree')
        valid = np.isfinite(S) & (S>0)
        weight = 4*wf[c].delta_f
        numerator += weight*np.sum(np.conj(H[valid])*D[valid]/S[valid]).real
        norm += weight*np.sum(abs(H[valid])**2/S[valid])
    if not np.isfinite(norm) or norm<=0:
        return -np.inf, 0.
    numerator = max(0.,float(numerator))
    return numerator/np.sqrt(norm), numerator/norm


def profile_merger_time(wf, data_fs, psd, channels, n, dt, tc, bounds):
    """Maximise the real physical-template correlation within the time bounds."""
    q = np.zeros(n//2+1, dtype=complex)
    norm = 0.
    df = wf[channels[0]].delta_f
    for c in channels:
        H,D,S = wf[c].data,data_fs[c].data,psd[c].data
        valid = np.isfinite(S) & (S>0)
        q[valid] += np.conj(H[valid])*D[valid]/S[valid]
        norm += 4*df*np.sum(abs(H[valid])**2/S[valid])
    if not np.isfinite(norm) or norm<=0:
        return -np.inf,tc
    correlation = np.fft.irfft(q,n=n)*n*2*df/np.sqrt(norm)
    lower = int(np.ceil((bounds[0]-tc)/dt))
    upper = int(np.floor((bounds[1]-tc)/dt))
    lags = np.arange(lower,upper+1)
    if not len(lags):
        return -np.inf,tc
    index = int(lags[np.argmax(correlation[lags % n])])
    left,centre,right = correlation[(index+np.array([-1,0,1])) % n]
    curvature = left-2*centre+right
    offset = np.clip(.5*(left-right)/curvature,-.5,.5) if curvature<0 else 0.
    time = float(np.clip(tc+(index+offset)*dt,*bounds))
    peak = centre-.25*(left-right)*offset
    return max(0.,float(peak)),time


def profile_physical_fit(v, data, psd, waveform_params_shared, optimisation_channels,
                         tc_reference, time_bounds, raise_err=False, logarithmic_mass=False):
    """Profile merger time while the swarm searches the physical shape parameters."""
    full = np.insert(v,7,tc_reference) if len(v)==9 else np.asarray(v).copy()
    if logarithmic_mass:
        full[0] = 10.**full[0]
    wf = build_aligned_waveform(full,data,waveform_params_shared,raise_err)
    if wf is None:
        return -np.inf,full
    first = next(iter(data.values()))
    data_fs = waveform_params_shared.get('data_fs') or _data_frequencyseries(data)
    fit_stat,full[7] = profile_merger_time(wf,data_fs,psd,optimisation_channels,
                                    len(first),first.delta_t,full[7],time_bounds)
    return fit_stat,full


def compute_statistic_core(v, data, psd, waveform_params_shared, optimisation_channels,
                             raise_err=False, tc_reference=None, time_bounds=None, logarithmic_mass=False):
    if time_bounds is not None:
        return profile_physical_fit(v,data,psd,waveform_params_shared,optimisation_channels,
                                    tc_reference,time_bounds,raise_err,logarithmic_mass)[0]
    wf = build_aligned_waveform(v,data,waveform_params_shared,raise_err)
    if wf is None:
        return -np.inf
    data_fs = waveform_params_shared.get('data_fs') or _data_frequencyseries(data)
    return coherent_fit(wf,data_fs,psd,optimisation_channels)[0]


def get_statistic_and_distance_factor(wf, data, psd, optimisation_channels):
    fit_stat, amp = coherent_fit(wf,_data_frequencyseries(data),psd,optimisation_channels)
    return fit_stat, 1/abs(amp) if abs(amp)>0 else np.inf


def shift_and_normalise_waveform(wf_fd, coalescence_in_segment, data, psd,
                                 optimisation_channels, delta_t, output_channels):
    """Normalise a cyclic-zero-merger FD template for plotting."""
    first = next(iter(data.values()))
    aligned = {c:wf_fd[c].cyclic_time_shift(coalescence_in_segment) for c in output_channels}
    fit_stat, amp = coherent_fit(aligned,_data_frequencyseries(data),psd,optimisation_channels)
    out = {c:TimeSeries(np.fft.irfft(amp*aligned[c].data,n=len(first))/delta_t,
                       epoch=first.epoch,delta_t=delta_t) for c in output_channels}
    return out,fit_stat,1/abs(amp) if abs(amp)>0 else np.inf


def compute_minus_statistic(positions, **kwargs):
    return np.array([-compute_statistic_core(v,**kwargs) for v in positions])


def load_arguments():
    parser = argparse.ArgumentParser()
    add_hwilas_common_options(parser)
    parser.add_argument(
        "--input-file",
        required=True,
        help="File containing coincident events"
    )
    parser.add_argument(
        "--statistic-threshold",
        type=float,
        default=40,
        help="When deciding the events to optimise / add to signal timeseries, use this threshold. "
        "Default 40."
    )
    parser.add_argument(
        "--minimum-peak-separation",
        type=float,
        default=86400,
        help="Minimum separation of peaks to be optimised. Seconds. Default 86400 (1 day)"
    )
    parser.add_argument(
        "--bank-file",
        required=True
    )
    parser.add_argument(
        "--output-timeseries-file",
        help="File containing the timeseries for all optimised events"
    )
    parser.add_argument(
        "--output-parameter-file",
        help="File containing the optimised event parameters"
    )
    parser.add_argument(
        "--psd-file",
        help="File containing the PSD"
    )
    parser.add_argument(
        "--data-files",
        nargs='+',
        required=True,
        help="File(s) containing the data"
    )
    parser.add_argument(
        "--optimisation-channels",
        nargs='+',
        default=['LISA_A', 'LISA_E'],
        help="Channels that will be used for optimisation. "
        "Default LISA_A and LISA_E."
    )
    parser.add_argument("--fit-channels",nargs='+',
                        help="Channels constraining the physical fit; defaults to the coincidence channels")
    parser.add_argument(
        "--output-timeseries-channels",
        nargs='+',
        default=['LISA_A', 'LISA_E', 'LISA_T'],
        help="Channels predicted from the fitted physical parameters. "
        "Default LISA_A, LISA_E and LISA_T."
    )
    parser.add_argument(
        "--f-lower",
        type=float,
        default=1e-6,
    )
    parser.add_argument(
        "--optimisation-window-before",
        type=float,
        default=30 * 86400,
        help="Time in seconds before merger to include in optimisation window. "
        "Default 30 days."
    )
    parser.add_argument(
        "--optimisation-window-after",
        type=float,
        default=30 * 86400,
        help="Time in seconds after merger to include in optimisation window. "
        "Default 30 days."
    )
    parser.add_argument(
        "--n-particles", "--population-size",
        type=int,
        default=100,
        help="Actual population size for PSO or DE (not a per-dimension multiplier)"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=606,
        help="Random seed for the initial population passed to the optimiser"
    )
    parser.add_argument(
        "--bounds-window-relative-to-initial",
        default={},
        nargs='+',
        action=KeyValueAction,
        type=float,
        help="A parameter:window_size paired list of bounds to use for the optimiser. "
        "e.g. mchirp:100 will give an mchirp window of 100 Msol either side of the template."
    )
    parser.add_argument(
        "--maximum-bounds",
        nargs='+',
        action=KeyValueAction,
        type=float,
        default={},
        help="Set maximum bounds on these parameters according to given parameters. "
        "Given in parameter:maximum_bound pairs e..g. mchirp:5e7. This will be set as the bound if "
        "--bounds-window-relative-to-initial is not given, or if the template plus the bounds "
        "is larger than this value"
    )
    parser.add_argument(
        "--minimum-bounds",
        nargs='+',
        action=KeyValueAction,
        type=float,
        default={},
        help="Set minimum bounds on these parameters according to given parameters. "
        "Given in parameter:minimum_bound pairs e.g. mchirp:5e5. This will be set as the bound if "
        "--bounds-window-relative-to-initial is not given, or if the template minus the bounds "
        "is smaller than this value"
    )

    parser.add_argument(
        '--tdi-generation',
        choices=['1.5','2'],
        default="1.5",
        help="Which generation of TDI to use when generating the waveform"
    )

    parser.add_argument("--psd-files",nargs='+',action=KeyValueAction,
                        help="Channel-specific PSDs as channel:filename pairs")
    parser.add_argument("--f-upper",type=float,default=.09)
    parser.add_argument('--approximant', '--waveform-model', dest='waveform_model', type=parse_approximant,
                        default='fastbhb', help='Physical follow-up waveform; detection bank is unchanged')
    parser.add_argument('--refinement-approximant', '--refinement-waveform-model',
                        dest='refinement_waveform_model', type=parse_approximant,
                        help='Optional final model for phase reseeding, local refinement and output')
    parser.add_argument('--mass-coordinate', choices=['log', 'linear'], default='log',
                        help='Chirp-mass coordinate for swarm sampling and local refinement')
    parser.add_argument('--orbits', '--orbit', dest='orbit', choices=['analytic', 'esa'], default='analytic')
    parser.add_argument('--refinement-orbits', dest='refinement_orbit', choices=['analytic', 'esa'],
                        help='Refinement orbits; defaults to --orbits')
    parser.add_argument('--refinement-f-ref', type=float, help='Refinement phase reference frequency; defaults to --f-ref')
    parser.add_argument('--data-time-origin', type=float,
                        help='Absolute time of data sample zero; otherwise use stored data_time_origin/t0_raw')
    parser.add_argument('--f-ref', type=float, help='Optimisation phase reference frequency in Hz; defaults to f-lower')
    parser.add_argument('--thm-oversampling', type=int, default=4)
    parser.add_argument('--thm-response-nodes', type=int, default=16384)
    parser.add_argument('--generator-keywords', nargs='+', action=KeyValueAction,
                        type=generator_keyword_value, default=None, metavar='KEY:VALUE',
                        help='Generator settings, e.g. response_nodes:4096 padding:1000; '
                             'values accept JSON numbers, booleans, arrays or strings')
    parser.add_argument('--refinement-generator-keywords', nargs='+', action=KeyValueAction,
                        type=generator_keyword_value, default=None, metavar='KEY:VALUE',
                        help='Override or add generator keywords for refinement')
    parser.add_argument('--harmonic-seed-factors', type=float, nargs='+',
                        help='Chirp-mass seed multipliers; THM defaults to 1, 1.5, 2 for higher-mode aliases')
    parser.add_argument('--optimizer', '--optimiser', choices=['pso', 'de'], type=str.lower,
                        default='pso', help='Population optimiser; default pso')
    parser.add_argument('--de-strategy', default='best1bin', choices=[
        'best1bin', 'best1exp', 'rand1bin', 'rand1exp', 'rand2bin', 'rand2exp',
        'randtobest1bin', 'randtobest1exp', 'currenttobest1bin', 'currenttobest1exp',
        'best2bin', 'best2exp'])
    parser.add_argument('--de-mutation', type=float, nargs=2, default=(.5, 1.),
                        metavar=('MIN', 'MAX'), help='DE mutation dithering range; default 0.5 1.0')
    parser.add_argument('--de-recombination', type=float, default=.7)
    parser.add_argument('--de-tol', type=float, default=.01)
    parser.add_argument('--de-atol', type=float, default=0.)
    parser.add_argument('--sky-sweep-every', type=int, default=20,
                        help='Explore gwreparam sky/helicity branches every N iterations/generations and at the end; default 20; 0 disables')
    parser.add_argument("--n-iterations",type=int,default=100,
                        help='Maximum PSO iterations or DE generations')
    parser.add_argument("--n-processes",type=int,default=1,
                        help="Population workers within one event, for PSO or DE; default 1")
    parser.add_argument("--event-workers",type=int,default=1,
                        help="Concurrent complete event fits, including local refinement; default 1")
    parser.add_argument('--event-fraction', default='0/1', metavar='N/M',
                        help='Analyse interleaved group N of M (0 <= N < M); default 0/1')
    parser.add_argument("--polish-evaluations",type=int,default=1000,
                        help="Maximum bounded local-refinement evaluations after PSO or DE; 0 disables it")
    parser.add_argument("--phase-seed-count",type=int,default=16,
                        help="Physical coalescence phases checked before the swarm; default 16")
    parser.add_argument("--previous-parameter-file",
                        help="Flag candidates within the saved optimisation windows of this earlier round")
    parser.add_argument(
        "--previous-event-files",
        nargs='+',
        help=(
            "Earlier event catalogues (HDF5 or CSV) used to exclude candidates within "
            "--minimum-peak-separation before optimisation"
        )
    )
    args = parser.parse_args()
    args.generator_keywords = args.generator_keywords or {}
    reserved = set(PARAMETERS) | {'mass1', 'mass2', 'distance', 'start_time', 'delta_t',
        'length', 'ifos', 'tdi', 'data_time_origin', 'f_lower', 'f_ref', 'orbit'}
    for keywords in [args.generator_keywords, args.refinement_generator_keywords or {}]:
        if reserved.intersection(keywords):
            parser.error('Generator keywords cannot override fitted parameters or grid/reference settings: '
                         + ', '.join(sorted(reserved.intersection(keywords))))
    if args.f_ref is None:
        args.f_ref = reference_frequency(args.waveform_model, args.f_lower)
    for stage in [stage_settings(args), stage_settings(args, True)]:
        if stage['waveform_model'] == 'fastbhb' and stage['orbit'] != 'analytic':
            parser.error('FastBHB supports analytic orbits only; use THM for ESA orbits')
        if not np.isfinite(stage['f_ref']) or (stage['f_ref'] < args.f_lower
                if stage['waveform_model'] in ('fastbhb','phenomthm') else stage['f_ref'] < 0):
            parser.error('Built-in models require finite f-ref >= f-lower; external models require nonnegative f-ref')
        if stage['waveform_model'] not in ('fastbhb','phenomthm'):
            validate = getattr(external_generator(stage['waveform_model']), 'validate_settings', None)
            if validate is not None:
                try:
                    validate(f_ref=stage['f_ref'], orbit=stage['orbit'], **stage['generator_keywords'])
                except ValueError as error:
                    parser.error(str(error))
    if args.harmonic_seed_factors is None:
        args.harmonic_seed_factors = [1., 1.5, 2.] if final_waveform_model(args) == 'phenomthm' else [1.]
    if not np.isfinite(args.harmonic_seed_factors).all() or min(args.harmonic_seed_factors) <= 0:
        parser.error('Harmonic seed factors must be finite and positive')
    if args.thm_oversampling < 1 or args.thm_response_nodes < 2:
        parser.error('Require f-ref >= f-lower, positive THM oversampling and at least two response nodes')
    try:
        event_partition(0, args.event_fraction)
    except ValueError as error:
        parser.error(str(error))
    if args.psd_file is None and args.psd_files is None:
        parser.error("Provide --psd-file or --psd-files")
    if args.polish_evaluations < 0:
        parser.error("Polish evaluations must be nonnegative")
    if args.sky_sweep_every < 0:
        parser.error('Sky sweep interval must be nonnegative')
    if args.statistic_threshold < 0 or args.minimum_peak_separation < 0:
        parser.error("Statistic threshold and minimum separation must be nonnegative")
    if stage_settings(args, True) != stage_settings(args) and not args.polish_evaluations:
        parser.error('A refinement waveform model requires positive polish-evaluations')
    if min(args.n_particles,args.n_iterations,args.n_processes,args.event_workers,args.phase_seed_count)<1:
        parser.error("Particle, iteration and process counts must be positive")

    if args.event_workers>1 and args.n_processes>1:
        parser.error("Use event workers or particle workers, not both")
    if args.optimizer == 'de':
        if args.n_particles < 5:
            parser.error('DE requires a population of at least 5')
        if args.de_strategy.startswith('rand2') and args.n_particles < 6:
            parser.error('DE rand2 strategies require a population of at least 6')
        controls = [*args.de_mutation, args.de_recombination, args.de_tol, args.de_atol]
        if (not np.isfinite(controls).all() or not 0 <= args.de_mutation[0] < args.de_mutation[1] < 2
                or not 0 <= args.de_recombination <= 1 or min(args.de_tol, args.de_atol) < 0):
            parser.error('Invalid DE mutation, recombination or convergence tolerances')

    args.fit_channels = args.fit_channels or args.optimisation_channels
    if args.psd_files and any(c not in args.psd_files for c in args.fit_channels):
        parser.error("Provide a PSD for every fit channel")
    init_logging(args.verbose)

    # Do a checks on the arguments:
    # Either parameter or timeseries output file (or both) must be given
    if args.output_parameter_file is None and args.output_timeseries_file is None:
        parser.error(
            "Either --output-parameter-file or --output-timeseries-file or both must be given"
        )

    return args

def get_events_to_optimise(args):
    logger.info("Getting events which need optimised")
    with h5py.File(args.input_file, 'r') as input_file:
        coincs = {}
        coincs["template_id"] = input_file["template_id"][:]
        
        for channel in args.optimisation_channels:
            coincs[f"trigger_id_{channel}"] = input_file[f"trigger_id_{channel}"][:]
            if f"statistic_{channel}" in input_file or f"snr_{channel}" in input_file:
                coincs[f"statistic_{channel}"] = read_statistic(input_file, channel)
            coincs[f"time_{channel}"] = input_file[f"time_{channel}"][:]
        # Use the stored ranking statistic, or reconstruct it when absent.
        if 'statistic' in input_file:
            candidate_stat = input_file['statistic'][:]
        elif 'snr' in input_file:
            candidate_stat = input_file['snr'][:]
        else:
            candidate_stat = np.sqrt(sum(coincs[f'statistic_{c}']**2 for c in args.optimisation_channels))

    candidate_stat = np.asarray(candidate_stat, dtype=float)
    if candidate_stat.shape != coincs['template_id'].shape or not np.isfinite(candidate_stat).all():
        raise ValueError('Candidate statistics must be finite and match the event count')
    # Channel times remain available for initialisation; their mean sets event separation.
    candidate_time = (
        coincs[f'time_{args.optimisation_channels[0]}'] + coincs[f'time_{args.optimisation_channels[1]}']
    ) / 2
    candidate_times_by_channel = {
        channel: coincs[f"time_{channel}"] for channel in args.optimisation_channels
    }
    candidate_template_id = coincs["template_id"]
    logger.info("%d events in file", len(candidate_stat))

    # remove anything not above the statistic threshold
    above_threshold = candidate_stat >= args.statistic_threshold
    candidate_stat = candidate_stat[above_threshold]
    candidate_time = candidate_time[above_threshold]
    for channel in args.optimisation_channels:
        candidate_times_by_channel[channel] = (
            candidate_times_by_channel[channel][above_threshold]
        )
    candidate_template_id = candidate_template_id[above_threshold]
    logger.info(
        "%d events remain after thresholding on statistic value %.1f",
        len(candidate_stat),
        args.statistic_threshold
    )

    def event_times(table):
        if 'tc' in table:
            return np.asarray(table['tc'], dtype=float)
        if 'time_s' in table:
            return np.asarray(table['time_s'], dtype=float)
        if 'end_time' in table:
            return np.asarray(table['end_time'], dtype=float)
        channels = [key[5:] for key in table if key.startswith('time_')]
        if channels:
            return np.mean([table[f'time_{channel}'] for channel in channels], axis=0)
        channels = [key[5:-8] for key in table if key.startswith('time_') and key.endswith('_seconds')]
        if channels:
            return np.mean([table[f'time_{channel}_seconds'] for channel in channels], axis=0)
        raise ValueError('Could not determine event times from previous event table')

    def load_previous_times(path):
        path = Path(path)
        if path.suffix.lower() == '.csv':
            table = np.genfromtxt(path, delimiter=',', names=True, dtype=None, encoding='utf-8')
            names = table.dtype.names
            if names is None:
                raise ValueError(f'CSV previous event file has no header columns: {path}')
            if table.shape == (0,):
                return np.array([], dtype=float)
            columns = {name: np.atleast_1d(table[name]) for name in names}
            return event_times(columns)
        with h5py.File(path, 'r') as f:
            table = {
                key: value[:] for key, value in f.items()
                if isinstance(value, h5py.Dataset) and value.ndim == 1
            }
        return event_times(table)

    previous_files = list(getattr(args, 'previous_event_files', []) or [])
    if getattr(args, 'previous_parameter_file', None):
        previous_files.append(args.previous_parameter_file)
    previous_files = list(dict.fromkeys(str(path) for path in previous_files))
    if previous_files:
        previous_times = np.concatenate([load_previous_times(path) for path in previous_files])
        previous_times = np.asarray(previous_times, dtype=float)
        if not np.isfinite(previous_times).all():
            raise ValueError('Previous event times must be finite')
        far_from_previous = np.all(
            np.abs(candidate_time[:, None] - previous_times[None, :]) > args.minimum_peak_separation,
            axis=1
        )
        excluded = int(np.count_nonzero(~far_from_previous))
        candidate_stat = candidate_stat[far_from_previous]
        candidate_time = candidate_time[far_from_previous]
        for channel in args.optimisation_channels:
            candidate_times_by_channel[channel] = candidate_times_by_channel[channel][far_from_previous]
        candidate_template_id = candidate_template_id[far_from_previous]
        logger.info(
            "%d events remain after excluding %d near previous events within %.1f s",
            len(candidate_stat),
            excluded,
            args.minimum_peak_separation
        )

    # Keep loudest-first representatives without confusing sorted and original indices.
    kept = []
    for i in np.argsort(-candidate_stat):
        if all(abs(candidate_time[i]-candidate_time[j]) >= args.minimum_peak_separation for j in kept):
            kept.append(int(i))
    candidate_stat, candidate_time, candidate_template_id = candidate_stat[kept], candidate_time[kept], candidate_template_id[kept]
    candidate_times_by_channel = {c:t[kept] for c,t in candidate_times_by_channel.items()}

    logger.info(
        "%d events remain after minimum separation requirement",
        len(candidate_stat)
    )

    # Sort back into statistic order
    stat_sort = np.argsort(candidate_stat)[::-1]
    candidate_stat = np.array(candidate_stat[stat_sort])
    candidate_time = np.array(candidate_time[stat_sort])
    for channel in args.optimisation_channels:
        candidate_times_by_channel[channel] = (
            np.array(candidate_times_by_channel[channel][stat_sort])
        )
    candidate_template_id = np.array(candidate_template_id[stat_sort])

    for i, (time, stat, tempid) in enumerate(zip(candidate_time, candidate_stat, candidate_template_id)):
        logger.debug(
            "Event %d / %d at time %.1f days with statistic %.3f, template %d",
            i+1, len(candidate_stat),
            time / 86400,
            stat,
            tempid
        )

    return candidate_stat, candidate_time, candidate_times_by_channel, candidate_template_id

_BANK_PARAMETERS = [
    'mass1',
    'mass2',
    'spin1z',
    'spin2z',
    'eclipticlongitude',
    'eclipticlatitude',
    'inclination',
    'polarization',
    'distance'
]

_PHASE_FIELD_CANDIDATES = (
    'coa_phase',
    'CoaPhase',
    'PhaseAtCoalescence',
    'phi_ref',
)

def get_bank_parameters(bank_filename, candidate_template_id):
    logger.info("Getting bank parameters for the top events")

    # template IDs need to be unique for reading from the file,
    # but by using return_inverse=True
    # we can get things back in the original order
    unique_tempids, inverse_indices = np.unique(
        candidate_template_id,
        return_inverse=True
    )

    bank = {}
    with h5py.File(bank_filename, 'r') as bank_file:
        for parameter in _BANK_PARAMETERS:
            unique_params = bank_file[parameter][unique_tempids]
            # Restore the input order using the inverse indexes.
            bank[parameter] = unique_params[inverse_indices]

        phase_field = None
        for candidate in _PHASE_FIELD_CANDIDATES:
            if candidate in bank_file:
                phase_field = candidate
                break

        if phase_field is not None:
            unique_phase = bank_file[phase_field][unique_tempids]
            phase = unique_phase[inverse_indices]
        else:
            phase = np.zeros(len(candidate_template_id), dtype=np.float64)
            logger.warning(
                "No coalescence phase field found in bank file. "
                "Falling back to phase=0 for optimisation waveforms. "
                "Looked for: %s",
                ", ".join(_PHASE_FIELD_CANDIDATES),
            )

        bank['coa_phase'] = phase
        bank['phi_ref'] = phase

    return bank

def get_data_file_starts_ends(data_files, channel):
    n_files = len(data_files)
    data_starts = np.zeros(n_files, dtype=np.float64)
    data_ends = np.zeros(n_files, dtype=np.float64)
    # delta_t has to be the same for all files
    delta_t = None
    n_samples = 0
    for i, data_filename in enumerate(data_files):
        with h5py.File(data_filename, 'r') as data_file:
            epoch = data_file.attrs['epoch']
            file_delta_t = data_file.attrs['delta_t']
            if delta_t not in [None, file_delta_t]:
                raise ValueError(
                    "All data files must have the same delta_t"
                )
            else:
                delta_t = file_delta_t

            n_samples += len(data_file[channel])
            duration = file_delta_t * len(data_file[channel])
            # For end time, we need to check if there is any padding at the end
            # of the data and subtract it
            padding_duration = data_file.attrs.get('padding_duration', 0)
            duration -= padding_duration
            end_time = epoch + duration
            data_starts[i] = epoch
            data_ends[i] = end_time

    return data_starts, data_ends, delta_t

def identify_file_of_coinc(data_starts,data_ends,coinc_times):
    ids=[]
    for time in coinc_times:
        candidates=np.flatnonzero((data_starts<=time)&(time<data_ends))
        if not len(candidates):
            raise ValueError(f"No data file covers event at {time}")
        ids.append(int(candidates[0]))
    return np.array(ids,dtype=np.uint32)


def _optimise_event(task):
    """Fit one independent data window and return only its model segment."""
    (args, idx, count, path, time, channel_time, template_id, statistic,
     bank, dt, previous_flag, previous_index) = task
    with h5py.File(path, 'r') as f:
        source_epoch = float(f.attrs['epoch'])
        length = len(f[args.optimisation_channels[0]])
        j0 = max(0, int(np.floor((time-args.optimisation_window_before-source_epoch)/dt)))
        j1 = min(length, int(np.ceil((time+args.optimisation_window_after-source_epoch)/dt)))
        if j1<=j0:
            raise ValueError('Empty optimisation window')
        n = next_fast_len(j1-j0)
        if n % 2:
            n = next_fast_len(n+1)
        event_epoch = source_epoch+j0*dt
        data = {c:TimeSeries(np.pad(np.asarray(f[c][j0:j1]), (0,n-(j1-j0))),
                             epoch=event_epoch,delta_t=dt) for c in args.fit_channels}
        scale = 1. if f.attrs.get('divide_by_laser_frequency',True) else float(f.attrs.get('laser_frequency',2.816e14))
    initial = {p:float(bank[p]) for p in _BANK_PARAMETERS}
    initial['mchirp'], initial['eta'] = mchirp_eta_from_mass1_mass2(initial['mass1'],initial['mass2'])
    initial['tc'] = float(channel_time)
    initial['coa_phase'] = float(bank['coa_phase'])
    lower, upper = [], []
    for p in PARAMETERS:
        lo = args.minimum_bounds.get(p, 0. if p=='coa_phase' else -np.inf)
        hi = args.maximum_bounds.get(p, 2*np.pi if p=='coa_phase' else np.inf)
        relative = args.bounds_window_relative_to_initial
        if p in relative:
            lo, hi = max(lo,initial[p]-relative[p]), min(hi,initial[p]+relative[p])
        elif 'log_'+p in relative:
            lo, hi = max(lo,initial[p]*10**(-relative['log_'+p])), min(hi,initial[p]*10**relative['log_'+p])
        if p=='tc':
            lo,hi=max(lo,event_epoch+dt),min(hi,event_epoch+(j1-j0-1)*dt)
        if not np.isfinite([lo,hi]).all() or hi<=lo:
            raise ValueError(f'Provide finite increasing bounds for {p}')
        lower.append(lo)
        upper.append(hi)
    seed = np.clip([initial[p] for p in PARAMETERS],lower,upper)
    rng = np.random.default_rng(args.seed+idx)
    population = rng.uniform(lower,upper,(args.n_particles,len(PARAMETERS)))
    population[0] = seed
    df = 1/(n*dt)
    psds = {}
    for c in args.fit_channels:
        psd_path = args.psd_files[c] if args.psd_files else args.psd_file
        psds[c] = psd_from_txt(psd_path,n//2+1,df,args.f_lower,is_asd_file=False)
        frequency = np.arange(n//2+1)*df
        psds[c].data[(frequency<args.f_lower)|(frequency>args.f_upper)] = np.inf
    shared = dict(f_lower=args.f_lower,f_upper=args.f_upper,tdi=args.tdi_generation,
                  distance=initial['distance'],template_scale=scale,data_fs=_data_frequencyseries(data))
    shared.update(waveform_model=getattr(args, 'waveform_model', 'fastbhb'),
                  f_ref=getattr(args, 'f_ref', args.f_lower), orbit=getattr(args, 'orbit', 'analytic'),
                  data_time_origin=getattr(args, 'data_time_origin', 0.),
                  thm_oversampling=getattr(args, 'thm_oversampling', 4),
                  thm_response_nodes=getattr(args, 'thm_response_nodes', 16384),
                  generator_keywords=getattr(args, 'generator_keywords', {}))
    shared.update(stage_settings(args))
    kwargs = dict(data=data,psd=psds,waveform_params_shared=shared,
                  optimisation_channels=args.fit_channels,
                  tc_reference=seed[7],time_bounds=(lower[7],upper[7]))
    # A (2,2) detection can follow a higher harmonic at a different apparent mass.
    phase_candidates = [seed[9], *np.linspace(lower[9],upper[9],args.phase_seed_count,endpoint=False)]
    initial_fit_stat = -np.inf
    bank_seed = seed.copy()
    seed_harmonic_factor = 1.
    for factor in getattr(args, 'harmonic_seed_factors', [1.]):
        chirp_mass = bank_seed[0]*factor
        if not lower[0] <= chirp_mass <= upper[0]:
            continue
        for phase in phase_candidates:
            candidate = bank_seed.copy()
            candidate[0], candidate[9] = chirp_mass, phase
            fit_stat,candidate = profile_physical_fit(candidate,**kwargs,raise_err=True)
            if fit_stat>initial_fit_stat:
                initial_fit_stat,seed = fit_stat,candidate
                seed_harmonic_factor = factor
    if not np.isfinite(initial_fit_stat) or initial_fit_stat<=0:
        raise ValueError('Initial template has no positive physical-fit statistic')
    initial_fit_stat = compute_statistic_core(seed,data,psds,shared,args.fit_channels,raise_err=True)
    kwargs['tc_reference'] = seed[7]
    shape_indices = [i for i,p in enumerate(PARAMETERS) if p!='tc']
    shape_lower,shape_upper = np.asarray(lower)[shape_indices],np.asarray(upper)[shape_indices]
    shape_seed = seed[shape_indices].copy()
    logarithmic_mass = getattr(args, "mass_coordinate", "log") == "log"
    if logarithmic_mass:
        shape_lower[0], shape_upper[0], shape_seed[0] = np.log10(
            [shape_lower[0], shape_upper[0], shape_seed[0]])
    kwargs["logarithmic_mass"] = logarithmic_mass
    population = rng.uniform(shape_lower,shape_upper,(args.n_particles,len(shape_indices)))
    # Keep half the swarm near the bank seed while the rest explores the bounds.
    near = max(1,args.n_particles//2)
    spread = .05*(shape_upper-shape_lower)
    if not logarithmic_mass:
        spread[0] = min(spread[0],.03*shape_seed[0])
    population[:near] = np.clip(shape_seed+rng.normal(size=(near,len(shape_indices)))*spread,
                               shape_lower,shape_upper)
    population[0] = shape_seed
    seed_masses = mass1_mass2_from_mchirp_eta(seed[0],seed[1])
    initial_chi_eff = chi_eff_from_component_spins(*seed_masses,seed[2],seed[3])
    logger.info(
        'Event %d / %d before optimisation: time=%.3f s (%.6f days), '
        'initial template statistic=%.3f, chirp mass=%.6g Msun, chi_eff=%.6f; '
        'candidate statistic=%.3f, template=%d',
        idx, count, seed[7], seed[7]/86400., initial_fit_stat, seed[0], initial_chi_eff,
        statistic, template_id)
    if previous_flag:
        logger.info('Event %d / %d FLAG: coincidence lies inside first-round event %d optimisation window',
                    idx, count, previous_index)
    logger.info('Event %d / %d optimiser=%s, population=%d, iterations=%d, workers=%d',
                idx,count,args.optimizer,args.n_particles,args.n_iterations,args.n_processes)
    cost, best = optimise_population(args,population,shape_lower,shape_upper,kwargs,idx)
    if not np.isfinite(cost) or -cost<initial_fit_stat:
        best = shape_seed
    refinement_settings = stage_settings(args, True)
    if refinement_settings != stage_settings(args):
        # Re-evaluate both seeds in THM; statistics from different models are incomparable.
        _, transferred = profile_physical_fit(best, **kwargs, raise_err=True)
        shared.update(refinement_settings)
        initial_fit_stat, seed = reseed_refinement(
            [transferred, seed], kwargs, lower, upper, args.phase_seed_count)
        kwargs['tc_reference'] = seed[7]
        best = seed[shape_indices].copy()
        if logarithmic_mass:
            best[0] = np.log10(best[0])
        logger.info('Event %d refinement model=%s, orbit=%s, reseeded statistic=%.6g',
                    idx, shared['waveform_model'], shared['orbit'], initial_fit_stat)
    polish_evaluations,polish_success = 0,False
    if args.polish_evaluations:
        # Normalise parameter units for a bounded local refinement.
        width = shape_upper-shape_lower
        def objective_value(position):
            return -compute_statistic_core(shape_lower+position*width,**kwargs)
        position = np.clip((best-shape_lower)/width,0,1)
        initial_cost = objective_value(position)
        with tqdm(total=args.polish_evaluations, desc=f'Event {idx} refinement', unit='evaluation') as progress:
            def local_objective(position):
                value = objective_value(position)
                progress.update(1)
                return value
            local = minimize(local_objective,position,method='Nelder-Mead',
                bounds=[(0.,1.)]*len(position),
                options=dict(maxfev=args.polish_evaluations,xatol=1e-5,fatol=1e-4))
        polish_evaluations,polish_success = local.nfev,bool(local.success)
        if np.isfinite(local.fun) and local.fun<initial_cost:
            best = shape_lower+local.x*width
        logger.info('Event %d / %d refinement complete: %d / %d evaluations, converged=%s; %s',
                    idx,count,polish_evaluations,args.polish_evaluations,
                    polish_success,local.message)
    _,best = profile_physical_fit(best,**kwargs,raise_err=True)
    wf = build_aligned_waveform(best,data,shared,raise_err=True)
    fit_stat,amplitude = coherent_fit(wf,shared['data_fs'],psds,args.fit_channels)
    if amplitude<=0 or fit_stat<initial_fit_stat:
        best = seed
        wf = build_aligned_waveform(best,data,shared,raise_err=True)
        fit_stat,amplitude = coherent_fit(wf,shared['data_fs'],psds,args.fit_channels)
    if amplitude<=0:
        raise ValueError('No positive distance scale for fitted event')
    fitted_distance = initial['distance']/amplitude
    physical_shared = {**shared,'distance':fitted_distance}
    # Predict every output channel from the same distance and physical phase.
    wf = build_aligned_waveform(best,data,physical_shared,raise_err=True,
                                waveform_channels=args.output_timeseries_channels)
    fit_stat,_ = coherent_fit(wf,shared['data_fs'],psds,args.fit_channels)
    segments = {c:(np.fft.irfft(wf[c].data,n=n)/dt)[:j1-j0]
                for c in args.output_timeseries_channels}
    record = dict(zip(PARAMETERS,best))
    record['mass1'],record['mass2'] = mass1_mass2_from_mchirp_eta(best[0],best[1])
    record.update(global_event_index=idx,initial_template_id=template_id,initial_stat=statistic,optimised_stat=fit_stat,
                  distance=fitted_distance,
                  amplitude_phase=0., initial_time=float(time),
                  seed_harmonic_factor=seed_harmonic_factor,
                  initial_fit_stat=initial_fit_stat,
                  optimisation_window_start=event_epoch,
                  optimisation_window_end=source_epoch+j1*dt,
                  in_previous_window=previous_flag, previous_event_index=previous_index,
                  polish_evaluations=polish_evaluations,polish_success=polish_success)
    record['f_lower'] = (args.f_lower if shared['waveform_model'] != 'fastbhb' else
        lower_frequency_for_duration(record,
        dict(f_lower=args.f_lower, f_upper=args.f_upper, pre_merger_days=1.)))
    logger.info(
        'Event %d / %d after optimisation: time=%.3f s (%.6f days), '
        'optimised statistic=%.3f, chirp mass=%.6g Msun, chi_eff=%.6f',
        idx, count, best[7], best[7]/86400., fit_stat, best[0],
        chi_eff_from_component_spins(record['mass1'],record['mass2'],best[2],best[3]))
    return record, event_epoch, segments


def main():
    args = load_arguments()
    stats, times, channel_times, ids = get_events_to_optimise(args)
    selected_indices = event_partition(len(times), args.event_fraction)
    previous_flags, previous_indices = previous_window_flags(times, args.previous_parameter_file)
    bank = get_bank_parameters(args.bank_file, ids)
    # All partitions use global ranks for seeds and select from the same candidate list.
    signature = hashlib.sha256()
    settings = {k:v for k,v in vars(args).items() if k not in
                ('event_fraction', 'output_parameter_file', 'output_timeseries_file',
                 'event_workers', 'n_processes', 'verbose')}
    signature.update(json.dumps(settings, sort_keys=True, default=str).encode())
    for values in (stats, times, ids, *(bank[k] for k in sorted(bank))):
        signature.update(np.asarray(values).tobytes())
    starts, ends = [], []
    metadata = None
    for path in args.data_files:
        stamp = Path(path).stat()
        signature.update(json.dumps([str(Path(path).resolve()),stamp.st_size,stamp.st_mtime_ns]).encode())
        with h5py.File(path, 'r') as f:
            dt = float(f.attrs['delta_t'])
            if metadata is not None and dt != metadata['delta_t']:
                raise ValueError('Data files must share a cadence')
            metadata = dict(f.attrs)
            starts.append(float(f.attrs['epoch']))
            ends.append(starts[-1]+len(f[args.optimisation_channels[0]])*dt)
    if getattr(args, 'data_time_origin', None) is None:
        args.data_time_origin = metadata.get('data_time_origin', metadata.get('t0_raw'))
    if any(stage_settings(args, refinement)['waveform_model'] != 'fastbhb'
           for refinement in (False, True)) and args.data_time_origin is None:
        if any(stage_settings(args, refinement)['waveform_model'] != 'fastbhb' and
               stage_settings(args, refinement)['orbit'] == 'esa' for refinement in (False, True)):
            raise ValueError('ESA fits require --data-time-origin or stored data_time_origin/t0_raw metadata')
        args.data_time_origin = 0.
    args.prepared_generator_keywords = {}
    for refinement in (False, True):
        stage = stage_settings(args, refinement)
        prepared = prepare_generator(stage['waveform_model'], orbit=stage['orbit'],
            data_time_origin=args.data_time_origin, start_time=min(starts), end_time=max(ends),
            keywords=stage['generator_keywords'])
        args.prepared_generator_keywords['refinement' if refinement else 'optimisation'] = prepared
    signature.update(json.dumps(args.prepared_generator_keywords,sort_keys=True).encode())
    optimisation_signature = signature.hexdigest()
    epoch, end = min(starts), max(ends)
    signals = {c:np.zeros(round((end-epoch)/dt)) for c in args.output_timeseries_channels}
    records = []
    chunks = identify_file_of_coinc(np.array(starts), np.array(ends), times)
    logger.info('Event fraction %s: optimising %d of %d globally selected events',
                args.event_fraction,len(selected_indices),len(times))
    tasks = [
        (args, idx, len(times), args.data_files[chunks[idx]], float(times[idx]),
         float(channel_times[args.optimisation_channels[0]][idx]), int(ids[idx]), float(stats[idx]),
         {p:float(values[idx]) for p,values in bank.items()}, dt,
         bool(previous_flags[idx]), int(previous_indices[idx]))
        for idx in selected_indices
    ]
    def collect(results):
        # Combine in candidate-statistic order for reproducible seeds and overlapping sums.
        for record, event_epoch, segments in results:
            offset = round((event_epoch-epoch)/dt)
            for c, segment in segments.items():
                signals[c][offset:offset+len(segment)] += segment
            records.append(record)
    if args.event_workers>1 and tasks:
        logger.info('Dispatching events in decreasing candidate statistic order to %d workers; completion order may differ',
                    args.event_workers)
        # Spawn workers with independent HDF5 handles and per-event random seeds.
        with ProcessPoolExecutor(max_workers=min(args.event_workers,len(tasks)),
                mp_context=get_context('spawn'), initializer=init_logging,
                initargs=(args.verbose,)) as pool:
            collect(pool.map(_optimise_event,tasks))
    else:
        collect(map(_optimise_event,tasks))
    if args.output_parameter_file:
        with h5py.File(args.output_parameter_file,'w') as f:
            f.attrs.update(optimisation_window_before=args.optimisation_window_before,
                           optimisation_window_after=args.optimisation_window_after,
                           tdi_generation=args.tdi_generation, f_upper=args.f_upper,
                           f_lower_floor=args.f_lower, pre_merger_days=1., taper_seconds=3600.,
                           initial_position_deg=-140., mode_array=DEFAULT_HM_MODE_ARRAY,
                           fit_channels=args.fit_channels,local_refinement_max_evaluations=args.polish_evaluations,
                           event_workers=args.event_workers,particle_workers=args.n_processes,
                           optimizer=args.optimizer, population_size=args.n_particles,
                           max_iterations=args.n_iterations, seed=args.seed,
                           event_fraction=args.event_fraction,
                           selected_event_count=len(times), optimisation_signature=optimisation_signature,
                           sky_sweep_every=args.sky_sweep_every,
                           optimisation_statistic='physical_network_snr',
                           de_strategy=args.de_strategy, de_mutation=args.de_mutation,
                           de_recombination=args.de_recombination,
                           de_tol=args.de_tol, de_atol=args.de_atol,
                           amplitude_model='positive physical distance; physical coalescence phase')
            final = stage_settings(args, True)
            model = final['waveform_model']
            f.attrs['fastbhb_phase_reference_applied'] = True
            f.attrs['population_waveform_model'] = args.waveform_model
            f.attrs['population_f_ref'] = args.f_ref
            f.attrs['population_orbit'] = args.orbit
            f.attrs['population_generator_keywords'] = json.dumps(stage_settings(args)['generator_keywords'])
            f.attrs.update(waveform_model=model,
                           approximant={'phenomthm':'IMRPhenomTHM','fastbhb':'IMRPhenomHM'}.get(model,model),
                           mode_array=DEFAULT_THM_MODE_ARRAY if model == 'phenomthm' else DEFAULT_HM_MODE_ARRAY,
                           f_ref=final['f_ref'], orbit=final['orbit'],
                           data_time_origin=float(args.data_time_origin or 0.),
                           thm_oversampling=getattr(args, 'thm_oversampling', 4),
                           thm_response_nodes=getattr(args, 'thm_response_nodes', 16384))
            f.attrs['generator_keywords'] = json.dumps(final['generator_keywords'])
            f.attrs['mass_coordinate'] = getattr(args, 'mass_coordinate', 'log')
            f.attrs['harmonic_seed_factors'] = getattr(args, 'harmonic_seed_factors', [1.])
            f.create_dataset('seed_harmonic_factor', data=np.asarray([r.get('seed_harmonic_factor', 1.) for r in records]))
            if args.previous_parameter_file:
                f.attrs['previous_parameter_file'] = args.previous_parameter_file
            if args.previous_event_files:
                f.attrs['previous_event_files'] = json.dumps(args.previous_event_files)
            for key in (*PARAMETERS,'mass1','mass2','global_event_index','initial_template_id','initial_stat','optimised_stat',
                        'distance','amplitude_phase','initial_time','initial_fit_stat','f_lower',
                        'optimisation_window_start','optimisation_window_end',
                        'in_previous_window','previous_event_index','polish_evaluations','polish_success'):
                dtype = bool if key in ('in_previous_window','polish_success') else np.int64 if key in ('global_event_index','initial_template_id','previous_event_index','polish_evaluations') else float
                f.create_dataset(key,data=np.asarray([r[key] for r in records], dtype=dtype))
    if args.output_timeseries_file:
        with h5py.File(args.output_timeseries_file,'w') as f:
            f.attrs.update(metadata)
            f.attrs['epoch'],f.attrs['delta_t'] = epoch,dt
            f.attrs['amplitude_model'] = 'positive physical distance; physical coalescence phase'
            f.attrs['event_fraction'] = args.event_fraction
            f.attrs['optimisation_signature'] = optimisation_signature
            for c, signal in signals.items():
                f.create_dataset(c,data=signal,compression='gzip' if parse_fraction(args.event_fraction)[1]>1 else None)


if __name__ == '__main__':
    main()
