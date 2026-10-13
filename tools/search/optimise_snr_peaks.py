"""
Optimise the physical-fit statistic for selected search candidates.
"""

import argparse
import copy
import hashlib
import json
from pathlib import Path
from functools import partial
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
from hwilas.types import KeyValueAction
from hwilas.search.rounds import previous_window_flags
from hwilas.search.sky_modes import sky_mode_vectors
from hwilas.search.partitions import event_partition, parse_fraction

logger = logging.getLogger(__name__)


PARAMETERS = ('mchirp', 'eta', 'spin1z', 'spin2z', 'eclipticlongitude',
              'eclipticlatitude', 'inclination', 'tc', 'polarization', 'coa_phase')


def _de_objective(position, objective_kwargs):
    """Scalar physical-fit objective, pickleable for spawned DE workers."""
    return -compute_statistic_core(position, **objective_kwargs)


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
        maxiter=args.n_iterations, seed=args.seed+event_index, strategy=args.de_strategy,
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
                        reference_tc=p['tc']-float(first.epoch))
        cutoff = lower_frequency_for_duration(p,settings)
        settings['f_lower'] = cutoff
        # Translate orbit and time together; every candidate uses the same FFT grid.
        tm = template(p,float(first.epoch),len(first),settings,modes=DEFAULT_HM_MODE_ARRAY)
        t = float(first.epoch)+np.arange(len(first))*first.delta_t
        ramp = .5-.5*np.cos(np.pi*np.clip((t-(p['tc']-86400.))/3600.,0,1))
        result = {}
        for c in channels:
            h = first.delta_t*np.fft.rfft(tm[c]*ramp)
            h[:int(cutoff/(1/(len(first)*first.delta_t)))] = 0
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
                         tc_reference, time_bounds, raise_err=False):
    """Profile merger time while the swarm searches the physical shape parameters."""
    full = np.insert(v,7,tc_reference) if len(v)==9 else np.asarray(v).copy()
    wf = build_aligned_waveform(full,data,waveform_params_shared,raise_err)
    if wf is None:
        return -np.inf,full
    first = next(iter(data.values()))
    data_fs = waveform_params_shared.get('data_fs') or _data_frequencyseries(data)
    fit_stat,full[7] = profile_merger_time(wf,data_fs,psd,optimisation_channels,
                                    len(first),first.delta_t,full[7],time_bounds)
    return fit_stat,full


def compute_statistic_core(v, data, psd, waveform_params_shared, optimisation_channels,
                             raise_err=False, tc_reference=None, time_bounds=None):
    if time_bounds is not None:
        return profile_physical_fit(v,data,psd,waveform_params_shared,optimisation_channels,
                                    tc_reference,time_bounds,raise_err)[0]
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
    args = parser.parse_args()
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
    kwargs = dict(data=data,psd=psds,waveform_params_shared=shared,
                  optimisation_channels=args.fit_channels,
                  tc_reference=seed[7],time_bounds=(lower[7],upper[7]))
    # A physical phase seed avoids relying on an arbitrary complex rotation.
    phase_candidates = [seed[9], *np.linspace(lower[9],upper[9],args.phase_seed_count,endpoint=False)]
    initial_fit_stat = -np.inf
    for phase in phase_candidates:
        candidate = seed.copy()
        candidate[9] = phase
        fit_stat,candidate = profile_physical_fit(candidate,**kwargs,raise_err=True)
        if fit_stat>initial_fit_stat:
            initial_fit_stat,seed = fit_stat,candidate
    if not np.isfinite(initial_fit_stat) or initial_fit_stat<=0:
        raise ValueError('Initial template has no positive physical-fit statistic')
    initial_fit_stat = compute_statistic_core(seed,data,psds,shared,args.fit_channels,raise_err=True)
    kwargs['tc_reference'] = seed[7]
    shape_indices = [i for i,p in enumerate(PARAMETERS) if p!='tc']
    shape_lower,shape_upper = np.asarray(lower)[shape_indices],np.asarray(upper)[shape_indices]
    shape_seed = seed[shape_indices]
    population = rng.uniform(shape_lower,shape_upper,(args.n_particles,len(shape_indices)))
    # Keep half the swarm near the bank seed while the rest explores the bounds.
    near = max(1,args.n_particles//2)
    spread = .05*(shape_upper-shape_lower)
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
                  initial_fit_stat=initial_fit_stat,
                  optimisation_window_start=event_epoch,
                  optimisation_window_end=source_epoch+j1*dt,
                  in_previous_window=previous_flag, previous_event_index=previous_index,
                  polish_evaluations=polish_evaluations,polish_success=polish_success)
    record['f_lower'] = lower_frequency_for_duration(record,
        dict(f_lower=args.f_lower, f_upper=args.f_upper, pre_merger_days=1.))
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
            if args.previous_parameter_file:
                f.attrs['previous_parameter_file'] = args.previous_parameter_file
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
