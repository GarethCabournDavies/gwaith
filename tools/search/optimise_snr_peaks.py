"""
Optimise SNR for peaks found in coincidences
"""

import argparse
import copy
from concurrent.futures import ProcessPoolExecutor
from multiprocessing import get_context
import logging
import h5py
import numpy as np
import pyswarms as ps

from gwaith.types import TimeSeries, FrequencySeries
from scipy.fft import next_fast_len
from scipy.optimize import minimize
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

logger = logging.getLogger(__name__)


PARAMETERS = ('mchirp', 'eta', 'spin1z', 'spin2z', 'eclipticlongitude',
              'eclipticlatitude', 'inclination', 'tc', 'polarization', 'coa_phase')


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
    rho,full[7] = profile_merger_time(wf,data_fs,psd,optimisation_channels,
                                    len(first),first.delta_t,full[7],time_bounds)
    return rho,full


def compute_network_snr_core(v, data, psd, waveform_params_shared, optimisation_channels,
                             raise_err=False, tc_reference=None, time_bounds=None):
    if time_bounds is not None:
        return profile_physical_fit(v,data,psd,waveform_params_shared,optimisation_channels,
                                    tc_reference,time_bounds,raise_err)[0]
    wf = build_aligned_waveform(v,data,waveform_params_shared,raise_err)
    if wf is None:
        return -np.inf
    data_fs = waveform_params_shared.get('data_fs') or _data_frequencyseries(data)
    return coherent_fit(wf,data_fs,psd,optimisation_channels)[0]


def get_snr_and_distance_factor(wf, data, psd, optimisation_channels):
    rho, amp = coherent_fit(wf,_data_frequencyseries(data),psd,optimisation_channels)
    return rho, 1/abs(amp) if abs(amp)>0 else np.inf


def shift_and_normalise_waveform(wf_fd, coalescence_in_segment, data, psd,
                                 optimisation_channels, delta_t, output_channels):
    """Normalise a cyclic-zero-merger FD template for plotting."""
    first = next(iter(data.values()))
    aligned = {c:wf_fd[c].cyclic_time_shift(coalescence_in_segment) for c in output_channels}
    rho, amp = coherent_fit(aligned,_data_frequencyseries(data),psd,optimisation_channels)
    out = {c:TimeSeries(np.fft.irfft(amp*aligned[c].data,n=len(first))/delta_t,
                       epoch=first.epoch,delta_t=delta_t) for c in output_channels}
    return out,rho,1/abs(amp) if abs(amp)>0 else np.inf


def compute_minus_network_snr(positions, **kwargs):
    return np.array([-compute_network_snr_core(v,**kwargs) for v in positions])


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
        "--n-particles",
        type=int,
        default=100,
        help="Number of particles to use in the optimiser"
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
    parser.add_argument("--n-iterations",type=int,default=100)
    parser.add_argument("--n-processes",type=int,default=1,
                        help="Particle workers within one event; default 1")
    parser.add_argument("--event-workers",type=int,default=1,
                        help="Concurrent complete event fits, including local refinement; default 1")
    parser.add_argument("--polish-evaluations",type=int,default=1000,
                        help="Maximum bounded local-refinement evaluations after the swarm; 0 disables it")
    parser.add_argument("--phase-seed-count",type=int,default=16,
                        help="Physical coalescence phases checked before the swarm; default 16")
    parser.add_argument("--previous-parameter-file",
                        help="Flag candidates within the saved optimisation windows of this earlier round")
    args = parser.parse_args()
    if args.psd_file is None and args.psd_files is None:
        parser.error("Provide --psd-file or --psd-files")
    if args.polish_evaluations < 0:
        parser.error("Polish evaluations must be nonnegative")
    if args.statistic_threshold < 0 or args.minimum_peak_separation < 0:
        parser.error("SNR threshold and minimum separation must be nonnegative")
    if min(args.n_particles,args.n_iterations,args.n_processes,args.event_workers,args.phase_seed_count)<1:
        parser.error("Particle, iteration and process counts must be positive")

    if args.event_workers>1 and args.n_processes>1:
        parser.error("Use event workers or particle workers, not both")

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
            coincs[f"snr_{channel}"] = input_file[f"snr_{channel}"][:]
            coincs[f"time_{channel}"] = input_file[f"time_{channel}"][:]

    coinc_stat = (
        coincs[f'snr_{args.optimisation_channels[0]}'] ** 2 + coincs[f'snr_{args.optimisation_channels[1]}'] ** 2
    ) ** 0.5
    # The mean of the two channel times is only used to set the minimum distance
    # between events chosen for optimisation. The per-channel trigger times are
    # kept separately and used for the initial point / plot, so that the SNR in
    # each channel at its own peak time is not thrown away.
    coinc_time = (
        coincs[f'time_{args.optimisation_channels[0]}'] + coincs[f'time_{args.optimisation_channels[1]}']
    ) / 2
    coinc_times_by_channel = {
        channel: coincs[f"time_{channel}"] for channel in args.optimisation_channels
    }
    coinc_tempid = coincs["template_id"]
    logger.info("%d events in file", len(coinc_stat))

    # remove anything not above the statistic threshold
    above_threshold = coinc_stat >= args.statistic_threshold
    coinc_stat = coinc_stat[above_threshold]
    coinc_time = coinc_time[above_threshold]
    for channel in args.optimisation_channels:
        coinc_times_by_channel[channel] = (
            coinc_times_by_channel[channel][above_threshold]
        )
    coinc_tempid = coinc_tempid[above_threshold]
    logger.info(
        "%d events remain after thresholding on statistic value %.1f",
        len(coinc_stat),
        args.statistic_threshold
    )

    # Keep loudest-first representatives without confusing sorted and original indices.
    kept = []
    for i in np.argsort(-coinc_stat):
        if all(abs(coinc_time[i]-coinc_time[j]) >= args.minimum_peak_separation for j in kept):
            kept.append(int(i))
    coinc_stat, coinc_time, coinc_tempid = coinc_stat[kept], coinc_time[kept], coinc_tempid[kept]
    coinc_times_by_channel = {c:t[kept] for c,t in coinc_times_by_channel.items()}

    logger.info(
        "%d events remain after minimum separation requirement",
        len(coinc_stat)
    )

    # Sort back into statistic order
    stat_sort = np.argsort(coinc_stat)[::-1]
    coinc_stat = np.array(coinc_stat[stat_sort])
    coinc_time = np.array(coinc_time[stat_sort])
    for channel in args.optimisation_channels:
        coinc_times_by_channel[channel] = (
            np.array(coinc_times_by_channel[channel][stat_sort])
        )
    coinc_tempid = np.array(coinc_tempid[stat_sort])

    for i, (time, stat, tempid) in enumerate(zip(coinc_time, coinc_stat, coinc_tempid)):
        logger.debug(
            "Event %d / %d at time %.1f days with statistic %.3f, template %d",
            i+1, len(coinc_stat),
            time / 86400,
            stat,
            tempid
        )

    return coinc_stat, coinc_time, coinc_times_by_channel, coinc_tempid

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

def get_bank_parameters(bank_filename, coinc_tempid):
    logger.info("Getting bank parameters for the top events")

    # template IDs need to be unique for reading from the file,
    # but by using return_inverse=True
    # we can get things back in the original order
    unique_tempids, inverse_indices = np.unique(
        coinc_tempid,
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
            phase = np.zeros(len(coinc_tempid), dtype=np.float64)
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
    initial_snr = -np.inf
    for phase in phase_candidates:
        candidate = seed.copy()
        candidate[9] = phase
        rho,candidate = profile_physical_fit(candidate,**kwargs,raise_err=True)
        if rho>initial_snr:
            initial_snr,seed = rho,candidate
    if not np.isfinite(initial_snr) or initial_snr<=0:
        raise ValueError('Initial template has no positive physical SNR')
    initial_snr = compute_network_snr_core(seed,data,psds,shared,args.fit_channels,raise_err=True)
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
        'initial template SNR (higher-order modes)=%.3f, chirp mass=%.6g Msun, chi_eff=%.6f; '
        'coincidence quadrature-sum SNR ((2,2) only)=%.3f, template=%d',
        idx+1, count, seed[7], seed[7]/86400., initial_snr, seed[0], initial_chi_eff,
        statistic, template_id)
    if previous_flag:
        logger.info('Event %d / %d FLAG: coincidence lies inside first-round event %d optimisation window',
                    idx+1, count, previous_index+1)
    np.random.seed(args.seed+idx)
    optimiser = ps.single.GlobalBestPSO(n_particles=args.n_particles,dimensions=len(shape_indices),
        options={'c1':.5,'c2':.3,'w':.9},bounds=(shape_lower,shape_upper),init_pos=population)
    cost, best = optimiser.optimize(compute_minus_network_snr,iters=args.n_iterations,
        n_processes=args.n_processes if args.n_processes>1 else None,verbose=False,**kwargs)
    if not np.isfinite(cost) or -cost<initial_snr:
        best = shape_seed
    polish_evaluations,polish_success = 0,False
    if args.polish_evaluations:
        # Normalise parameter units for a bounded local refinement.
        width = shape_upper-shape_lower
        def local_objective(position):
            return -compute_network_snr_core(shape_lower+position*width,**kwargs)
        position = np.clip((best-shape_lower)/width,0,1)
        local = minimize(local_objective,position,method='Nelder-Mead',
            bounds=[(0.,1.)]*len(position),
            options=dict(maxfev=args.polish_evaluations,xatol=1e-5,fatol=1e-4))
        polish_evaluations,polish_success = local.nfev,bool(local.success)
        if np.isfinite(local.fun) and local.fun<local_objective(position):
            best = shape_lower+local.x*width
        logger.debug('Event %d / %d local refinement: %d evaluations, converged=%s',
                     idx+1,count,polish_evaluations,polish_success)
    _,best = profile_physical_fit(best,**kwargs,raise_err=True)
    wf = build_aligned_waveform(best,data,shared,raise_err=True)
    rho,amplitude = coherent_fit(wf,shared['data_fs'],psds,args.fit_channels)
    if amplitude<=0 or rho<initial_snr:
        best = seed
        wf = build_aligned_waveform(best,data,shared,raise_err=True)
        rho,amplitude = coherent_fit(wf,shared['data_fs'],psds,args.fit_channels)
    if amplitude<=0:
        raise ValueError('No positive distance scale for fitted event')
    fitted_distance = initial['distance']/amplitude
    physical_shared = {**shared,'distance':fitted_distance}
    # Predict every output channel from the same distance and physical phase.
    wf = build_aligned_waveform(best,data,physical_shared,raise_err=True,
                                waveform_channels=args.output_timeseries_channels)
    rho,_ = coherent_fit(wf,shared['data_fs'],psds,args.fit_channels)
    segments = {c:(np.fft.irfft(wf[c].data,n=n)/dt)[:j1-j0]
                for c in args.output_timeseries_channels}
    record = dict(zip(PARAMETERS,best))
    record['mass1'],record['mass2'] = mass1_mass2_from_mchirp_eta(best[0],best[1])
    record.update(initial_template_id=template_id,initial_stat=statistic,optimised_stat=rho,
                  distance=fitted_distance,
                  amplitude_phase=0., initial_time=float(time),
                  initial_fit_snr=initial_snr,
                  optimisation_window_start=event_epoch,
                  optimisation_window_end=source_epoch+j1*dt,
                  in_previous_window=previous_flag, previous_event_index=previous_index,
                  polish_evaluations=polish_evaluations,polish_success=polish_success)
    record['f_lower'] = lower_frequency_for_duration(record,
        dict(f_lower=args.f_lower, f_upper=args.f_upper, pre_merger_days=1.))
    logger.info(
        'Event %d / %d after optimisation: time=%.3f s (%.6f days), '
        'optimised SNR (higher-order modes)=%.3f, chirp mass=%.6g Msun, chi_eff=%.6f',
        idx+1, count, best[7], best[7]/86400., rho, best[0],
        chi_eff_from_component_spins(record['mass1'],record['mass2'],best[2],best[3]))
    return record, event_epoch, segments


def main():
    args = load_arguments()
    stats, times, channel_times, ids = get_events_to_optimise(args)
    previous_flags, previous_indices = previous_window_flags(times, args.previous_parameter_file)
    bank = get_bank_parameters(args.bank_file, ids)
    starts, ends = [], []
    metadata = None
    for path in args.data_files:
        with h5py.File(path, 'r') as f:
            dt = float(f.attrs['delta_t'])
            if metadata is not None and dt != metadata['delta_t']:
                raise ValueError('Data files must share a cadence')
            metadata = dict(f.attrs)
            starts.append(float(f.attrs['epoch']))
            ends.append(starts[-1]+len(f[args.optimisation_channels[0]])*dt)
    epoch, end = min(starts), max(ends)
    signals = {c:np.zeros(round((end-epoch)/dt)) for c in args.output_timeseries_channels}
    records = []
    chunks = identify_file_of_coinc(np.array(starts), np.array(ends), times)
    logger.info('Optimising %d events in decreasing coincident network SNR order', len(times))
    tasks = [
        (args, idx, len(times), args.data_files[chunks[idx]], float(times[idx]),
         float(channel_times[args.optimisation_channels[0]][idx]), int(ids[idx]), float(stats[idx]),
         {p:float(values[idx]) for p,values in bank.items()}, dt,
         bool(previous_flags[idx]), int(previous_indices[idx]))
        for idx in range(len(times))
    ]
    def collect(results):
        # Combine in SNR order so seeded fits and overlapping sums are reproducible.
        for record, event_epoch, segments in results:
            offset = round((event_epoch-epoch)/dt)
            for c, segment in segments.items():
                signals[c][offset:offset+len(segment)] += segment
            records.append(record)
    if args.event_workers>1 and tasks:
        logger.info('Dispatching events in decreasing SNR order to %d workers; completion order may differ',
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
                           amplitude_model='positive physical distance; physical coalescence phase')
            if args.previous_parameter_file:
                f.attrs['previous_parameter_file'] = args.previous_parameter_file
            for key in (*PARAMETERS,'mass1','mass2','initial_template_id','initial_stat','optimised_stat',
                        'distance','amplitude_phase','initial_time','initial_fit_snr','f_lower',
                        'optimisation_window_start','optimisation_window_end',
                        'in_previous_window','previous_event_index','polish_evaluations','polish_success'):
                dtype = bool if key in ('in_previous_window','polish_success') else np.int64 if key in ('initial_template_id','previous_event_index','polish_evaluations') else float
                f.create_dataset(key,data=np.asarray([r[key] for r in records], dtype=dtype))
    if args.output_timeseries_file:
        with h5py.File(args.output_timeseries_file,'w') as f:
            f.attrs.update(metadata)
            f.attrs['epoch'],f.attrs['delta_t'] = epoch,dt
            f.attrs['amplitude_model'] = 'positive physical distance; physical coalescence phase'
            for c, signal in signals.items():
                f.create_dataset(c,data=signal)


if __name__ == '__main__':
    main()
