"""
Optimise SNR for peaks found in coincidences
"""

import argparse
import logging
import h5py
import numpy as np
import copy
from matplotlib import pyplot as plt

import pyswarms as ps

import pycbc
from pycbc.types import TimeSeries
from pycbc import conversions as conv
import pycbc.psd
import pycbc.filter
import pycbc.waveform

logger = logging.getLogger(__name__)

class KeyValueAction(argparse.Action):
    """
    Argparse action to accept key:value pairs.
    These are stored in a dictionary.
    """
    def __call__(self, parser, namespace, values, option_string=None):
        if getattr(namespace, self.dest, None) is None:
            setattr(namespace, self.dest, {})
        for value in values:
            key, val_str = value.split(':', 1)
            val = float(val_str)
            getattr(namespace, self.dest)[key] = val



def compute_network_snr_core(v, data, psd, waveform_params_shared, raise_err=False):
    mtotal = conv.mtotal_from_mchirp_eta(v[0], v[1])
    mass1 = conv.mass1_from_mtotal_eta(mtotal, v[1])
    mass2 = conv.mass2_from_mtotal_eta(mtotal, v[1])

    # get tlen out from the data
    ifos = list(data.keys())
    delta_t = data[ifos[0]].delta_t
    
    waveform_params = copy.deepcopy(waveform_params_shared)
    waveform_params.update({
        "mass1": mass1,
        "mass2": mass2,
        "spin1z": v[2],
        "spin2z": v[3],
        "eclipticlongitude": v[4],
        "eclipticlatitude": v[5],
        "inclination": v[6],
        "tc": v[7],
        "polarization": v[8],
    })

    try:
        wf = pycbc.waveform.get_fd_det_waveform(
            **waveform_params,
            channels=args.channels,
            delta_t=delta_t,
            ifos=ifos
        )
    except RuntimeError as err:
        if raise_err:
            raise err
        # The waveform generation has failed - instead of actually failing,
        # just say that this waveform is not allowed
        return -np.inf
    
    snrsq = 0
    for channel in data.keys():
        snrsq += pycbc.filter.matched_filter(
            wf[channel],
            data[channel],
            psd=psd,
        )[0] ** 2

    return snrsq ** 0.5


def compute_minus_network_snr(positions, **kwargs):
    costs = np.empty(len(positions), dtype=np.float64)
    for idx, v in enumerate(positions):
        nsnr = compute_network_snr_core(v, **kwargs)
        logger.debug("snr: %.3f", nsnr)
        costs[idx] = -nsnr
    return costs


parser = argparse.ArgumentParser()
pycbc.add_common_pycbc_options(parser)
parser.add_argument(
    "--input-file",
    required=True,
    help="File containing coincident events"
)
parser.add_argument(
    "--n-loudest",
    type=int,
    default=10,
    help="Optimise & remove the N loudest events. Default 10"
)
parser.add_argument(
    "--statistic-threshold",
    type=float,
    default=60,
    help="When deciding the events to optimise / add to signal timeseries, use this threshold. "
    "Default 60."
)
parser.add_argument(
    "--bank-file",
    required=True
)
parser.add_argument(
    "--output-file",
    help="File containing the optimised event parameters"
)
parser.add_argument(
    "--psd-file",
    required=True,
    help="File containing the PSD"
)
parser.add_argument(
    "--data-file",
    required=True,
    help="File containing the data"
)
parser.add_argument(
    "--input-timeseries-file",
    help="File containing the data with optimised events as a timeseries"
)
parser.add_argument(
    "--output-timeseries-file",
    help="File containing the data with optimised events as a timeseries"
)
parser.add_argument(
    "--channels",
    nargs='+',
    default=['LISA_A', 'LISA_E'],
    help="Channels that will be used for optimisation and removal. "
    "Default LISA_A and LISA_E. "
)
parser.add_argument(
    "--f-lower",
    type=float,
    default=1e-6,
)
parser.add_argument(
    "--testing-plots",
    help="Directory into which to store plots used for testing"
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
    help="A parameter:window_size paired list of bounds to use for the optimiser. "
    "e.g. mchirp:100 will give an mchirp window of 100 Msol either side of the template."
)
parser.add_argument(
    "--maximum-bounds",
    nargs='+',
    action=KeyValueAction,
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
    default={},
    help="Set minimum bounds on these parameters according to given parameters. "
    "Given in parameter:minimum_bound pairs e.g. mchirp:5e5. This will be set as the bound if "
    "--bounds-window-relative-to-initial is not given, or if the template minus the bounds "
    "is smaller than this value"
)

args = parser.parse_args()

# Some libraries, like pyswarms, may add a default StreamHandler to the root
# logger upon import. This can lead to duplicate log messages. To prevent this,
# we find and remove any pre-existing StreamHandlers before initializing
# our own logging.
logging.getLogger().handlers.clear()



pycbc.init_logging(args.verbose)

logger.info("Getting events which need optimised")
with h5py.File(args.input_file, 'r') as input_file:
    coincs = {}
    coincs["template_id"] = input_file["template_id"][:]
    
    for channel in args.channels:
        coincs[f"trigger_id_{channel}"] = input_file[f"trigger_id_{channel}"][:]
        coincs[f"snr_{channel}"] = input_file[f"snr_{channel}"][:]
        coincs[f"time_{channel}"] = input_file[f"time_{channel}"][:]

coinc_stat = (
    coincs[f'snr_{args.channels[0]}'] ** 2 + coincs[f'snr_{args.channels[1]}'] ** 2
) ** 0.5
coinc_time = (
    coincs[f'time_{args.channels[0]}'] + coincs[f'time_{args.channels[1]}']
) / 2
coinc_tempid = coincs["template_id"]

above_threshold = coinc_stat > args.statistic_threshold
coinc_stat = coinc_stat[above_threshold]
coinc_time = coinc_time[above_threshold]
coinc_tempid = coinc_tempid[above_threshold]

if len(coinc_stat) > args.n_loudest:
    stat_sort = np.argsort(coinc_stat)[::-1]
    coinc_stat = coinc_stat[stat_sort[:args.n_loudest]]
    coinc_time = coinc_time[stat_sort[:args.n_loudest]]
    coinc_tempid = coinc_tempid[stat_sort[:args.n_loudest]]

# template IDs need to be unique for reading from the file, but by using return_inverse=True
# We can get things back in the original order
unique_tempids, inverse_indices = np.unique(coinc_tempid, return_inverse=True)

logger.info("Getting bank parameters for the top events")
bank_parameters = [
    'mass1',
    'mass2',
    'spin1z',
    'spin2z',
    'eclipticlongitude',
    'eclipticlatitude',
    'inclination',
    'polarization'
]
bank = {}
with h5py.File(args.bank_file, 'r') as bank_file:
    for parameter in bank_parameters:
        unique_params = bank_file[parameter][unique_tempids]
        # Restore the input order using the inverse indexes.
        bank[parameter] = unique_params[inverse_indices]

logger.info("Loading Data")
data = {}
with h5py.File(args.data_file, 'r') as data_file:
    data = {
        channel: TimeSeries(
            data_file[channel][:],
            epoch=data_file.attrs['epoch'],
            delta_t=data_file.attrs['delta_t'],
        )
        for channel in args.channels
    }
    original_data_epoch = data_file.attrs['epoch']
    delta_t = data_file.attrs['delta_t']

original_dlen = len(data[args.channels[0]])
original_tlen = original_dlen * delta_t
original_delta_f = 1. / original_tlen
original_flen = original_tlen // 2 + 1


if args.input_timeseries_file is not None:
    logger.info("Loading signal timeseries from file")
    with h5py.File(args.input_timeseries_file, 'r') as input_ts_file:
        signal_timeseries = {
            channel:TimeSeries(
                input_ts_file[channel][:],
                epoch=original_data_epoch,
                delta_t=delta_t,
            )
            for channel in args.channels
        }
else:
    # Initialise empty TimeSeries which will contain the signal timeseries
    signal_timeseries = {
        channel: TimeSeries(
            np.zeros(len(data[channel])),
            epoch=original_data_epoch,
            delta_t=delta_t,
        )
        for channel in args.channels
    }

# We are only going to output chirp mass and time as the parameters that can be
# estimated relatively accurately and passed to parameter estimation, but the optimiser
# needs other parameters passed to it
optimiser_parameters = [
    'mchirp',
    'eta',
    'spin1z',
    'spin2z',
    'eclipticlongitude',
    'eclipticlatitude',
    'inclination',
    'tc',
    "polarization",
]

logger.info("Determining bounds for parameters not relative to initial")

# All optimiser parameters need to have min and max. These should be set from the dynamic options or
# static min/max
for parameter in optimiser_parameters:
    if parameter in args.bounds_window_relative_to_initial:
        # min/max are going to be set dynamically
        continue
    if parameter not in args.maximum_bounds and parameter not in args.minimum_bounds:
        raise ValueError(
            f"Minimum and Maximum values need to be set for {parameter}"
        )

static_parameter_bounds = {
    parameter: (None, None)
    for parameter in optimiser_parameters
}

for parameter in optimiser_parameters:
    min_bound = -np.inf
    max_bound = np.inf
    if parameter in args.maximum_bounds:
        max_bound = min(max_bound, args.maximum_bounds[parameter])
    if parameter in args.minimum_bounds:
        min_bound = max(min_bound, args.minimum_bounds[parameter])
    static_parameter_bounds[parameter] = (min_bound, max_bound)

logger.info("Overwhitening the data")
ref_channel = args.channels[0]
ref_freq_data = data[ref_channel].to_frequencyseries()
psd = pycbc.psd.from_txt(
    args.psd_file,
    len(ref_freq_data),
    ref_freq_data.delta_f,
    ref_freq_data.delta_f,
    is_asd_file=False,
)
psd.data[0] = np.inf
psd.data[1] = np.inf
    
for idx, (stat, time) in enumerate(zip(coinc_stat, coinc_time)):
    logger.info("Event %d at time %.1f days with statistic %.3f", idx, time / 86400, stat)

    logger.info("Getting parameters")
    this_event_template_id = coinc_tempid[idx]
    this_event_template = {
        parameter: bank[parameter][idx]
        for parameter in bank_parameters
    }

    mchirp_init = conv.mchirp_from_mass1_mass2(
        this_event_template['mass1'],
        this_event_template['mass2']
    )
    eta_init = conv.eta_from_mass1_mass2(
        this_event_template['mass1'],
        this_event_template['mass2']
    )
    this_event_template['mchirp'] = mchirp_init
    this_event_template['eta'] = eta_init
    this_event_template['tc'] = time

    logger.info("Setting up the initial population of particles for the optimiser")

    # Start from a fresh copy of static bounds for each event so there is no
    # mutation carried across loop iterations.
    bounds_this_event = {
        parameter: static_parameter_bounds[parameter]
        for parameter in optimiser_parameters
    }
    for parameter in optimiser_parameters:
        if parameter in args.bounds_window_relative_to_initial:
            event_max_bound = this_event_template[parameter] + args.bounds_window_relative_to_initial[parameter]
            event_min_bound = this_event_template[parameter] - args.bounds_window_relative_to_initial[parameter]
            bounds_this_event[parameter] = (
                max(static_parameter_bounds[parameter][0], event_min_bound),
                min(static_parameter_bounds[parameter][1], event_max_bound)
            )
        elif f"log_{parameter}" in args.bounds_window_relative_to_initial:
            max_bound_log = np.log10(this_event_template[parameter]) + args.bounds_window_relative_to_initial[f"log_{parameter}"]
            min_bound_log = np.log10(this_event_template[parameter]) - args.bounds_window_relative_to_initial[f"log_{parameter}"]
            bounds_this_event[parameter] = (
                max(static_parameter_bounds[parameter][0], 10 ** min_bound_log),
                min(static_parameter_bounds[parameter][1], 10 ** max_bound_log)
            )


    logger.info("Setting up population for optimiser")
    rng = np.random.default_rng(seed=args.seed)
    min_bounds = np.array([bounds_this_event[p][0] for p in optimiser_parameters])
    max_bounds = np.array([bounds_this_event[p][1] for p in optimiser_parameters])
    population = rng.uniform(
        # Iterate using optimiser_parameters to ensure consistent ordering
        low=min_bounds,
        high=max_bounds,
        size=(args.n_particles, len(optimiser_parameters)),
    )
    # Make sure that the template itself is in the population
    population[0] = np.array([this_event_template[p] for p in optimiser_parameters])

    logger.info("Truncating data")
    # cuttingthe data to 2 months before and 1 month after
    merger_idx = int((time - original_data_epoch) / delta_t)
    start_idx = merger_idx - int(2 * 30 * 86400 / delta_t)
    end_idx = merger_idx + int(1 * 30 * 86400 / delta_t)
    data_thismerger = {
        channel: data[channel][start_idx:end_idx]
        for channel in args.channels
    }

    tlen = (end_idx - start_idx) * delta_t
    delta_f = 1. / tlen
    flen = len(data_thismerger[args.channels[0]]) // 2 + 1

    waveform_params_shared = {
        'f_lower': args.f_lower,
        'low-frequency-cutoff': args.f_lower, 
        'tdi': '1.5',
        't_offset': 0,
        'cutoff_deltat': 0,
        't_obs_start': int(tlen + 0.5),
        'delta_f': delta_f,
        'f_final': 1. / delta_t / 2 + delta_f / 2, # This is a hack to ensure we get the point at f_final returned
        'approximant':'BBHX_PhenomD',
        'mode_array':  [(2,2)],
        'length': flen,
    }

    logger.info("Setting up optimiser")
    optimizer = ps.single.GlobalBestPSO(
        n_particles=args.n_particles,
        dimensions=len(optimiser_parameters),
        options={'c1': 0.5, 'c2': 0.3, 'w': 0.9},
        bounds=(min_bounds, max_bounds),
        init_pos=population,
    )
    extra_args = {
        "data": data_thismerger,
        "psd": psd,
        "waveform_params_shared": waveform_params_shared
    }

    minimised_neg_stat, results = optimizer.optimize(
        compute_minus_network_snr,
        iters=int(10),
        n_processes=None,
        **extra_args,
    )

    maximised_stat = -minimised_neg_stat
    if maximised_stat > stat:
        logging.info("Statistic increased from %.1f to %.1f", stat, maximised_stat)
    elif maximised_stat == stat:
        logging.info("Statistic unchanged from %.1f", stat)
    else:
        logging.info("Statistic decreased from %.1f to %.1f", stat, maximised_stat)
        maximised_template = this_event_template

    
    # Get the maximised template from the results
    maximised_template = copy.deepcopy(waveform_params_shared)
    maximised_template.update({
        parameter: results[idx]
        for idx, parameter in enumerate(optimiser_parameters)
    })

    maximised_mtotal = conv.mtotal_from_mchirp_eta(
        maximised_template['mchirp'],
        maximised_template['eta']
    )
    maximised_mass1 = conv.mass1_from_mtotal_eta(
        maximised_mtotal,
        maximised_template['eta']
    )
    maximised_mass2 = conv.mass2_from_mass1_eta(
        maximised_mass1,
        maximised_template['eta']
    )
    maximised_template['mass1'] = maximised_mass1
    maximised_template['mass2'] = maximised_mass2
    print(this_event_template, maximised_template)

    # We need to update the lengths to the original data
    # for the signal timeseries
    maximised_template["t_obs_start"] = int(original_tlen + 0.5)
    maximised_template["delta_f"] = original_delta_f
    maximised_template["length"] = original_flen

    # Get the waveform for the maximised point
    maximised_wf = pycbc.waveform.get_fd_det_waveform(
        **maximised_template,
        channels=args.channels,
        delta_t=delta_t,
        ifos=args.channels
    )
    maximised_wf_ts = {
        channel: maximised_wf[channel].to_timeseries(delta_t=delta_t)
        for channel in args.channels
    }

    fig, ax = plt.subplots()
    ax.plot(
        maximised_wf_ts[args.channels[0]].sample_times,
        maximised_wf_ts[args.channels[0]],
        label="Maximised WF"
    )
    ax.plot(
        data[args.channels[0]].sample_times,
        data[args.channels[0]],
        label="Data"
    )
    ax.legend(loc='upper right')
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Amplitude')
    fig.savefig('test_wf_output.png')

    for channel in args.channels:
        signal_timeseries[channel] += maximised_wf_ts[channel]
        

logger.info("Done!")