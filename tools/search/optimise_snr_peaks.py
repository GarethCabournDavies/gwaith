"""
Optimise SNR for peaks found in coincidences
"""

import gc
import argparse
import copy
import logging
import h5py
import numpy as np
import pyswarms as ps

from gwaith.types import TimeSeries

from hwilas import add_hwilas_common_options, init_logging
from hwilas.conversions import mchirp_eta_from_mass1_mass2, mass1_mass2_from_mchirp_eta
from hwilas.filter import matched_filter_frequencyseries, sigmasq_frequencyseries
from hwilas.psd.read import from_txt as psd_from_txt
from hwilas.waveforms.bbhx_utils import TIME_OFFSET_20_DEGREES
from hwilas.waveforms.bbhx_phenom import DEFAULT_HM_MODE_ARRAY
from hwilas.types import KeyValueAction
from hwilas.waveforms import get_fd_waveform
from hwilas.template_banks.helpers import waveform_from_bank

logger = logging.getLogger(__name__)


def build_aligned_waveform(v, data, waveform_params_shared, raise_err=False, waveform_channels=None):
    mass1, mass2 = mass1_mass2_from_mchirp_eta(v[0], v[1])

    if waveform_channels is None:
        waveform_channels = list(data.keys())

    for channel in data.keys():  # Use data.keys() (optimisation channels) to get delta_t and epoch
        try:
            delta_t = data[channel].delta_t
            data_start_time = data[channel].epoch
            break
        except KeyError:
            continue
    else:
        raise KeyError(
            "No channels from passed option are present in data"
        )

    
    waveform_params = copy.deepcopy(waveform_params_shared)
    waveform_params.setdefault("distance", 1.0)

    # The time of coalescence is given as an absolute GPS time in v[7].
    # We need to convert it to a time relative to the start of the data segment
    # for the waveform generator.
    coalescence_in_segment = v[7] - data_start_time

    waveform_params.update({
        "mass1": mass1,
        "mass2": mass2,
        "spin1z": v[2],
        "spin2z": v[3],
        "eclipticlongitude": v[4],
        "eclipticlatitude": v[5],
        "inclination": v[6],
        "tc": coalescence_in_segment,
        "polarization": v[8],
    })

    try:
        wf = get_fd_waveform(
            **waveform_params,
            ifos=waveform_channels
        )
    except RuntimeError:
        if raise_err:
            raise
        return None

    # The time alignment is now handled by setting `tc` directly in the
    # waveform generator, so no explicit time shift is needed.
    return wf


def compute_network_snr_core(v, data, psd, waveform_params_shared, optimisation_channels, raise_err=False):
    wf = build_aligned_waveform(
        v,
        data,
        waveform_params_shared,
        raise_err=raise_err,
        # Generate waveform for all channels that might be needed for output, but only use optimisation_channels for SNR
        waveform_channels=list(data.keys()) # data here contains only optimisation channels
    )
    if wf is None:
        # The waveform generation has failed - instead of actually failing,
        # just say that this waveform is not allowed.
        return -np.inf

    snrsq = 0.0
    for channel in optimisation_channels:
        snr_ts = matched_filter_frequencyseries(
            wf[channel],
            data[channel].to_frequencyseries(),
            psd=psd,
            n_samples=len(data[channel]),
            epoch=float(data[channel].epoch),
        )

        # v[7] is the absolute GPS time of coalescence.
        # The snr_ts.sample_times represent the merger time of the template.
        # Find the SNR at the specified coalescence time.
        time_offset_from_epoch = v[7] - snr_ts.epoch
        index_at_coalescence = int(round(time_offset_from_epoch / snr_ts.delta_t))

        if 0 <= index_at_coalescence < len(snr_ts.data):
            snr_at_coalescence = snr_ts.data[index_at_coalescence]
        else:
            logger.warning("Coalescence time %.3f is outside the range of SNR time series for channel %s. Returning -inf.", v[7], channel)
            return -np.inf

        snrsq += abs(snr_at_coalescence) ** 2

    return snrsq ** 0.5


def get_snr_and_distance_factor(wf, data, psd, optimisation_channels):
    """
    Calculate the network SNR and distance factor for a given waveform.

    The distance factor is the value to divide a unit-distance (1 Mpc)
    waveform by to match the amplitude of the signal in the data.
    """
    snrsq = 0.0 # This is the network SNR squared
    hhsq = 0.0
    for channel in optimisation_channels:
        snr_ts = matched_filter_frequencyseries(
            wf[channel],
            data[channel].to_frequencyseries(),
            psd=psd,
            n_samples=len(data[channel]),
            epoch=float(data[channel].epoch),
        )

        snr_at_coalescence = snr_ts.abs().max()

        snrsq += snr_at_coalescence ** 2
        hhsq += sigmasq_frequencyseries(wf[channel], psd)

    network_snr = snrsq ** 0.5
    if hhsq <= 0 or snrsq <= 0:
        return network_snr, np.inf

    distance_factor = (hhsq ** 0.5) / network_snr
    return network_snr, distance_factor


def shift_and_normalise_waveform(
    wf_fd,
    coalescence_in_segment,
    data,
    psd, # This PSD is for the optimisation channels
    optimisation_channels,
    delta_t,
    output_channels
):
    """
    Takes a frequency domain waveform with the merger at the end, shifts it
    to the correct time, calculates the distance factor, normalises it, and
    returns a time series version for plotting, along with the calculated
    network SNR and distance factor.
    """
    # 1. Align waveform to data
    wf_aligned = {} # This will contain all output channels
    # The epoch should be the same for all channels, so we can just get it
    # from the first optimisation channel which is guaranteed to be in data.
    epoch = float(data[optimisation_channels[0]].epoch)
    for channel in output_channels:
        wf_aligned[channel] = wf_fd[channel].copy()
        wf_aligned[channel].epoch = epoch
        wf_aligned[channel] = wf_aligned[channel].cyclic_time_shift(
            coalescence_in_segment
        )

    # 2. Get SNR and distance factor using the aligned waveform
    network_snr, distance_factor = get_snr_and_distance_factor(
        wf_aligned,
        data, # This data contains only optimisation channels
        psd,
        optimisation_channels,
    )

    # 3. Normalise the aligned waveform and convert to a time series
    wf_ts = {} # This will contain all output channels
    for channel in output_channels:
        if np.isfinite(distance_factor):
            wf_aligned[channel] /= distance_factor
        wf_ts[channel] = wf_aligned[channel].to_timeseries(delta_t=delta_t)

    return wf_ts, network_snr, distance_factor

    
def compute_minus_network_snr(positions, **kwargs):
    costs = np.empty(len(positions), dtype=np.float64)
    for idx, v in enumerate(positions):
        nsnr = compute_network_snr_core(v, **kwargs)
        logger.debug("snr: %.3f", nsnr)
        costs[idx] = -nsnr
    return costs


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
        default=60,
        help="When deciding the events to optimise / add to signal timeseries, use this threshold. "
        "Default 60."
    )
    parser.add_argument(
        "--minimum-peak-separation",
        type=float,
        default=604800,
        help="Minimum separation of peaks to be optimised. Seconds. Default 604800 (1 week)"
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
        required=True,
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
    parser.add_argument(
        "--output-timeseries-channels",
        nargs='+',
        default=['LISA_A', 'LISA_E', 'LISA_T'], # Default to all three for output
        help="Channels that will be used for optimisation. "
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

    args = parser.parse_args()

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
    above_threshold = coinc_stat > args.statistic_threshold
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

    # Sort the events by time to make the next step easier
    time_sort = np.argsort(coinc_time)
    coinc_stat = coinc_stat[time_sort]
    coinc_time = coinc_time[time_sort]
    for channel in args.optimisation_channels:
        coinc_times_by_channel[channel] = (
            coinc_times_by_channel[channel][time_sort]
        )
    coinc_tempid = coinc_tempid[time_sort]

    # Now, we use a greedy algorithm to sort through the remaining events and 
    # remove anything within the minimum separation window.
    # We will optimise however many are above the threshold and separated by the
    # minimum peak separation
    loudest_idx = 0
    while True:
        if not any(abs(np.diff(coinc_time)) < args.minimum_peak_separation):
            break

        # If this is the Nth iteration, the the Nth loudest of the remaining
        # triggers
        stat_sort = np.argsort(coinc_stat)
        loudest_time = coinc_time[stat_sort[loudest_idx]]

        # These are the events to be removed
        within_window = abs(coinc_time - loudest_time) < args.minimum_peak_separation
        # Dont remove the event itself though
        within_window[loudest_idx] = False

        logger.info(
            "Removing %d events within %.1f days of loudest event at time %.3f",
            np.sum(within_window),
            args.minimum_peak_separation / 86400,
            loudest_time / 86400
        )
        coinc_stat = coinc_stat[~within_window]
        coinc_time = coinc_time[~within_window]
        for channel in args.optimisation_channels:
            coinc_times_by_channel[channel] = (
                coinc_times_by_channel[channel][~within_window]
            )
        coinc_tempid = coinc_tempid[~within_window]

        loudest_idx += 1

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
            "Event %d at time %.1f days with statistic %.3f, template %d",
            i,
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

def identify_file_of_coinc(data_starts, data_ends, coinc_times):
    """For each coinc time, identify which data file it belongs within"""
    n_coincs = len(coinc_times)
    coinc_chunks = np.zeros(n_coincs, dtype=np.uint32)
    for i, coinc_time in enumerate(coinc_times):
        start_idx = np.searchsorted(data_starts, coinc_time) - 1
        end_idx = np.searchsorted(data_ends, coinc_time)
        if start_idx == end_idx:
            # The coinc lies perfectly within exactly one chunk
            logger.debug(
                "The event at %.3f days is entirely in chunk %d",
                coinc_time / 86400,
                start_idx
            )
            coinc_chunks[i] = start_idx
        elif (end_idx - start_idx) == 1:
            # The coinc straddles the overlap between two chunks
            logger.debug(
                "The event at %.3f days is in the overlap between chunks %d and %d",
                coinc_time / 86400,
                start_idx,
                end_idx
            )
            # In this case always return the earlier chunk
            coinc_chunks[i] = start_idx
        else:
            raise ValueError(
                "Could Not identify the chunk for event at %.3f days",
                coinc_time / 86400
            )
    return coinc_chunks


def main():

    args = load_arguments()

    # Some libraries, like pyswarms, may add a default StreamHandler to the root
    # logger upon import. This can lead to duplicate log messages. To prevent this,
    # we find and remove any pre-existing StreamHandlers before initializing
    # our own logging.
    logging.getLogger().handlers.clear()

    init_logging(args.verbose)

    coinc_stat, coinc_time, coinc_times_by_channel, coinc_tempid = (
        get_events_to_optimise(args)
    )

    bank = get_bank_parameters(args.bank_file, coinc_tempid)

    logger.info("Getting start and end times for data files")
    data_file_starts, data_file_ends, delta_t = get_data_file_starts_ends(
        args.data_files,
        args.optimisation_channels[0]
    )

    full_duration = data_file_ends[-1] - data_file_starts[0]
    n_samples_total = int(full_duration / delta_t)

    logger.debug(
        "Data files start, end at %s",
        " ".join(f"{dfs/86400:.1f}-{dfe/86400:.1f}" for dfs, dfe in zip(data_file_starts, data_file_ends, strict=True))
    )

    logger.info("Identifying which coincs are in which data files")
    coinc_chunks = identify_file_of_coinc(
        data_file_starts,
        data_file_ends,
        coinc_time
    )

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

    optimised_parameters = {
        'initial_template_id': [],
        'initial_stat': [],
        'optimised_stat': [],
        'mchirp': [],
        'eta': [],
        'mass1': [],
        'mass2': [],
        'spin1z': [],
        'spin2z': [],
        'eclipticlongitude': [],
        'eclipticlatitude': [],
        'inclination': [],
        'tc': [],
        'polarization': [],
        'distance': [],
    }

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

    output_timeseries = {
        channel: TimeSeries(
            data = np.zeros(n_samples_total, dtype=np.float32),
            epoch=float(data_file_starts[0]),
            delta_t=delta_t
        )
        for channel in args.output_timeseries_channels
    }

    

    for idx, (stat, time) in enumerate(zip(coinc_stat, coinc_time)):
        logger.info("Event %d at time %.1f days with statistic %.3f", idx, time / 86400, stat)
        # Find data file for this event
        event_data_file = args.data_files[coinc_chunks[idx]]

        logger.info(f"Using data file {event_data_file} for event at time {time}")

        logger.info("Getting parameters")
        initial_template_id = coinc_tempid[idx]
        initial_template = {
            parameter: bank[parameter][idx]
            for parameter in _BANK_PARAMETERS
        }
        initial_template['coa_phase'] = bank['coa_phase'][idx]
        initial_template['phi_ref'] = bank['phi_ref'][idx]

        mchirp_init, eta_init = mchirp_eta_from_mass1_mass2(
            initial_template['mass1'],
            initial_template['mass2']
        )
        # The bank stores mass1/mass2 but the optimiser works in terms of
        # chirp mass and symmetric mass ratio, so store those derived values
        # here for use when setting bounds and the initial population.
        initial_template['mchirp'] = mchirp_init
        initial_template['eta'] = eta_init
        # Use the trigger time of the first channel as the coalescence time for
        # the initial point / optimisation. The other channels are shifted by
        # their per-channel trigger times when plotting, so that each channel's
        # waveform peaks at its own trigger time.
        initial_template['tc'] = float(
            coinc_times_by_channel[args.optimisation_channels[0]][idx]
        )

        logger.info("Setting up the initial population of particles for the optimiser")

        # Start from a fresh copy of static bounds for each event so there is no
        # mutation carried across loop iterations.
        bounds_this_event = {
            parameter: static_parameter_bounds[parameter]
            for parameter in optimiser_parameters
        }
        for parameter in optimiser_parameters:
            if parameter in args.bounds_window_relative_to_initial:
                event_max_bound = initial_template[parameter] + args.bounds_window_relative_to_initial[parameter]
                event_min_bound = initial_template[parameter] - args.bounds_window_relative_to_initial[parameter]
                bounds_this_event[parameter] = (
                    max(static_parameter_bounds[parameter][0], event_min_bound),
                    min(static_parameter_bounds[parameter][1], event_max_bound)
                )
            elif f"log_{parameter}" in args.bounds_window_relative_to_initial:
                max_bound_log = np.log10(
                        initial_template[parameter]
                    ) + args.bounds_window_relative_to_initial[f"log_{parameter}"]

                min_bound_log = np.log10(
                    initial_template[parameter]
                 ) - args.bounds_window_relative_to_initial[f"log_{parameter}"]

                bounds_this_event[parameter] = (
                    max(static_parameter_bounds[parameter][0], 10 ** min_bound_log),
                    min(static_parameter_bounds[parameter][1], 10 ** max_bound_log)
                )


        logger.info("Setting up population for optimiser")
        rng = np.random.default_rng(seed=args.seed)
        min_bounds = np.array(
            [bounds_this_event[p][0] for p in optimiser_parameters]
        )
        max_bounds = np.array(
            [bounds_this_event[p][1] for p in optimiser_parameters]
        )
        population = rng.uniform(
            # Iterate using optimiser_parameters to ensure consistent ordering
            low=min_bounds,
            high=max_bounds,
            size=(args.n_particles, len(optimiser_parameters)),
        )

        logger.debug(
            "Making sure that the template itself is in the population by "
            "overwriting the first point"
        )
        population[0] = np.array(
            [initial_template[p] for p in optimiser_parameters]
        )
        for i, p in enumerate(optimiser_parameters):
            logger.debug(
                "Parameter %s, initial position: %.6f, bounds: [%.6f, %.6f]",
                p,
                population[0, i],
                min_bounds[i],
                max_bounds[i]
            )

        logger.info("Loading and truncating data for this event")
        with h5py.File(event_data_file, 'r') as data_file:

            data_thismerger = {}
            for channel in args.optimisation_channels:
                ts_data = data_file[channel]
                data_thismerger[channel] = TimeSeries(
                    ts_data,
                    epoch=data_file.attrs['epoch'],
                    delta_t=data_file.attrs['delta_t']
                )
                n_samples = len(data_thismerger[channel])
            delta_t = data_file.attrs['delta_t']
            tlen = n_samples * delta_t
        delta_f = 1. / tlen
        flen = n_samples // 2 + 1

        psd = psd_from_txt(
            args.psd_file,
            flen,
            delta_f,
            delta_f,
            is_asd_file=False,
        )
        psd.data[0] = np.inf
        psd.data[1] = np.inf

        waveform_params_shared = {
            'f_lower': args.f_lower,
            'low-frequency-cutoff': args.f_lower, 
            'tdi': args.tdi_generation,
            't_offset': TIME_OFFSET_20_DEGREES,
            'cutoff_deltat': 0,
            't_obs_start': tlen,
            'delta_f': delta_f,
            'f_final': 1. / delta_t / 2 + delta_f / 2, # This is a hack to ensure we get the point at f_final returned
            'approximant':'BBHX_PhenomHM',
            'mode_array':  DEFAULT_HM_MODE_ARRAY,
            'length': flen,
            'tc': tlen
        }

        logger.info(
            "Getting SNR for the initial event using time from channel %s",
            args.optimisation_channels[0]
        )
        v_initial_params = np.array([initial_template[p] for p in optimiser_parameters])

        t_coalescence = v_initial_params[7]
        # The coalescence time (v[7] / this_event_template['tc']) is an absolute
        # GPS time. Convert it to a time relative to the start of the data segment,
        # which is what we need to place the merger within the data's frame.
        coalescence_in_segment = (
            t_coalescence - float(data_thismerger[args.optimisation_channels[0]].epoch)
        )

        # Generate the initial template waveform for all output channels.
        # This is done in exactly the same way as the
        # matched_filter_search.py search: with tc set to the full data duration so
        # that the merger sits at the *end* of the waveform vector (see
        # get_fd_waveform_from_bbhx, which cyclically shifts the merger to the end).
        # The bank's distance is used so the template amplitude matches the injected
        # signal in the data.
        waveform_params_this_event = copy.deepcopy(waveform_params_shared)
        waveform_params_this_event['coa_phase'] = float(initial_template['coa_phase'])
        waveform_params_this_event['phi_ref'] = float(initial_template['phi_ref'])

        initial_wf_params = copy.deepcopy(waveform_params_this_event)
        initial_wf_params.update({
            "mass1": initial_template['mass1'],
            "mass2": initial_template['mass2'],
            "spin1z": initial_template['spin1z'],
            "spin2z": initial_template['spin2z'],
            "eclipticlongitude": initial_template['eclipticlongitude'],
            "eclipticlatitude": initial_template['eclipticlatitude'],
            "inclination": initial_template['inclination'],
            "polarization": initial_template['polarization'],
            "coa_phase": initial_template['coa_phase'],
            "phi_ref": initial_template['phi_ref'],
            "tc": tlen,
            "distance": initial_template['distance']
        })


        initial_wf_fd = get_fd_waveform(
            **initial_wf_params,
            ifos=args.output_timeseries_channels, # Generate for all output channels
        )

        initial_wf_ts, initial_stat, estimated_distance_factor = shift_and_normalise_waveform(
            initial_wf_fd,
            coalescence_in_segment,
            data_thismerger,
            psd, # This PSD is for the optimisation channels
            args.optimisation_channels, # Only use optimisation channels for SNR calculation
            delta_t,
            args.output_timeseries_channels # Output all specified channels
        )
        logger.debug("Cleaning up `initial_wf_fd` after last use")
        del initial_wf_fd

        logger.info("Initial waveform network SNR: %.3f", initial_stat)
        logger.info(
            "Estimated distance factor for initial waveform: %.3f",
            estimated_distance_factor
        )

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
            "waveform_params_shared": waveform_params_this_event,
            "optimisation_channels": args.optimisation_channels # Pass optimisation channels to SNR core
        }

        minimised_neg_stat, results = optimizer.optimize(
            compute_minus_network_snr,
            iters=10,
            n_processes=None,
            **extra_args,
        )
        logger.debug("Cleaning up `optimizer` after last use")
        del optimizer

        maximised_stat = -minimised_neg_stat
        if maximised_stat > initial_stat:
            logger.info("Statistic increased from %.1f to %.1f", initial_stat, maximised_stat)
        elif maximised_stat == initial_stat:
            logger.info("Statistic unchanged from %.1f", initial_stat)
        else:
            logger.warning(
                "Statistic decreased from %.1f to %.1f - this shouldn't happen!",
                initial_stat,
                maximised_stat
            )

        logger.debug(
            "Maximised template: %s",
            ", ".join(
                f"{parameter}: {results[idx]}"
                for idx, parameter
                in enumerate(optimiser_parameters)
            )
        
        )

        logger.debug("Getting the maximised template from the results")
        maximised_template = copy.deepcopy(waveform_params_shared)
        maximised_template.update({
            parameter: results[idx]
            for idx, parameter
            in enumerate(optimiser_parameters)
        })
        maximised_template['coa_phase'] = waveform_params_this_event.get('coa_phase', 0.0)
        maximised_template['phi_ref'] = waveform_params_this_event.get('phi_ref', 0.0)

        maximised_mass1, maximised_mass2 = mass1_mass2_from_mchirp_eta(
            maximised_template['mchirp'],
            maximised_template['eta']
        )
        maximised_template['mass1'] = maximised_mass1
        maximised_template['mass2'] = maximised_mass2


        maximised_wf = build_aligned_waveform( # Build waveform for all output channels
            results,
            data_thismerger,
            waveform_params_this_event,
            raise_err=True,
            waveform_channels=args.output_timeseries_channels
        )
        logger.debug("Cleaning up `results` after last use")
        del results

        maximised_time_offset = maximised_template['tc'] - float(data_thismerger[args.optimisation_channels[0]].epoch)

        # Capture the actual start time of this event's data file. The per-event
        # waveform is anchored to this epoch, so we need it later (after
        # `data_thismerger` has been deleted) to place the waveform at the
        # correct absolute position within the full-data timeseries.
        event_data_file_epoch = float(data_thismerger[args.optimisation_channels[0]].epoch)

        maximised_wf_ts, _, maximised_distance_factor = shift_and_normalise_waveform(
            maximised_wf,
            maximised_time_offset,
            data_thismerger,
            psd,
            args.optimisation_channels,
            delta_t,
            args.output_timeseries_channels
        )
        logger.debug("Cleaning up `maximised_wf` and `psd` after last use")
        del maximised_wf
        del psd
        maximised_template['distance'] = maximised_distance_factor

        logger.debug("Cleaning up `initial_wf_ts` and `data_thismerger` after last use")
        del initial_wf_ts
        del data_thismerger


        optimised_parameters['initial_template_id'].append(int(initial_template_id))
        optimised_parameters['initial_stat'].append(float(initial_stat))
        optimised_parameters['optimised_stat'].append(float(maximised_stat))

        for parameter in [
            'mchirp', 'eta', 'mass1', 'mass2', 'spin1z', 'spin2z',
            'eclipticlongitude', 'eclipticlatitude', 'inclination', 'tc',
            'polarization', 'distance'
        ]:
            optimised_parameters[parameter].append(float(maximised_template[parameter]))
        logger.debug("Cleaning up `maximised_template` and `data_thismerger` after last use")
        del maximised_template

        for channel in args.output_timeseries_channels:
            logger.debug("Resize and timeshift to the full-data timeseries")
            maximised_wf_ts[channel].resize(n_samples_total)
            # Shift by this event's own data-file epoch relative to the start of
            # the full-data buffer. This is NOT the chunk boundary
            # (data_file_starts[coinc_chunks[idx]]): a file can begin at an offset
            # from its chunk boundary (e.g. front padding), so we must anchor on
            # the actual file epoch to place the event at the correct absolute time.
            # NB: cyclic_time_shift returns a NEW TimeSeries and does NOT mutate in
            # place, so the result MUST be captured -- otherwise the shift is
            # silently discarded and the event is added at the wrong (unshifted)
            # position within the full-data buffer.
            maximised_wf_ts[channel] = maximised_wf_ts[channel].cyclic_time_shift(
                event_data_file_epoch - data_file_starts[0]
            )
            maximised_wf_ts[channel].epoch = data_file_starts[0]

        for channel in args.output_timeseries_channels: # Add to all output channels
            output_timeseries[channel] += maximised_wf_ts[channel]

        logger.debug("Cleaning up `maximised_wf_ts` after last use")
        del maximised_wf_ts
        gc.collect()

    if args.output_parameter_file is not None:
        logger.info("Writing optimised parameters to %s", args.output_parameter_file)
        with h5py.File(args.output_parameter_file, 'w') as output_parameter_file:
            for key, vals in optimised_parameters.items():
                output_parameter_file.create_dataset(key, data=np.array(vals), compression='gzip')

    if args.output_timeseries_file is not None:
        logger.info("Writing optimised signal timeseries to %s", args.output_timeseries_file)
        with h5py.File(args.output_timeseries_file, 'w') as output_ts_file:
            output_ts_file.attrs['epoch'] = float(output_timeseries[args.optimisation_channels[0]].epoch)
            output_ts_file.attrs['delta_t'] = float(delta_t) # delta_t is consistent across all channels
            for channel in args.output_timeseries_channels: # Write all output channels
                output_ts_file.create_dataset(
                    channel,
                    data=output_timeseries[channel].data,
                    compression='gzip',
                )

    logger.info("Done!")