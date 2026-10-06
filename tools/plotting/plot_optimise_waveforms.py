"""Plot initial and maximised waveforms for optimised events.

This tool writes ``initial_wf_<idx>.png`` and ``maximised_wf_<idx>.png``
plots for each optimised event.

For each event stored in the optimiser's ``--parameter-file`` it:

* looks up the initial (bank) template using the stored ``initial_template_id``,
* regenerates the initial template waveform from that bank row, and
* regenerates the maximised template waveform from the stored optimised
  parameters,

then plots both against the data chunk used for that event.

Because the initial template is not persisted by the optimiser (only its bank ID
is), this tool requires the same ``--bank-file`` and ``--input-file`` that were
passed to ``hwilas_optimise_snr_peaks``, in addition to the parameter file, data
files, PSD and channel options.

Example::

    python tools/plotting/plot_optimise_waveforms.py \
        --parameter-file output/optimise/optimise_parameters.hdf \
        --input-file output/search/coincident_triggers.hdf \
        --bank-file template_banks/bank.hdf \
        --data-files chunk_0/data.hdf chunk_1/data.hdf chunk_2/data.hdf \
        --psd-file psd_estimation/output/A_sangria_hm_SMOOTHED_PSD.txt \
        --optimisation-channels LISA_A LISA_E \
        --output-timeseries-channels LISA_A LISA_E LISA_T \
        --output-directory plots/optimise
"""

import argparse
import logging
import os
import sys

# Make the repository root importable so we can reuse the optimiser's helpers.
_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

import h5py
import numpy as np

from matplotlib import use

use('Agg')
from matplotlib import pyplot as plt

from gwaith.types import TimeSeries

from hwilas import add_hwilas_common_options, add_truth_arguments, init_logging, load_truth_file
from hwilas.psd.read import from_txt as psd_from_txt
# Higher-mode (l, m) array used by the plotting to build its higher-mode template.
DEFAULT_HM_MODE_ARRAY = [(2, 2), (2, 1), (3, 3), (3, 2), (4, 4), (4, 3)]
from hwilas.waveforms.constants import TIME_OFFSET_20_DEGREES
from hwilas.waveforms import get_fd_waveform

# Reuse the exact waveform-construction / normalisation helpers and event-chunk
# bookkeeping from the optimiser so the regenerated waveforms are identical to
# those plotted inline during the search.
from tools.search.optimise_snr_peaks import (
    _BANK_PARAMETERS,
    get_bank_parameters,
    get_data_file_starts_ends,
    identify_file_of_coinc,
    shift_and_normalise_waveform,
)

logger = logging.getLogger(__name__)

# Parameters stored per-event in the optimise parameter file. These are the
# same parameters that hwilas_optimise_snr_peaks writes out.
_PARAMETER_KEYS = [
    "initial_template_id",
    "initial_stat",
    "optimised_stat",
    "mchirp",
    "eta",
    "mass1",
    "mass2",
    "spin1z",
    "spin2z",
    "eclipticlongitude",
    "eclipticlatitude",
    "inclination",
    "tc",
    "polarization",
    "distance",
]


def load_arguments():
    parser = argparse.ArgumentParser(
        description="Plot initial and maximised waveforms for optimised events."
    )
    add_hwilas_common_options(parser)
    add_truth_arguments(parser)

    parser.add_argument(
        "--parameter-file",
        required=True,
        help="HDF5 file of optimised event parameters (from hwilas_optimise_snr_peaks)"
    )
    parser.add_argument(
        "--input-file",
        required=True,
        help="File containing coincident events (same --input-file passed to "
             "hwilas_optimise_snr_peaks); used to map each event to its data chunk"
    )
    parser.add_argument(
        "--bank-file",
        required=True,
        help="Template bank file (same --bank-file passed to hwilas_optimise_snr_peaks)"
    )
    parser.add_argument(
        "--data-files",
        nargs='+',
        required=True,
        help="File(s) containing the data, in the same order as passed to "
             "hwilas_optimise_snr_peaks"
    )
    parser.add_argument(
        "--psd-file",
        required=True,
        help="PSD file used for waveform normalisation / SNR"
    )
    parser.add_argument(
        "--optimisation-channels",
        nargs='+',
        default=['LISA_A', 'LISA_E'],
        help="Channels used for optimisation and the data frame. Default LISA_A LISA_E."
    )
    parser.add_argument(
        "--output-timeseries-channels",
        nargs='+',
        default=['LISA_A', 'LISA_E', 'LISA_T'],
        help="Channels to plot waveforms for. Default LISA_A LISA_E LISA_T."
    )
    parser.add_argument(
        "--f-lower",
        type=float,
        default=1e-6,
        help="Lower frequency bound for waveform generation"
    )
    parser.add_argument(
        "--tdi-generation",
        choices=['1.5', '2'],
        default='1.5',
        help="Which generation of TDI to use when generating the waveform"
    )
    parser.add_argument(
        "--output-directory",
        required=True,
        help="Directory in which to save the plots"
    )

    return parser.parse_args()


def load_parameters(parameter_file):
    """Load per-event optimised parameters from the parameter file."""
    events = {key: [] for key in _PARAMETER_KEYS}
    with h5py.File(parameter_file, 'r') as f:
        for key in _PARAMETER_KEYS:
            if key not in f:
                raise KeyError(f"Parameter file is missing dataset '{key}'")
            events[key] = np.asarray(f[key][:])
    return events


def load_event_data(event_data_file, optimisation_channels):
    """Load a single event's data chunk for the optimisation channels.

    Returns ``(data_thismerger, delta_t, tlen)`` where ``data_thismerger`` maps
    each optimisation channel to a :class:`gwaith.types.TimeSeries`.
    """
    with h5py.File(event_data_file, 'r') as data_file:
        data_thismerger = {}
        n_samples = None
        for channel in optimisation_channels:
            ts_data = data_file[channel]
            data_thismerger[channel] = TimeSeries(
                ts_data,
                epoch=data_file.attrs['epoch'],
                delta_t=data_file.attrs['delta_t']
            )
            n_samples = len(data_thismerger[channel])
        delta_t = data_file.attrs['delta_t']
        tlen = n_samples * delta_t
    return data_thismerger, delta_t, tlen


def _make_waveform_params_shared(delta_t, delta_f, flen, f_lower, tdi_generation):
    """Build the shared waveform-generator parameters (mirrors the optimiser)."""
    return {
        'f_lower': f_lower,
        'low-frequency-cutoff': f_lower,
        'tdi': tdi_generation,
        't_offset': TIME_OFFSET_20_DEGREES,
        'cutoff_deltat': 0,
        't_obs_start': 0.0,
        'delta_f': delta_f,
        'f_final': 1. / delta_t / 2 + delta_f / 2,
        'approximant': 'FastBHB',
        'mode_array': DEFAULT_HM_MODE_ARRAY,
        'length': flen,
    }


def _plot_waveform(
    fig_title,
    data_thismerger,
    initial_wf_ts,
    maximised_wf_ts,
    optimisation_channels,
    output_timeseries_channels,
    t_coalescence,
    output_path,
    truth_ts=None,
):
    """Plot the data plus the initial and maximised waveforms for one event.

    Mirrors the inline ``--testing-plots`` figures from the optimiser: the data
    and initial template are drawn for the optimisation channels, while the
    maximised template is drawn for all output timeseries channels. If
    ``truth_ts`` is given (a mapping of channel -> :class:`TimeSeries`), the raw
    injected-signal waveform from the truth file is also overlaid for the
    optimisation channels.
    """
    fig, ax = plt.subplots()
    for channel in optimisation_channels:
        ax.plot(
            data_thismerger[channel].sample_times / 86400,
            data_thismerger[channel],
            label=f"Data {channel}",
            alpha=0.2,
        )
    if truth_ts is not None:
        for channel in optimisation_channels:
            if channel in truth_ts:
                ax.plot(
                    truth_ts[channel].sample_times / 86400,
                    truth_ts[channel],
                    label=f"Truth {channel}",
                    color='r',
                    alpha=0.7,
                )
    if initial_wf_ts is not None:
        for channel in optimisation_channels:
            ax.plot(
                initial_wf_ts[channel].sample_times / 86400,
                initial_wf_ts[channel],
                label=f"Initial template {channel}",
                linestyle='--',
            )
    if maximised_wf_ts is not None:
        for channel in output_timeseries_channels:
            ax.plot(
                maximised_wf_ts[channel].sample_times / 86400,
                maximised_wf_ts[channel],
                label=f"Maximised template {channel}",
                linestyle=':',
            )
    ax.legend(loc='lower left')
    ax.set_title(fig_title)
    ax.set_xlabel('Time (days)')
    ax.set_ylabel('Amplitude')
    ax.set_xlim(
        t_coalescence / 86400 - 0.15,
        t_coalescence / 86400 + 0.05
    )
    ax.set_yscale('symlog', linthresh=1e-20)
    fig.savefig(output_path)
    plt.close(fig)


def main():
    args = load_arguments()

    # Some libraries may add a default StreamHandler to the root logger upon
    # import; clear any pre-existing handlers before initialising our own.
    logging.getLogger().handlers.clear()
    init_logging(args.verbose)

    logger.info("Starting")

    events = load_parameters(args.parameter_file)
    n_events = len(events["initial_template_id"])
    logger.info(
        "Loaded %d optimised events from %s", n_events, args.parameter_file
    )

    # Optionally load the raw injected-signal ("truth") waveform so it can be
    # overlaid on the per-event figures. The truth timeseries is a full-length
    # AET set, so we convert each channel to a TimeSeries once and reuse it for
    # every event (the plot x-range clips it to the event window).
    truth_ts = None
    if args.truth_file is not None:
        logger.info("Loading truth data from %s", args.truth_file)
        truth = load_truth_file(args.truth_file)
        logger.info(
            "Loaded %d %s truth source(s)", truth.n_sources, truth.source_group
        )
        if truth.timeseries is not None and truth.delta_t is not None:
            truth_ts = {
                channel: TimeSeries(
                    np.asarray(values, dtype=np.float64),
                    epoch=truth.epoch,
                    delta_t=truth.delta_t,
                )
                for channel, values in truth.timeseries.items()
            }
        else:
            logger.warning(
                "Truth file %s has no timeseries; the truth waveform overlay will be skipped.",
                args.truth_file,
            )

    # The initial template is identified by its bank id; the event's data chunk
    # and coalescence time come from the coincident-events input file. These are
    # aligned to the parameter-file rows (both are written out in the same
    # statistic-sorted order).
    with h5py.File(args.input_file, 'r') as input_file:
        coinc_tempid = np.asarray(input_file["template_id"][:])
        coinc_time_by_channel = {
            channel: np.asarray(input_file[f"time_{channel}"][:])
            for channel in args.optimisation_channels
        }

    if len(coinc_tempid) != n_events:
        raise ValueError(
            "The input file has %d events but the parameter file has %d; "
            "they must correspond one-to-one." % (len(coinc_tempid), n_events)
        )

    bank = get_bank_parameters(args.bank_file, coinc_tempid)

    data_file_starts, data_file_ends, _ = get_data_file_starts_ends(
        args.data_files,
        args.optimisation_channels[0]
    )
    # Mean of the two optimisation-channel times is used to locate the chunk.
    coinc_time = np.mean(
        [coinc_time_by_channel[c] for c in args.optimisation_channels], axis=0
    )
    coinc_chunks = identify_file_of_coinc(data_file_starts, data_file_ends, coinc_time)

    os.makedirs(args.output_directory, exist_ok=True)

    for idx in range(n_events):
        event = {key: float(events[key][idx]) for key in _PARAMETER_KEYS}
        initial_template_id = int(event["initial_template_id"])
        logger.info(
            "Plotting event %d (template_id=%s, mchirp=%.1f)",
            idx,
            initial_template_id,
            event["mchirp"],
        )

        # Initial template from the bank (same row as used by the optimiser).
        initial_template = {
            parameter: float(bank[parameter][idx]) for parameter in _BANK_PARAMETERS
        }
        t_coalescence = float(
            coinc_time_by_channel[args.optimisation_channels[0]][idx]
        )

        # Load this event's data chunk.
        event_data_file = args.data_files[coinc_chunks[idx]]
        logger.info("Using data file %s for event at time %.1f days",
                    event_data_file, t_coalescence / 86400)
        data_thismerger, delta_t, tlen = load_event_data(
            event_data_file, args.optimisation_channels
        )
        delta_f = 1. / tlen
        flen = int(tlen / delta_t) // 2 + 1

        psd = psd_from_txt(args.psd_file, flen, delta_f, delta_f, is_asd_file=False)
        psd.data[0] = np.inf
        psd.data[1] = np.inf

        waveform_params_shared = _make_waveform_params_shared(
            delta_t, delta_f, flen, args.f_lower, args.tdi_generation
        )

        # --- Initial (bank) template waveform -------------------------------
        initial_wf_params = dict(waveform_params_shared)
        initial_wf_params.update({
            "mass1": initial_template["mass1"],
            "mass2": initial_template["mass2"],
            "spin1z": initial_template["spin1z"],
            "spin2z": initial_template["spin2z"],
            "eclipticlongitude": initial_template["eclipticlongitude"],
            "eclipticlatitude": initial_template["eclipticlatitude"],
            "inclination": initial_template["inclination"],
            "polarization": initial_template["polarization"],
            "tc": tlen,  # merger at end of vector; shifted below to coalescence
            "distance": initial_template["distance"],
        })
        initial_wf_fd = get_fd_waveform(
            "FastBHB",
            **initial_wf_params,
            ifos=args.output_timeseries_channels,
        )
        initial_wf_ts, _, _ = shift_and_normalise_waveform(
            initial_wf_fd,
            t_coalescence - float(data_thismerger[args.optimisation_channels[0]].epoch),
            data_thismerger,
            psd,
            args.optimisation_channels,
            delta_t,
            args.output_timeseries_channels,
        )

        # Initial-template figure (data + initial template only).
        _plot_waveform(
            f"Initial waveform for event {idx}",
            data_thismerger,
            initial_wf_ts,
            None,
            args.optimisation_channels,
            args.output_timeseries_channels,
            t_coalescence,
            os.path.join(args.output_directory, f"initial_wf_{idx}.png"),
            truth_ts=truth_ts,
        )

        # --- Maximised template waveform ------------------------------------
        maximised_time_offset = (
            event["tc"] - float(data_thismerger[args.optimisation_channels[0]].epoch)
        )
        maximised_wf_params = dict(waveform_params_shared)
        maximised_wf_params.update({
            "mass1": event["mass1"],
            "mass2": event["mass2"],
            "spin1z": event["spin1z"],
            "spin2z": event["spin2z"],
            "eclipticlongitude": event["eclipticlongitude"],
            "eclipticlatitude": event["eclipticlatitude"],
            "inclination": event["inclination"],
            "polarization": event["polarization"],
            "tc": maximised_time_offset,  # merger placed at the optimised time
            "distance": 1.0,  # normalised by distance factor below
        })
        maximised_wf_fd = get_fd_waveform(
            "FastBHB",
            **maximised_wf_params,
            ifos=args.output_timeseries_channels,
        )
        maximised_wf_ts, _, _ = shift_and_normalise_waveform(
            maximised_wf_fd,
            0.0,  # tc already encodes the offset; no further shift needed
            data_thismerger,
            psd,
            args.optimisation_channels,
            delta_t,
            args.output_timeseries_channels,
        )

        # Maximised-template figure (data + initial + maximised).
        _plot_waveform(
            f"Maximised waveform for event {idx}",
            data_thismerger,
            initial_wf_ts,
            maximised_wf_ts,
            args.optimisation_channels,
            args.output_timeseries_channels,
            t_coalescence,
            os.path.join(args.output_directory, f"maximised_wf_{idx}.png"),
            truth_ts=truth_ts,
        )

        logger.info("Saved plots for event %d", idx)

    logger.info("Done!")


if __name__ == "__main__":
    main()
