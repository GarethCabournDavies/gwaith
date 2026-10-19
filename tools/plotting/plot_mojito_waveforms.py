"""Visualise MBHB waveforms for Mojito data.

This tool mirrors the Sangria waveform plotter but targets processed Mojito
channel files (root-level LISA_* datasets plus epoch/delta_t attributes).

For each source in a standardised truth file:
1. Plot the noisy data stream and, when available, the truth noiseless sum.
2. Regenerate a waveform using the selected generator.
"""

import argparse
import gc
import logging
import os

import h5py
import numpy as np
from matplotlib import pyplot as plt

from hwilas import add_hwilas_common_options, init_logging
from hwilas.data_processing.tdi import linear_AET_conversion
from hwilas.data_processing.truth import add_truth_arguments, load_truth_file
from hwilas.waveforms.unified_generation import generate_waveform

logger = logging.getLogger(__name__)

SECONDS_PER_DAY = 86400.0

XYZ_CHANNELS = ("LISA_X", "LISA_Y", "LISA_Z")
AET_CHANNELS = ("LISA_A", "LISA_E", "LISA_T")
VALID_CHANNELS = XYZ_CHANNELS + AET_CHANNELS


def _scalar(value):
    """Return a Python float from a scalar or a single-element array."""
    arr = np.asarray(value).ravel()
    if arr.size == 0:
        raise ValueError("Encountered an empty catalog field while reading truth data.")
    return float(arr[0])


def _pick_field(record, names, *, default=None):
    """Return the first available field value from a structured record."""
    for name in names:
        if name in record.dtype.names:
            return _scalar(record[name])
    if default is not None:
        return default
    raise KeyError(f"Missing required truth catalog field(s): {names}")


def load_mojito_channels(filename):
    """Load root-level LISA_* datasets and return (channels, epoch, delta_t)."""
    with h5py.File(filename, "r") as f:
        epoch = float(f.attrs["epoch"])
        delta_t = float(f.attrs["delta_t"])

        channels = {}
        for channel in VALID_CHANNELS:
            if channel in f:
                channels[channel] = np.asarray(f[channel][:], dtype=float).ravel()

    if not channels:
        raise ValueError(
            "No LISA_* channel datasets found in input file. "
            "Expected at least one of LISA_A/LISA_E/LISA_T or LISA_X/LISA_Y/LISA_Z."
        )

    return channels, epoch, delta_t


def load_signals_from_truth(truth):
    """Convert standardised truth catalog rows into waveform parameter dicts."""
    cat = truth.catalog
    if cat.dtype.names is None:
        raise ValueError("Truth catalog has no structured fields.")

    signals = []
    for i in range(len(cat)):
        rec = cat[i]
        spin1_polar_angle = None
        spin2_polar_angle = None
        spin1_polar_candidates = ["spin1_polar_angle", "PolarAngleOfSpin1"]
        spin2_polar_candidates = ["spin2_polar_angle", "PolarAngleOfSpin2"]
        if any(name in rec.dtype.names for name in spin1_polar_candidates):
            spin1_polar_angle = _pick_field(rec, spin1_polar_candidates)
        if any(name in rec.dtype.names for name in spin2_polar_candidates):
            spin2_polar_angle = _pick_field(rec, spin2_polar_candidates)

        signals.append(
            {
                "index": i,
                "mass1": _pick_field(
                    rec,
                    ["PrimaryMassSSBFrame", "mass1", "Mass1", "PrimaryMassSourceFrame"],
                ),
                "mass2": _pick_field(
                    rec,
                    ["SecondaryMassSSBFrame", "mass2", "Mass2", "SecondaryMassSourceFrame"],
                ),
                "spin1z": _pick_field(rec, ["spin1z", "Spin1z"], default=0.0),
                "spin2z": _pick_field(rec, ["spin2z", "Spin2z"], default=0.0),
                "distance": _pick_field(rec, ["distance", "Distance"], default=1.0),
                "eclipticlongitude": _pick_field(
                    rec,
                    ["eclipticlongitude", "EclipticLongitude", "Longitude"],
                ),
                "eclipticlatitude": _pick_field(
                    rec,
                    ["eclipticlatitude", "EclipticLatitude", "Latitude"],
                ),
                "inclination": _pick_field(rec, ["inclination", "Inclination"], default=0.0),
                "polarization": _pick_field(
                    rec,
                    ["polarization", "polarisation", "Polarization", "Polarisation"],
                    default=0.0,
                ),
                "phi_ref": _pick_field(
                    rec,
                    ["phi_ref", "coa_phase", "CoaPhase", "PhaseAtCoalescence"],
                    default=0.0,
                ),
                "coa_phase": _pick_field(
                    rec,
                    ["coa_phase", "phi_ref", "CoaPhase", "PhaseAtCoalescence"],
                    default=0.0,
                ),
                "tc": _pick_field(
                    rec,
                    [
                        "tc",
                        "CoalescenceTime",
                        "TimeCoalescencePhenomTPHMSSBFrame",
                    ],
                ),
                "spin1_polar_angle": spin1_polar_angle,
                "spin2_polar_angle": spin2_polar_angle,
            }
        )
    return signals


def generate_signal_waveform(
    signal,
    detector_times,
    *,
    channels=None,
    orbit="analytic",
    waveform_span_days=None,
    waveform="FastBHB",
    tdi_generation="1.5",
    generator_keywords=None, f_ref=None, data_time_origin=None,
):
    """Regenerate a detector-frame waveform and align it to detector_times.
    
    Parameters
    ----------
    signal : dict
        Signal parameters from truth catalog.
    detector_times : array_like
        The full data time grid; returned waveforms are aligned to it.
    channels : list of str, optional
        Subset of TDI channels to return (LISA_A, LISA_E, LISA_T).
    orbit : {'analytic', 'oem'}
        Orbit source for PhenomX. Ignored by FastBHB.
    waveform_span_days : float or None
        If given, only generate over this span around coalescence.
    waveform : str
        Approximant name.
    tdi_generation : str
        TDI version ('1.5' or '2.0').
    """
    from hwilas.waveforms.plugins import normalise_approximant, prepare_generator, reference_frequency
    waveform = normalise_approximant(waveform)
    signal = dict(signal)
    if data_time_origin is not None: signal['data_time_origin'] = data_time_origin
    tc = signal["tc"]
    t_min = float(detector_times[0])
    t_max = float(detector_times[-1])
    delta_t = float(detector_times[1] - detector_times[0])

    if waveform_span_days is None:
        gen_times = detector_times
    else:
        half = 0.5 * waveform_span_days * SECONDS_PER_DAY
        lo = max(t_min, tc - half)
        hi = min(t_max, tc + half)
        n = int(round((hi - lo) / delta_t)) + 1
        gen_times = np.linspace(lo, hi, n)

    signal.update(prepare_generator(waveform, orbit=orbit,data_time_origin=signal.get('data_time_origin'),
        start_time=float(gen_times[0]),end_time=float(gen_times[-1])+delta_t,keywords=generator_keywords))
    signal['f_ref'] = reference_frequency(waveform,signal.get('f_lower',1e-5),f_ref)
    # Use the unified waveform generation interface
    data = generate_waveform(
        signal,
        approximant=waveform,
        detector_times=gen_times,
        tdi_generation=tdi_generation,
        orbit=orbit,
        channels=channels,
    )

    if waveform_span_days is None:
        return {channel: np.asarray(arr, dtype=float) for channel, arr in data.items()}

    out = {}
    for channel, arr in data.items():
        full = np.zeros(len(detector_times), dtype=float)
        lo_idx = int(np.searchsorted(detector_times, gen_times[0], side="left"))
        hi_idx = int(np.searchsorted(detector_times, gen_times[-1], side="right"))
        n_copy = min(hi_idx - lo_idx, len(arr))
        if n_copy > 0:
            full[lo_idx : lo_idx + n_copy] = arr[:n_copy]
        out[channel] = full
    return out


def _index_range(t, t_start, t_end):
    """Return inclusive [start, end) sample indices for a time interval."""
    start = int(np.searchsorted(t, t_start, side="left"))
    end = int(np.searchsorted(t, t_end, side="right"))
    return max(0, start), min(len(t), end)


def plot_full_dataset(times, noisy, noiseless, signals, channels, out_path, full_window_days=None):
    """Plot the full dataset (or centered window) with coalescence markers."""
    fig, axes = plt.subplots(
        len(channels),
        1,
        figsize=(12, 4.5 * len(channels)),
        sharex=True,
    )
    if len(channels) == 1:
        axes = [axes]

    if full_window_days is not None:
        centre = signals[0]["tc"] if signals else times[len(times) // 2]
        half = 0.5 * full_window_days * SECONDS_PER_DAY
        lo, hi = _index_range(times, centre - half, centre + half)
    else:
        lo, hi = 0, len(times)

    for ax, channel in zip(axes, channels):
        ax.plot(
            times[lo:hi] / SECONDS_PER_DAY,
            noisy[channel][lo:hi],
            label="Noisy data",
            color="tab:blue",
            alpha=0.7,
        )
        if noiseless is not None and channel in noiseless:
            ax.plot(
                times[lo:hi] / SECONDS_PER_DAY,
                noiseless[channel][lo:hi],
                label="Noiseless injection",
                color="tab:red",
                linewidth=1.2,
            )

        for sig in signals:
            tc_days = sig["tc"] / SECONDS_PER_DAY
            if lo <= np.searchsorted(times, sig["tc"]) < hi or full_window_days is None:
                ax.axvline(tc_days, color="k", linestyle="--", alpha=0.5)
                ax.annotate(
                    f"#{sig['index']}",
                    xy=(tc_days, 0),
                    xytext=(4, -12),
                    textcoords="offset points",
                    fontsize=8,
                )

        ax.set_ylabel(f"Strain ({channel})")
        ax.legend(loc="upper right")

    axes[-1].set_xlabel("Time (days)")
    fig.suptitle("Mojito full dataset: noisy data and waveform context")
    fig.tight_layout(rect=(0, 0, 1, 0.98))
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    logger.info("Wrote full-dataset plot to %s", out_path)


def plot_signal_window(
    times,
    noisy,
    noiseless,
    generated,
    signal,
    channels,
    window_before_days,
    window_after_days,
    out_path,
):
    """Plot a zoomed time window around one signal's coalescence."""
    t_start = signal["tc"] - window_before_days * SECONDS_PER_DAY
    t_end = signal["tc"] + window_after_days * SECONDS_PER_DAY
    lo, hi = _index_range(times, t_start, t_end)
    if hi <= lo:
        logger.warning(
            "Signal #%s coalescence %.3f s outside data range; skipping.",
            signal["index"],
            signal["tc"],
        )
        return

    t_win = times[lo:hi] - signal["tc"]

    fig, axes = plt.subplots(
        len(channels),
        1,
        figsize=(12, 5.0 * len(channels)),
        sharex=True,
    )
    if len(channels) == 1:
        axes = [axes]

    for ax, channel in zip(axes, channels):
        ax.plot(
            t_win / SECONDS_PER_DAY,
            noisy[channel][lo:hi],
            label="Noisy data",
            color="tab:blue",
            alpha=0.7,
        )
        if noiseless is not None and channel in noiseless:
            ax.plot(
                t_win / SECONDS_PER_DAY,
                noiseless[channel][lo:hi],
                label="Noiseless injection",
                color="tab:red",
                linewidth=1.4,
            )
        if generated is not None and channel in generated:
            ax.plot(
                t_win / SECONDS_PER_DAY,
                generated[channel][lo:hi],
                label="Regenerated waveform",
                color="tab:green",
                linewidth=1.2,
                linestyle="--",
            )

        ax.axvline(0.0, color="k", linestyle=":", alpha=0.6)
        ax.set_ylabel(f"Strain ({channel})")
        ax.legend(loc="upper right")

    axes[-1].set_xlabel("Time relative to coalescence (days)")
    fig.suptitle(
        f"Signal #{signal['index']}: "
        f"M1={signal['mass1']:.0f} M2={signal['mass2']:.0f} Msun, "
        f"D={signal['distance']:.0f} Mpc"
    )
    fig.tight_layout(rect=(0, 0, 1, 0.98))
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    logger.info("Wrote signal window plot to %s", out_path)


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    add_hwilas_common_options(parser)
    add_truth_arguments(parser)

    for action in parser._actions:
        if "--truth-file" in action.option_strings:
            action.required = True
            break

    parser.add_argument(
        "--input-file",
        required=True,
        help="Path to processed Mojito HDF5 data with root LISA_* datasets.",
    )
    parser.add_argument(
        "--output-dir",
        default="plots/mojito_waveforms",
        help="Directory where plots and diagnostics are written.",
    )
    parser.add_argument(
        "--channels",
        nargs="+",
        choices=list(VALID_CHANNELS),
        default=["LISA_A"],
        help=(
            "One or more TDI channels to plot (default LISA_A). "
            f"Valid: {', '.join(VALID_CHANNELS)}."
        ),
    )
    parser.add_argument(
        "--signals",
        type=int,
        nargs="*",
        default=None,
        help="Truth-catalog indices to process. Defaults to all signals.",
    )
    parser.add_argument(
        "--window-before-days",
        type=float,
        default=0.2,
        help="Per-signal plot window start, in days before coalescence.",
    )
    parser.add_argument(
        "--window-after-days",
        type=float,
        default=0.02,
        help="Per-signal plot window end, in days after coalescence.",
    )
    parser.add_argument(
        "--full-window-days",
        type=float,
        default=None,
        help="If set, restrict full plot to this many days around first selected signal.",
    )
    parser.add_argument(
        "--no-waveform",
        action="store_true",
        help="Skip waveform regeneration and only plot available data.",
    )
    parser.add_argument(
        "--waveform-span-days",
        type=float,
        default=2.0,
        help="Days of waveform regenerated around each coalescence (default 2).",
    )
    from hwilas.waveforms.cli import add_waveform_arguments
    add_waveform_arguments(parser, dest='waveform', aliases=('--waveform',))
    parser.add_argument(
        "--tdi-generation",
        choices=["1.5", "2"],
        default="1.5",
        help="TDI generation forwarded to waveform generation (default 1.5).",
    )

    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()
    init_logging(args.verbose)

    os.makedirs(args.output_dir, exist_ok=True)

    logger.info("Loading truth file from %s", args.truth_file)
    truth = load_truth_file(args.truth_file)
    signals = load_signals_from_truth(truth)
    if args.signals is not None:
        wanted = set(args.signals)
        signals = [s for s in signals if s["index"] in wanted]
    logger.info("Loaded %d MBHB signal(s) from truth catalog.", len(signals))

    noiseless = None
    if truth.timeseries is not None:
        noiseless = {ch: np.asarray(arr, dtype=float) for ch, arr in truth.timeseries.items()}

    logger.info("Loading Mojito channels from %s", args.input_file)
    noisy, epoch, delta_t = load_mojito_channels(args.input_file)

    for channel in args.channels:
        if channel not in noisy and channel in AET_CHANNELS and all(ch in noisy for ch in XYZ_CHANNELS):
            # Allow plotting AET channels from XYZ-only files via on-the-fly rotation.
            AET = linear_AET_conversion(noisy["LISA_X"], noisy["LISA_Y"], noisy["LISA_Z"])
            noisy.update(AET)
            break

    for channel in args.channels:
        if channel not in noisy:
            parser.error(f"Channel {channel} not found in input file: {args.input_file}")

    n_samples = len(noisy[args.channels[0]])
    times = epoch + delta_t * np.arange(n_samples)

    full_path = os.path.join(args.output_dir, "full_dataset.png")
    plot_full_dataset(
        times,
        noisy,
        noiseless,
        signals,
        args.channels,
        full_path,
        full_window_days=args.full_window_days,
    )

    for sig in signals:
        logger.info("Generating waveform for signal #%s", sig["index"])
        generated = None
        if not args.no_waveform:
            try:
                generated = generate_signal_waveform(
                    sig,
                    times,
                    channels=args.channels,
                    orbit=args.orbit,
                    waveform_span_days=args.waveform_span_days,
                    waveform=args.waveform,
                    tdi_generation=args.tdi_generation, generator_keywords=args.generator_keywords,
                    f_ref=args.f_ref,data_time_origin=args.data_time_origin,
                )
            except Exception as exc:  # noqa: BLE001 - continue through failed signal
                logger.warning(
                    "Waveform generation failed for signal #%s: %s",
                    sig["index"],
                    exc,
                )

        out_path = os.path.join(args.output_dir, f"signal_{sig['index']:03d}.png")
        plot_signal_window(
            times,
            noisy,
            noiseless,
            generated,
            sig,
            args.channels,
            args.window_before_days,
            args.window_after_days,
            out_path,
        )

        # Release large waveform arrays between signals to avoid memory growth.
        generated = None
        gc.collect()

    logger.info("Done. Outputs written to %s", args.output_dir)


if __name__ == "__main__":
    main()
