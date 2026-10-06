"""Visualise MBHB waveforms from a standardised truth file against the data.

This tool loads a standardised truth HDF5 file (as produced by
``hwilas_extract_truth``) and, for each injected massive binary black hole
(MBHB) signal in the catalog:

* regenerates the detector-frame waveform with the LDC FastBHB (lisabeta)
  generator (:func:`hwilas.waveforms.get_fd_waveform`), and
* compares it against the noisy data stream (``obs/tdi``) and the noiseless injected
  signal from the truth file.

Two kinds of figure are produced:

1. A **full-dataset** overview showing the whole time series for one channel, with the
   coalescence times of every signal marked.
2. One **per-signal window** per MBHB, zoomed to a configurable span around its
   coalescence time (default 0.2 days before and 0.02 days after), showing noisy data,
   noiseless injection and the regenerated waveform overlaid.

Example::

    hwilas_extract_truth \
        --data-file datasets/LDC2_sangria_hm_training.hdf \
        --source-group mbhb \
        --output-file truth_mbhb.h5

    hwilas_plot_sangria_waveforms \
        --truth-file truth_mbhb.h5 \
        --input-file datasets/LDC2_sangria_hm_training.hdf \
        --output-dir plots/sangria \
        --channels LISA_A LISA_E \
        --signals 0 1 2
"""

import argparse
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

# Seconds per day, used for the default window lengths.
SECONDS_PER_DAY = 86400.0

# HDF5 paths within a Sangria file (for the noisy data stream).
NOISY_TDI_PATH = "obs/tdi"

# Valid TDI channels: the XYZ frame and its AET rotation.
XYZ_CHANNELS = ("LISA_X", "LISA_Y", "LISA_Z")
AET_CHANNELS = ("LISA_A", "LISA_E", "LISA_T")
VALID_CHANNELS = XYZ_CHANNELS + AET_CHANNELS


def _scalar(value):
    """Return a python float from a possibly single-element (object) array."""
    if value is None:
        raise ValueError("Encountered a None catalog field; cannot convert to float.")
    arr = np.asarray(value).ravel()
    return float(arr[0])


def load_noisy_tdi_channels(filename):
    """Load the noisy XYZ TDI dataset and return it as an AET channel dictionary.

    Returns a dict mapping ``'LISA_A'``, ``'LISA_E'`` and ``'LISA_T'`` to 1-D numpy
    arrays, so that the same channel name can be used for noisy data, the noiseless
    injection and the (AET-converted) regenerated waveform.
    """
    with h5py.File(filename, "r") as f:
        ds = f[NOISY_TDI_PATH]
        X = np.asarray(ds["X"], dtype=float).ravel()
        Y = np.asarray(ds["Y"], dtype=float).ravel()
        Z = np.asarray(ds["Z"], dtype=float).ravel()
    return linear_AET_conversion(X, Y, Z)


def load_signals_from_truth(truth):
    """Convert a :class:`hwilas.data_processing.truth.TruthData` catalog into a list
    of per-signal parameter dictionaries.

    The standard truth file preserves the Sangria LDC field names (``Mass1``,
    ``Spin1``, ``PolarAngleOfSpin1``, etc.), so the same key mapping applies as
    before -- we just read from the structured array rather than a raw h5py dataset.
    """
    cat = truth.catalog
    signals = []
    for i in range(len(cat)):
        rec = cat[i]
        spin1_mag = _scalar(rec["Spin1"])
        spin2_mag = _scalar(rec["Spin2"])
        polar_angle_spin1 = _scalar(rec["PolarAngleOfSpin1"])
        polar_angle_spin2 = _scalar(rec["PolarAngleOfSpin2"])
        signals.append(
            {
                "index": i,
                "mass1": _scalar(rec["Mass1"]),
                "mass2": _scalar(rec["Mass2"]),
                # Line-of-sight spin components (|chi| cos theta).
                "spin1z": spin1_mag * np.cos(polar_angle_spin1),
                "spin2z": spin2_mag * np.cos(polar_angle_spin2),
                "spin1_polar_angle": polar_angle_spin1,
                "spin2_polar_angle": polar_angle_spin2,
                "distance": _scalar(rec["Distance"]),
                "eclipticlongitude": _scalar(rec["EclipticLongitude"]),
                "eclipticlatitude": _scalar(rec["EclipticLatitude"]),
                # The catalog's initial L-angles define the sky orientation; they
                # are used as the inclination/polarization proxies for the
                # regenerated waveform. Phase at coalescence is phi_ref.
                "inclination": _scalar(rec["InitialPolarAngleL"]),
                "polarization": _scalar(rec["InitialAzimuthalAngleL"]),
                "phi_ref": _scalar(rec["PhaseAtCoalescence"]),
                "coa_phase": _scalar(rec["PhaseAtCoalescence"]),
                "tc": _scalar(rec["CoalescenceTime"]),
            }
        )
    return signals


def generate_signal_waveform(
    signal,
    detector_times,
    orbit="analytic",
    waveform_span_days=None,
    waveform="FastBHB",
    tdi_generation="1.5",
):
    """Regenerate the AET-channel waveform for a single catalog signal.

    Returns a dict mapping ``'LISA_A'``, ``'LISA_E'`` and ``'LISA_T'`` to 1-D arrays
    aligned with ``detector_times`` (zeros outside the generation span).

    Parameters
    ----------
    detector_times : array_like
        The full data time grid; returned waveforms are aligned to it.
    orbit : {'analytic', 'oem'}
        Orbit source forwarded to the generator. ``"analytic"`` is required when the
        data grid (e.g. a training file starting at t=0) lies outside the OEM domain.
        Ignored by the FastBHB generator.
    waveform_span_days : float or None
        If given, only generate the waveform over this many days centred on the
        signal's coalescence time and zero-fill the rest of ``detector_times``. This
        keeps memory/CPU bounded for multi-year grids where an MBHB chirp occupies a
        short window; if ``None``, generate over the entire ``detector_times`` grid.
    waveform : str
        Approximant name: 'FastBHB' (default).
    """
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

    # Use the unified waveform generation interface
    waveform_data = generate_waveform(
        signal,
        approximant=waveform,
        detector_times=gen_times,
        tdi_generation=tdi_generation,
        orbit=orbit,
    )

    # Map the generated window back onto the full data grid (zero-filled elsewhere).
    if waveform_span_days is None:
        return {channel: np.asarray(arr, dtype=float) for channel, arr in waveform_data.items()}

    out = {}
    for channel, arr in waveform_data.items():
        full = np.zeros(len(detector_times), dtype=float)
        lo_idx = int(np.searchsorted(detector_times, gen_times[0], side="left"))
        hi_idx = int(np.searchsorted(detector_times, gen_times[-1], side="right"))
        n_copy = min(hi_idx - lo_idx, len(arr))
        if n_copy > 0:
            full[lo_idx : lo_idx + n_copy] = arr[:n_copy]
        out[channel] = full
    return out


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    add_hwilas_common_options(parser)
    add_truth_arguments(parser)

    # Override --truth-file to be required (the plotting tool always needs it).
    for action in parser._actions:
        if "--truth-file" in action.option_strings:
            action.required = True
            break

    parser.add_argument(
        "--input-file",
        required=True,
        help="Path to the Sangria HDF5 file containing the noisy TDI data (obs/tdi).",
    )
    parser.add_argument(
        "--output-dir",
        default="plots/sangria_waveforms",
        help="Directory in which to write the plots (created if missing).",
    )
    parser.add_argument(
        "--channels",
        nargs="+",
        choices=list(VALID_CHANNELS),
        default=["LISA_A"],
        help=(
            "One or more TDI channels to plot, each as its own subplot row "
            f"(default: LISA_A). Valid: {', '.join(VALID_CHANNELS)}."
        ),
    )
    parser.add_argument(
        "--signals",
        type=int,
        nargs="*",
        default=None,
        help="Catalog indices of signals to generate waveforms for. Defaults to all.",
    )
    parser.add_argument(
        "--window-before-days",
        type=float,
        default=0.2,
        help="Window start offset before coalescence in days (default 0.2).",
    )
    parser.add_argument(
        "--window-after-days",
        type=float,
        default=0.02,
        help="Window end offset after coalescence in days (default 0.02).",
    )
    parser.add_argument(
        "--full-window-days",
        type=float,
        default=None,
        help=(
            "If set, restrict the full-dataset overview to this many days centred on "
            "the first requested signal's coalescence time. Defaults to the whole file."
        ),
    )
    parser.add_argument(
        "--no-waveform",
        action="store_true",
        help="Skip waveform regeneration and only plot noisy vs noiseless data.",
    )
    parser.add_argument(
        "--waveform-span-days",
        type=float,
        default=2.0,
        help=(
            "Days of waveform regenerated around each signal's coalescence (default 2). "
            "The rest of the data grid is zero-filled. Use a large value to generate "
            "over the whole year at the cost of much higher memory/CPU."
        ),
    )
    parser.add_argument(
        "--waveform",
        default="FastBHB",
        help=(
            "Waveform generator to use for regeneration (default: FastBHB)."
        ),
    )
    parser.add_argument(
        "--orbit",
        default="analytic",
        choices=["analytic", "oem"],
        help=(
            "Orbit model for waveform regeneration. 'analytic' (default) uses an "
            "unbounded analytic orbit so any data time grid is valid; 'oem' uses the "
            "ESA nominal orbits, which only cover a fixed ~3926 day domain."
        ),
    )
    parser.add_argument(
        "--tdi-generation",
        choices=["1.5", "2"],
        default="1.5",
        help="TDI generation forwarded to waveform generation (default 1.5).",
    )
    return parser


def _index_range(t, t_start, t_end):
    """Return the inclusive [start, end) sample indices of ``t`` within a time range."""
    start = int(np.searchsorted(t, t_start, side="left"))
    end = int(np.searchsorted(t, t_end, side="right"))
    return max(0, start), min(len(t), end)


def plot_full_dataset(times, noisy, noiseless, signals, channels, out_path, full_window_days=None):
    """Plot the full (or windowed) time series with coalescence markers.

    ``noisy`` and ``noiseless`` map channel name -> 1-D array; one subplot row is
    drawn per entry in ``channels``.
    """
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
    fig.suptitle("Sangria full dataset: noisy vs noiseless MBHB injection")
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
    """Plot a zoomed window around a single signal's coalescence time.

    ``noisy``, ``noiseless`` and ``generated`` map channel name -> 1-D array; one
    subplot row is drawn per entry in ``channels``.
    """
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

    t_win = times[lo:hi] - signal["tc"]  # time relative to coalescence (seconds)

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
        ax.plot(
            t_win / SECONDS_PER_DAY,
            noiseless[channel][lo:hi],
            label="Noiseless injection",
            color="tab:red",
            linewidth=1.4,
        )
        if generated is not None:
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


def main():
    parser = build_parser()
    args = parser.parse_args()
    init_logging(args.verbose)

    os.makedirs(args.output_dir, exist_ok=True)

    # Load the standardised truth file (catalog + noiseless AET timeseries).
    logger.info("Loading truth file from %s", args.truth_file)
    truth = load_truth_file(args.truth_file)
    signals = load_signals_from_truth(truth)
    if args.signals is not None:
        wanted = set(args.signals)
        signals = [s for s in signals if s["index"] in wanted]
    logger.info("Loaded %d MBHB signals from truth catalog.", len(signals))

    # Noiseless injection: use the AET timeseries from the truth file.
    if truth.timeseries is None:
        parser.error(
            "Truth file has no timeseries; cannot plot noiseless injection. "
            "Re-run hwilas_extract_truth with waveform extraction enabled."
        )
    noiseless = {ch: np.asarray(arr, dtype=float) for ch, arr in truth.timeseries.items()}

    # Read the data grid metadata from the noisy TDI dataset.
    with h5py.File(args.input_file, "r") as f:
        attrs = dict(f[NOISY_TDI_PATH].attrs)
    delta_t = float(attrs["dt"])
    epoch = float(attrs.get("t0", 0.0))

    logger.info(
        "Loading noisy TDI stream from %s (dt=%.3f s, t0=%.1f s)",
        args.input_file,
        delta_t,
        epoch,
    )
    noisy = load_noisy_tdi_channels(args.input_file)

    for channel in args.channels:
        if channel not in noisy:
            parser.error(f"Channel {channel} not found in TDI data.")

    n_samples = len(noisy[args.channels[0]])
    times = epoch + delta_t * np.arange(n_samples)

    # Full-dataset overview.
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

    # Per-signal windows.
    for sig in signals:
        logger.info("Generating waveform for signal #%s", sig["index"])
        generated = None
        if not args.no_waveform:
            try:
                generated = generate_signal_waveform(
                    sig,
                    times,
                    orbit=args.orbit,
                    waveform_span_days=args.waveform_span_days,
                    waveform=args.waveform,
                    tdi_generation=args.tdi_generation,
                )
            except Exception as exc:  # noqa: BLE001 - keep going on a single failure
                logger.warning(
                    "Waveform generation failed for signal #%s: %s", sig["index"], exc
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

    logger.info("Done. Plots written to %s", args.output_dir)


if __name__ == "__main__":
    main()


