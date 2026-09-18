"""
Load a Mojito file, downsample, and save AET channels to HDF5.
"""

import argparse
import importlib
import logging

import h5py
import numpy as np

from hwilas import add_hwilas_common_options, init_logging
from hwilas.data_processing.mojito_processor import process_mojito_with_mojitoprocessor
from hwilas.data_processing.tdi import linear_AET_conversion

logger = logging.getLogger(__name__)


def _sorted_segment_keys(segments):
    """Return segment keys in deterministic order."""
    if not segments:
        raise ValueError("MojitoProcessor returned no segments.")
    return sorted(
        segments.keys(),
        key=lambda k: int(str(k).replace("segment", "")) if str(k).startswith("segment") else str(k),
    )


def _extract_requested_channels(segment, requested_channels):
    """Map/derive requested LISA_* channels from a MojitoProcessor segment."""
    seg_data = segment.data
    available = set(seg_data.keys())

    # Accept either LISA_* or bare A/E/T/X/Y/Z labels in source segment data.
    direct_map = {
        "LISA_A": "A",
        "LISA_E": "E",
        "LISA_T": "T",
        "LISA_X": "X",
        "LISA_Y": "Y",
        "LISA_Z": "Z",
    }

    output = {}
    missing = []
    for channel in requested_channels:
        if channel in seg_data:
            output[channel] = np.asarray(seg_data[channel], dtype=float)
            continue
        short = direct_map.get(channel)
        if short in available:
            output[channel] = np.asarray(seg_data[short], dtype=float)
            continue
        missing.append(channel)

    if not missing:
        return output

    # Derive A/E/T from X/Y/Z if needed.
    needs_aet = any(ch in missing for ch in ("LISA_A", "LISA_E", "LISA_T"))
    has_xyz = all(key in available for key in ("X", "Y", "Z"))
    has_lisa_xyz = all(key in available for key in ("LISA_X", "LISA_Y", "LISA_Z"))
    if needs_aet and (has_xyz or has_lisa_xyz):
        if has_xyz:
            X = np.asarray(seg_data["X"], dtype=float)
            Y = np.asarray(seg_data["Y"], dtype=float)
            Z = np.asarray(seg_data["Z"], dtype=float)
        else:
            X = np.asarray(seg_data["LISA_X"], dtype=float)
            Y = np.asarray(seg_data["LISA_Y"], dtype=float)
            Z = np.asarray(seg_data["LISA_Z"], dtype=float)
        derived = linear_AET_conversion(X, Y, Z)
        for ch in ("LISA_A", "LISA_E", "LISA_T"):
            if ch in missing:
                output[ch] = derived[ch]
                missing.remove(ch)

    if missing:
        raise ValueError(
            "Requested channels are unavailable from MojitoProcessor output: "
            f"{missing}. Available segment fields: {sorted(available)}"
        )

    return output


def _merge_segments(segments, requested_channels):
    """Extract requested channels from each segment and concatenate in order."""
    keys = _sorted_segment_keys(segments)
    merged = {ch: [] for ch in requested_channels}
    delta_t = None
    for key in keys:
        segment = segments[key]
        if delta_t is None:
            delta_t = float(segment.dt)
        elif not np.isclose(float(segment.dt), delta_t):
            raise ValueError(
                f"Inconsistent segment delta_t: {segment.dt} vs {delta_t}"
            )
        extracted = _extract_requested_channels(segment, requested_channels)
        for ch in requested_channels:
            merged[ch].append(np.asarray(extracted[ch], dtype=float))

    concatenated = {
        ch: np.concatenate(parts) if parts else np.array([], dtype=float)
        for ch, parts in merged.items()
    }
    return concatenated, delta_t, keys


def _read_input_epoch_and_duration_days(input_path):
    """Read native epoch and total duration (days) from Mojito L1 sampling."""
    try:
        MojitoL1File = getattr(importlib.import_module("mojito"), "MojitoL1File")

        with MojitoL1File(input_path) as f:
            sampling = f.tdis.time_sampling
            epoch = float(sampling.epoch) if hasattr(sampling, "epoch") else 0.0
            delta_t = float(sampling.dt)
            n_samples = int(sampling.size)
    except Exception:
        # Fallback for environments where the mojito package is unavailable.
        with h5py.File(input_path, "r") as f:
            epoch = float(f.attrs.get("epoch", 0.0))
            delta_t = float(f.attrs["delta_t"])
            first = f["tdis"]
            if isinstance(first, h5py.Group):
                if "x2" in first:
                    n_samples = int(first["x2"].shape[0])
                elif "a2" in first:
                    n_samples = int(first["a2"].shape[0])
                else:
                    keys = list(first.keys())
                    if not keys:
                        raise ValueError("Could not infer sample count from input file.")
                    n_samples = int(first[keys[0]].shape[0])
            else:
                n_samples = int(first.shape[0])
    duration_days = (n_samples * delta_t) / 86400.0
    return epoch, duration_days


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Load a Mojito file, process with MojitoProcessor, and save "
            "LISA channels to HDF5."
        )
    )
    add_hwilas_common_options(parser)
    parser.add_argument(
        "--data-file",
        required=True,
        help="Path to the input Mojito HDF5 file"
    )
    parser.add_argument(
        "--output-file",
        required=True,
        help="Path to the output HDF5 file"
    )
    parser.add_argument(
        "--channels",
        nargs='+',
        default=['LISA_A', 'LISA_E', 'LISA_T'],
        help="Channels to include in the output. Default: LISA_A LISA_E LISA_T"
    )
    parser.add_argument(
        "--delta-t",
        type=float,
        required=True,
        help="Target sample time for downsampling (seconds)."
    )
    parser.add_argument(
        "--low-cutoff-frequency",
        type=float,
        default=1e-5,
        help="High-pass cutoff frequency in Hz. Default: 1e-5"
    )
    parser.add_argument(
        "--edge-taper-duration",
        type=float,
        default=86400.0,
        help=(
            "Deprecated compatibility option. Ignored by MojitoProcessor-backed "
            "implementation."
        )
    )
    parser.add_argument(
        "--truncate-days",
        type=float,
        default=None,
        help=(
            "Optional segment length for MojitoProcessor. By default this command "
            "uses full-duration processing to preserve timing alignment."
        )
    )
    parser.set_defaults(divide_by_laser_frequency=True)
    parser.add_argument(
        "--divide-by-laser-frequency",
        dest="divide_by_laser_frequency",
        action="store_true",
        help=(
            "Divide Mojito output channels by the input file's laser_frequency "
            "attribute (enabled by default)."
        ),
    )
    parser.add_argument(
        "--no-divide-by-laser-frequency",
        dest="divide_by_laser_frequency",
        action="store_false",
        help=(
            "Deprecated compatibility option. MojitoProcessor always applies "
            "laser-frequency normalization."
        ),
    )

    args = parser.parse_args()

    init_logging(args.verbose)

    if not args.divide_by_laser_frequency:
        logger.warning(
            "--no-divide-by-laser-frequency was requested, but MojitoProcessor "
            "always normalizes by laser frequency. Continuing with normalization enabled."
        )
    if args.edge_taper_duration != 86400.0:
        logger.warning(
            "--edge-taper-duration is ignored in the MojitoProcessor-backed "
            "implementation."
        )

    target_fs = 1.0 / float(args.delta_t)
    logger.info("Processing Mojito data with MojitoProcessor from %s", args.data_file)
    logger.info("Target sampling: delta_t=%s s (fs=%s Hz)", args.delta_t, target_fs)

    input_epoch, duration_days = _read_input_epoch_and_duration_days(args.data_file)
    truncate_days = args.truncate_days if args.truncate_days is not None else duration_days

    result = process_mojito_with_mojitoprocessor(
        input_path=args.data_file,
        downsample_target_fs=target_fs,
        truncate_days=truncate_days,
        highpass_cutoff=args.low_cutoff_frequency,
        lowpass_cutoff=None,
        trim_fraction=0.0,
        window_type=None,
    )

    data, delta_t, segment_names = _merge_segments(result["segments"], args.channels)
    epoch = input_epoch

    logger.info("Saving loaded data to %s", args.output_file)

    with h5py.File(args.output_file, 'w') as f:
        f.attrs["epoch"] = epoch
        f.attrs["delta_t"] = delta_t
        f.attrs["divide_by_laser_frequency"] = True
        laser_frequency = result.get("laser_frequency", None)
        if laser_frequency is not None:
            f.attrs["laser_frequency"] = float(laser_frequency)
        f.attrs["source_segment"] = "all"
        f.attrs["source_segment_count"] = len(segment_names)
        f.attrs["source"] = "MojitoProcessor"
        f.attrs["processing_trim_fraction"] = 0.0
        f.attrs["processing_window"] = "none"
        for channel in args.channels:
            f.create_dataset(channel, data=np.asarray(data[channel], dtype=float))

    logger.info(
        "Done. Saved %d channels from %d segment(s) with delta_t=%.4f, epoch=%.2f",
        len(args.channels),
        len(segment_names),
        delta_t,
        epoch,
    )


if __name__ == "__main__":
    main()