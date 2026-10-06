"""
Load a Mojito file, downsample a specific window (or the full file), and save AET channels to HDF5.
"""

import argparse
import importlib
import logging
import math

import h5py
import numpy as np

from gwaith.types import TimeSeries
from hwilas import add_hwilas_common_options, init_logging
from hwilas.data_processing.tdi import linear_AET_conversion

try:
    from MojitoProcessor.process.sigprocess import process_pipeline
    from MojitoProcessor.io.read import load_file
except ImportError:
    process_pipeline = None

logger = logging.getLogger(__name__)


def _load_mojito_window(path, start_time, end_time, requested_channels):
    """Load a specific time window from a Mojito file, only loading requested channels."""
    mojito = importlib.import_module("mojito")
    with mojito.MojitoL1File(path) as f:
        tdi_sampling = f.tdis.time_sampling
        orbit_sampling = f.orbits.time_sampling

        # Convert times to indices
        epoch = float(tdi_sampling.epoch) if hasattr(tdi_sampling, "epoch") else 0.0
        tdi_start_idx = int((start_time - epoch) * tdi_sampling.fs) if start_time > epoch else 0
        tdi_end_idx = int((end_time - epoch) * tdi_sampling.fs)
        
        tdi_start_idx = max(0, min(tdi_start_idx, tdi_sampling.size))
        tdi_end_idx = max(0, min(tdi_end_idx, tdi_sampling.size))
        
        # Only load the channels we actually need to save memory
        tdis = {}
        needed_shorts = [ch.replace("LISA_", "") for ch in requested_channels]
        
        if "X" in needed_shorts: tdis["X"] = f.tdis.x2[tdi_start_idx:tdi_end_idx]
        if "Y" in needed_shorts: tdis["Y"] = f.tdis.y2[tdi_start_idx:tdi_end_idx]
        if "Z" in needed_shorts: tdis["Z"] = f.tdis.z2[tdi_start_idx:tdi_end_idx]
        if "A" in needed_shorts: tdis["A"] = f.tdis.a2[tdi_start_idx:tdi_end_idx]
        if "E" in needed_shorts: tdis["E"] = f.tdis.e2[tdi_start_idx:tdi_end_idx]
        if "T" in needed_shorts: tdis["T"] = f.tdis.t2[tdi_start_idx:tdi_end_idx]

        data = {
            "tdis": tdis,
            "fs": tdi_sampling.fs,
            "dt": tdi_sampling.dt,
            "t_tdi": tdi_sampling.t(slice(tdi_start_idx, tdi_end_idx)),
            "orbits": f.orbits.positions[:],
            "velocities": f.orbits.velocities[:],
            "orbit_times": orbit_sampling.t(slice(None)),
            "noise_estimates": {
                "xyz": f.noise_estimates.xyz[:],
                "aet": f.noise_estimates.aet[:],
            },
            "metadata": {
                "laser_frequency": f.laser_frequency,
                "pipeline_names": f.pipeline_names,
            },
        }
    return data, epoch


def _extract_requested_channels(segment, requested_channels):
    """Map/derive requested LISA_* channels from a MojitoProcessor segment."""
    seg_data = segment.data
    available = set(seg_data.keys())
    direct_map = {"LISA_A": "A", "LISA_E": "E", "LISA_T": "T", "LISA_X": "X", "LISA_Y": "Y", "LISA_Z": "Z"}
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

    needs_aet = any(ch in missing for ch in ("LISA_A", "LISA_E", "LISA_T"))
    has_xyz = all(key in available for key in ("X", "Y", "Z"))
    has_lisa_xyz = all(key in available for key in ("LISA_X", "LISA_Y", "LISA_Z"))
    if needs_aet and (has_xyz or has_lisa_xyz):
        X = np.asarray(seg_data["X"] if has_xyz else seg_data["LISA_X"], dtype=float)
        Y = np.asarray(seg_data["Y"] if has_xyz else seg_data["LISA_Y"], dtype=float)
        Z = np.asarray(seg_data["Z"] if has_xyz else seg_data["LISA_Z"], dtype=float)
        derived = linear_AET_conversion(X, Y, Z)
        for ch in ("LISA_A", "LISA_E", "LISA_T"):
            if ch in missing:
                output[ch] = derived[ch]
                missing.remove(ch)
    if missing:
        raise ValueError(f"Requested channels missing: {missing}")
    return output


def _read_input_epoch_and_duration_days(input_path):
    try:
        MojitoL1File = getattr(importlib.import_module("mojito"), "MojitoL1File")
        with MojitoL1File(input_path) as f:
            sampling = f.tdis.time_sampling
            epoch = float(sampling.epoch) if hasattr(sampling, "epoch") else 0.0
            delta_t = float(sampling.dt)
            n_samples = int(sampling.size)
    except Exception:
        with h5py.File(input_path, "r") as f:
            epoch = float(f.attrs.get("epoch", 0.0))
            delta_t = float(f.attrs["delta_t"])
            first = f["tdis"]
            if isinstance(first, h5py.Group):
                keys = list(first.keys())
                n_samples = int(first[keys[0]].shape[0])
            else:
                n_samples = int(first.shape[0])
    return epoch, (n_samples * delta_t) / 86400.0


def main():
    parser = argparse.ArgumentParser(
        description="Load a Mojito file, process with MojitoProcessor, and save channels to HDF5."
    )
    add_hwilas_common_options(parser)
    parser.add_argument("--data-file", required=True)
    parser.add_argument("--output-file", required=True)
    parser.add_argument("--channels", nargs='+', default=['LISA_A', 'LISA_E', 'LISA_T'])
    parser.add_argument("--delta-t", type=float, required=True)
    parser.add_argument("--low-cutoff-frequency", type=float, default=1e-5)
    parser.add_argument("--edge-taper-duration", type=float, default=86400.0)
    parser.add_argument("--truncate-days", type=float, default=None)
    parser.set_defaults(divide_by_laser_frequency=True)
    parser.add_argument("--divide-by-laser-frequency", dest="divide_by_laser_frequency", action="store_true")
    parser.add_argument("--no-divide-by-laser-frequency", dest="divide_by_laser_frequency", action="store_false")

    # New arguments for windowed processing
    parser.add_argument("--analysis-start-time", type=float, default=None,
                        help="Optional start time for processing a specific chunk.")
    parser.add_argument("--analysis-end-time", type=float, default=None,
                        help="Optional end time for processing a specific chunk.")
    parser.add_argument("--filter-taper-duration", type=float, default=15 * 86400,
                        help="Extra duration (s) to load before and after the analysis window to avoid filter edge effects.")

    args = parser.parse_args()
    init_logging(args.verbose)
    
    if process_pipeline is None:
        raise ImportError("MojitoProcessor is required. Install with: pip install git+https://github.com/OllieBurke/MojitoProcessor.git")

    target_fs = 1.0 / float(args.delta_t)
    logger.info("Processing Mojito data with MojitoProcessor from %s", args.data_file)
    logger.info("Target sampling: delta_t=%s s (fs=%s Hz)", args.delta_t, target_fs)

    input_epoch, duration_days = _read_input_epoch_and_duration_days(args.data_file)
    
    # Determine the window to load
    is_windowed = (args.analysis_start_time is not None and args.analysis_end_time is not None)
    
    if is_windowed:
        overlap = args.filter_taper_duration
        load_start = max(input_epoch, args.analysis_start_time - overlap)
        load_end = args.analysis_end_time + overlap
        logger.info("Windowed processing requested: loading Mojito slice from %.0fs to %.0fs", 
                    load_start, load_end)
        raw_data, _ = _load_mojito_window(args.data_file, load_start, load_end, args.channels)
    else:
        logger.info("Loading full file natively...")
        raw_data = load_file(args.data_file, load_days=None)
        load_start = input_epoch

    # Memory optimization: drop everything we don't need for the requested channels
    # The LTTs and unused TDIs take up gigabytes of memory and are completely 
    # unnecessary for basic AET downsampling.
    if "tdis" in raw_data:
        needed_tdis = [ch.replace("LISA_", "") for ch in args.channels]
        for key in list(raw_data["tdis"].keys()):
            if key not in needed_tdis:
                del raw_data["tdis"][key]
            
    raw_data.pop("ltts", None)
    raw_data.pop("ltt_derivatives", None)

    laser_frequency = raw_data.get("metadata", {}).get("laser_frequency", 2.816e14)

    logger.info("Running MojitoProcessor pipeline")
    processed_segments = process_pipeline(
        raw_data,
        channels=needed_tdis,
        downsample_kwargs={"target_fs": target_fs, "kaiser_window": 31.0},
        filter_kwargs={"highpass_cutoff": args.low_cutoff_frequency, "lowpass_cutoff": 0.8 * target_fs, "order": 2},
        trim_kwargs={"fraction": 0.0},
        truncate_kwargs={"days": args.truncate_days},
        window_kwargs={"window": "tukey", "alpha": 0.0125},
    )

    if not processed_segments:
        raise ValueError("No segments returned by pipeline")
    
    if args.truncate_days is not None and len(processed_segments) > 1:
        # Merge if truncate days was passed
        from hwilas.data_processing.process_mojito_data import _merge_segments
        data, delta_t, segment_names = _merge_segments(processed_segments, args.channels)
        final_epoch = input_epoch
    else:
        # Just use the single processed segment
        seg_id = list(processed_segments.keys())[0]
        segment = processed_segments[seg_id]
        extracted = _extract_requested_channels(segment, args.channels)
        
        if is_windowed:
            logger.info("Trimming filter taper and preparing analysis window")
            analysis_duration = args.analysis_end_time - args.analysis_start_time
            start_offset_s = max(0, args.analysis_start_time - load_start)
            start_idx = int(round(start_offset_s / args.delta_t))
            end_idx = start_idx + int(round(analysis_duration / args.delta_t))
            
            data = {}
            for ch in args.channels:
                data[ch] = extracted[ch][start_idx:end_idx].copy()
            final_epoch = args.analysis_start_time
        else:
            data = extracted
            final_epoch = input_epoch
            
    delta_t = args.delta_t

    logger.info("Saving loaded data to %s", args.output_file)

    with h5py.File(args.output_file, 'w') as f:
        f.attrs["epoch"] = final_epoch
        f.attrs["delta_t"] = delta_t
        f.attrs["divide_by_laser_frequency"] = True
        f.attrs["laser_frequency"] = float(laser_frequency)
        f.attrs["source_segment"] = "all"
        f.attrs["source_segment_count"] = 1
        f.attrs["source"] = "MojitoProcessor"
        f.attrs["processing_trim_fraction"] = 0.0
        f.attrs["processing_window"] = "none"
        for channel in args.channels:
            f.create_dataset(channel, data=np.asarray(data[channel], dtype=float))

    logger.info("Done. Saved %d channels with delta_t=%.4f, epoch=%.2f",
                len(args.channels), delta_t, final_epoch)


if __name__ == "__main__":
    main()
