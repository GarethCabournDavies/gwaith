"""Load, downsample, split, pad, and inpaint a specific chunk of Mojito L1 data."""

import argparse
import logging
import importlib
import math

import h5py
import numpy as np

from gwaith.types import TimeSeries
from hwilas import add_hwilas_common_options, init_logging
from hwilas.data_processing.tdi import linear_AET_conversion
from hwilas.data_processing.inpainting import gate_and_paint
from hwilas.psd.read import from_txt as psd_from_txt

try:
    from MojitoProcessor.process.sigprocess import process_pipeline
except ImportError:
    raise ImportError(
        "MojitoProcessor is required. Install with: "
        "pip install git+https://github.com/OllieBurke/MojitoProcessor.git"
    )

logger = logging.getLogger(__name__)


def _load_mojito_window(input_path, start_time, end_time):
    """Load a specific time window from a Mojito L1 file."""
    MojitoL1File = getattr(importlib.import_module("mojito"), "MojitoL1File")
    
    with MojitoL1File(input_path) as f:
        tdi_sampling = f.tdis.time_sampling
        ltt_sampling = f.ltts.time_sampling
        orbit_sampling = f.orbits.time_sampling
        
        epoch = float(tdi_sampling.epoch) if hasattr(tdi_sampling, "epoch") else 0.0
        
        tdi_start_idx = max(0, int(math.floor((start_time - epoch) * tdi_sampling.fs)))
        tdi_end_idx = min(tdi_sampling.size, int(math.ceil((end_time - epoch) * tdi_sampling.fs)))
        
        ltt_start_idx = max(0, int(math.floor((start_time - epoch) * ltt_sampling.fs)))
        ltt_end_idx = min(ltt_sampling.size, int(math.ceil((end_time - epoch) * ltt_sampling.fs)))
        
        data = {
            "tdis": {
                "X": f.tdis.x2[tdi_start_idx:tdi_end_idx],
                "Y": f.tdis.y2[tdi_start_idx:tdi_end_idx],
                "Z": f.tdis.z2[tdi_start_idx:tdi_end_idx],
                "A": f.tdis.a2[tdi_start_idx:tdi_end_idx],
                "E": f.tdis.e2[tdi_start_idx:tdi_end_idx],
                "T": f.tdis.t2[tdi_start_idx:tdi_end_idx],
            },
            "fs": tdi_sampling.fs,
            "dt": tdi_sampling.dt,
            "t_tdi": tdi_sampling.t(slice(tdi_start_idx, tdi_end_idx)),
            "ltts": {
                "12": f.ltts.ltt_12[ltt_start_idx:ltt_end_idx],
                "13": f.ltts.ltt_13[ltt_start_idx:ltt_end_idx],
                "21": f.ltts.ltt_21[ltt_start_idx:ltt_end_idx],
                "23": f.ltts.ltt_23[ltt_start_idx:ltt_end_idx],
                "31": f.ltts.ltt_31[ltt_start_idx:ltt_end_idx],
                "32": f.ltts.ltt_32[ltt_start_idx:ltt_end_idx],
            },
            "ltt_derivatives": {
                "12": f.ltts.ltt_derivative_12[ltt_start_idx:ltt_end_idx],
                "13": f.ltts.ltt_derivative_13[ltt_start_idx:ltt_end_idx],
                "21": f.ltts.ltt_derivative_21[ltt_start_idx:ltt_end_idx],
                "23": f.ltts.ltt_derivative_23[ltt_start_idx:ltt_end_idx],
                "31": f.ltts.ltt_derivative_31[ltt_start_idx:ltt_end_idx],
                "32": f.ltts.ltt_derivative_32[ltt_start_idx:ltt_end_idx],
            },
            "ltt_times": ltt_sampling.t(slice(ltt_start_idx, ltt_end_idx)),
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


def main():
    parser = argparse.ArgumentParser()
    add_hwilas_common_options(parser)
    parser.add_argument("--data-file", required=True)
    parser.add_argument("--output-file", required=True)
    parser.add_argument("--psd-file", required=True)
    parser.add_argument("--channels", nargs='+', default=['LISA_A', 'LISA_E'])
    parser.add_argument("--delta-t", type=float, required=True)
    parser.add_argument("--low-cutoff-frequency", type=float, default=1e-5)
    parser.add_argument("--analysis-start-time", type=float, required=True)
    parser.add_argument("--analysis-end-time", type=float, required=True)
    parser.add_argument("--padding-duration", type=float, default=30 * 86400)
    parser.add_argument("--filter-taper-duration", type=float, default=15 * 86400,
                        help="Extra duration (s) to load before and after the analysis window to avoid filter edge effects.")
    args = parser.parse_args()
    init_logging(args.verbose)

    # 1. Load an overlapping window to avoid filter edge effects
    target_fs = 1.0 / args.delta_t
    overlap = args.filter_taper_duration
    load_start = max(0, args.analysis_start_time - overlap)
    load_end = args.analysis_end_time + overlap

    logger.info("Loading Mojito slice from %.0fs to %.0fs (including %.0fs filter taper)", 
                load_start, load_end, overlap)
    raw_data, epoch = _load_mojito_window(args.data_file, load_start, load_end)

    # 2. Process the pipeline on this slice
    logger.info("Running MojitoProcessor pipeline")
    processed_segments = process_pipeline(
        raw_data,
        downsample_kwargs={"target_fs": target_fs, "kaiser_window": 31.0},
        filter_kwargs={"highpass_cutoff": args.low_cutoff_frequency, "lowpass_cutoff": 0.8 * target_fs, "order": 2},
        trim_kwargs={"fraction": 0.0},
        truncate_kwargs={"days": None},
        window_kwargs={"window": "tukey", "alpha": 0.0125},
    )

    if not processed_segments:
        raise ValueError("No segments returned by pipeline")
    
    seg_id = list(processed_segments.keys())[0]
    segment = processed_segments[seg_id]
    
    extracted = _extract_requested_channels(segment, args.channels)

    # 3. Trim off the filter taper and convert to TimeSeries
    # Note: process_pipeline returns data whose first sample corresponds to load_start
    logger.info("Trimming filter taper and preparing analysis window")
    analysis_duration = args.analysis_end_time - args.analysis_start_time
    
    # Calculate offset in downsampled points
    start_offset_s = max(0, args.analysis_start_time - load_start)
    start_idx = int(round(start_offset_s / args.delta_t))
    end_idx = start_idx + int(round(analysis_duration / args.delta_t))
    
    data = {}
    for ch in args.channels:
        trimmed = extracted[ch][start_idx:end_idx].copy()
        data[ch] = TimeSeries(trimmed, epoch=args.analysis_start_time, delta_t=args.delta_t)

    # 4. Pad and inpaint
    logger.info("Padding and inpainting")
    padded_tlen = analysis_duration + args.padding_duration
    padded_dlen = int(padded_tlen / args.delta_t)
    original_dlen = len(data[args.channels[0]])

    for ch in args.channels:
        data[ch].resize(padded_dlen)

    padded_flen = padded_dlen // 2 + 1
    padded_delta_f = 1. / padded_tlen
    inpainting_psd = psd_from_txt(
        args.psd_file,
        padded_flen,
        padded_delta_f,
        padded_delta_f,
        is_asd_file=False,
    )

    gap_starts = [original_dlen]
    gap_ends = [padded_dlen]

    for ch in args.channels:
        logger.info(f"Applying inpainting to padded + gated sections in {ch}")
        data[ch] = gate_and_paint(
            data[ch], gap_starts, gap_ends, inpainting_psd
        )

    # 5. Save to file
    logger.info("Saving padded chunk to %s", args.output_file)
    with h5py.File(args.output_file, 'w') as f:
        f.attrs["epoch"] = float(args.analysis_start_time)
        f.attrs["delta_t"] = float(args.delta_t)
        f.attrs["padding_duration"] = args.padding_duration
        for ch in args.channels:
            f.create_dataset(ch, data=data[ch].data)

    logger.info("Done.")


if __name__ == "__main__":
    main()
