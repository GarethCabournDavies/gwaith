"""Integration with MojitoProcessor for Mojito L1 data processing.

This module provides a high-level interface to MojitoProcessor's pipeline,
which handles filtering, downsampling, windowing, and TDI response correction.
"""
import logging
from pathlib import Path
from typing import Dict, Optional

import h5py
import numpy as np
from gwaith.types import TimeSeries

try:
    from MojitoProcessor.io.read import load_file
    from MojitoProcessor.process.sigprocess import process_pipeline
except ImportError:
    raise ImportError(
        "MojitoProcessor is required. Install with: "
        "pip install git+https://github.com/OllieBurke/MojitoProcessor.git"
    )

logger = logging.getLogger(__name__)


def process_mojito_with_mojitoprocessor(
    input_path: str,
    downsample_target_fs: float,
    truncate_days: float = 7.0,
    highpass_cutoff: float = 5e-6,
    lowpass_cutoff: Optional[float] = None,
    filter_order: int = 2,
    kaiser_beta: float = 31.0,
    trim_fraction: float = 0.02,
    window_type: str = "tukey",
    window_alpha: float = 0.0125,
    channels: Optional[list] = None,
    load_days: Optional[float] = None,
) -> Dict[str, Dict]:
    """Process Mojito L1 data using MojitoProcessor pipeline.

    The MojitoProcessor pipeline applies the following steps in order:
    1. Bandpass filter (configurable high/low pass cutoffs and order)
    2. Downsample to target sampling rate
    3. Trim edge artifacts (configurable fraction from each end)
    4. Truncate into segments of specified duration
    5. Apply window (Tukey, Hann, Hamming, Blackman, Planck, or None)

    Data is normalized by laser frequency internally within process_pipeline,
    output is in fractional frequency units (dimensionless).

    Args:
        input_path: Path to the input Mojito HDF5 file.
        downsample_target_fs: Target sampling rate in Hz.
        truncate_days: Segment length in days (splits dataset into chunks).
        highpass_cutoff: High-pass filter cutoff in Hz (always applied).
            Default: 5e-6 Hz.
        lowpass_cutoff: Low-pass filter cutoff in Hz. If None, only high-pass
            filtering is applied. Default: None (high-pass only).
        filter_order: Butterworth filter order. Default: 2.
        kaiser_beta: Kaiser window beta parameter for anti-aliasing filter.
            Higher = more aggressive anti-aliasing. Default: 31.0.
        trim_fraction: Fraction of post-downsample duration trimmed from each end.
            Default: 0.02 (2%).
        window_type: Window to apply: 'tukey', 'hann', 'hamming', 'blackman',
            'planck', or None for no windowing. Default: 'tukey'.
        window_alpha: Taper fraction for Tukey window. Default: 0.0125.
        channels: TDI channels to load (e.g., ['X', 'Y', 'Z'] or ['A', 'E', 'T']).
            If None, loads all available channels.
        load_days: Number of days to load from the file (lazy slicing).
            If None, loads the entire file.

    Returns:
        dict: Dictionary with keys:
            - 'segments': dict of segment_id -> SignalProcessor objects
            - 'raw_data': dict with 'orbits', 'noise_estimates', 'metadata'
            - 'laser_frequency': float (from metadata)
    """
    logger.info("Loading Mojito file: %s", input_path)

    # Load raw data (lazy slicing - only reads what is needed)
    raw_data = load_file(input_path, load_days=load_days)
    laser_frequency = raw_data.get("laser_frequency")

    if laser_frequency is None:
        if "metadata" in raw_data and "laser_frequency" in raw_data["metadata"]:
            laser_frequency = raw_data["metadata"]["laser_frequency"]
        else:
            # Fallback to Mojito default
            laser_frequency = 2.816e14
            logger.warning(
                "No laser_frequency found in data; using default %.3e Hz",
                laser_frequency,
            )
    else:
        logger.info("Using laser_frequency: %.3e Hz", laser_frequency)

    # Set default lowpass cutoff if not specified
    if lowpass_cutoff is None:
        lowpass_cutoff = 0.8 * downsample_target_fs
        logger.info("Auto-setting lowpass cutoff to 80%% of target fs: %.3e Hz",
                   lowpass_cutoff)

    # Prepare filter kwargs
    filter_kwargs = {
        "highpass_cutoff": highpass_cutoff,
        "lowpass_cutoff": lowpass_cutoff,
        "order": filter_order,
    }

    # Prepare downsample kwargs
    downsample_kwargs = {
        "target_fs": downsample_target_fs,
        "kaiser_window": kaiser_beta,
    }

    # Prepare trim kwargs
    trim_kwargs = {
        "fraction": trim_fraction,
    }

    # Prepare truncate kwargs
    truncate_kwargs = {
        "days": truncate_days,
    }

    # Prepare window kwargs
    window_kwargs = {
        "window": window_type,
        "alpha": window_alpha,
    }

    logger.info("Running MojitoProcessor pipeline:")
    logger.info("  Filter: HP=%.3e Hz, LP=%.3e Hz, order=%d",
               highpass_cutoff, lowpass_cutoff, filter_order)
    logger.info("  Downsample: target_fs=%.3e Hz, kaiser_beta=%.1f",
               downsample_target_fs, kaiser_beta)
    logger.info("  Trim: %.1f%% from each end", trim_fraction * 100)
    logger.info("  Truncate: %.1f-day segments", truncate_days)
    logger.info("  Window: %s (alpha=%.4f)", window_type, window_alpha)

    # Run the processing pipeline
    processed_segments = process_pipeline(
        raw_data,
        downsample_kwargs=downsample_kwargs,
        filter_kwargs=filter_kwargs,
        trim_kwargs=trim_kwargs,
        truncate_kwargs=truncate_kwargs,
        window_kwargs=window_kwargs,
    )

    logger.info("Pipeline complete. Generated %d segments.", len(processed_segments))

    return {
        "segments": processed_segments,
        "raw_data": raw_data,
        "laser_frequency": laser_frequency,
    }


def extract_tdi_timeseries(
    processed_segment,
    channel: str = "A",
    delta_t: Optional[float] = None,
    epoch: float = 0.0,
) -> TimeSeries:
    """Convert a MojitoProcessor SignalProcessor segment to gwaith TimeSeries.

    Args:
        processed_segment: SignalProcessor object from MojitoProcessor.
        channel: TDI channel ('A', 'E', 'T' for AET or 'X', 'Y', 'Z' for XYZ).
        delta_t: Sample time. If None, extracted from the segment.
        epoch: GPS epoch of the segment. Default: 0.0.

    Returns:
        gwaith TimeSeries object.
    """
    if delta_t is None:
        delta_t = processed_segment.dt

    # MojitoProcessor stores data in the 'data' dict
    data_array = processed_segment.data[channel]

    return TimeSeries(
        data_array,
        delta_t=delta_t,
        epoch=epoch,
    )


def save_segments_to_hdf5(
    output_path: str,
    result_dict: Dict,
    segment_ids: Optional[list] = None,
    channels: Optional[list] = None,
):
    """Save MojitoProcessor segments to HDF5 file in gwaith format.

    Args:
        output_path: Output HDF5 file path.
        result_dict: Dictionary returned from process_mojito_with_mojitoprocessor.
        segment_ids: List of segment IDs to save. If None, saves all segments.
        channels: List of TDI channels to save (e.g., ['A', 'E', 'T']).
            If None, saves all available channels.
    """
    segments = result_dict["segments"]
    laser_frequency = result_dict["laser_frequency"]

    if segment_ids is None:
        segment_ids = sorted(segments.keys())

    logger.info("Saving %d segments to %s", len(segment_ids), output_path)

    with h5py.File(output_path, "w") as f:
        # Store metadata
        f.attrs["laser_frequency"] = laser_frequency
        f.attrs["source"] = "MojitoProcessor"
        f.attrs["coordinate_system"] = "TDI"

        for seg_id in segment_ids:
            if seg_id not in segments:
                logger.warning("Segment %s not found, skipping", seg_id)
                continue

            segment = segments[seg_id]
            delta_t = segment.dt
            epoch = 0.0  # Default; adjust if segment has epoch info

            # Determine channels to save
            available_channels = list(segment.data.keys())
            if channels is None:
                channels_to_save = available_channels
            else:
                channels_to_save = [ch for ch in channels if ch in available_channels]

            if not channels_to_save:
                logger.warning(
                    "No matching channels found for segment %s. Available: %s",
                    seg_id, available_channels
                )
                continue

            # Create segment group
            seg_group = f.create_group(seg_id)
            seg_group.attrs["delta_t"] = delta_t
            seg_group.attrs["epoch"] = epoch
            seg_group.attrs["channel_count"] = len(channels_to_save)

            # Store channel data
            for ch in channels_to_save:
                data = segment.data[ch]
                seg_group.create_dataset(ch, data=data, compression="gzip")

            logger.debug("Saved segment %s with channels %s", seg_id, channels_to_save)

    logger.info("Successfully saved segments to %s", output_path)


if __name__ == "__main__":
    # Example usage
    logging.basicConfig(level=logging.INFO)
    result = process_mojito_with_mojitoprocessor(
        input_path="mojito_data.h5",
        downsample_target_fs=0.2,
        truncate_days=7.0,
        channels=["A", "E", "T"],
    )
    print(f"Processed {len(result['segments'])} segments")
