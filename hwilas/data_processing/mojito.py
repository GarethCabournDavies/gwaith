"""Mojito file processing helpers for GWAITH ingestion."""
import logging

import h5py
import numpy as np
from gwaith.types import TimeSeries
from mojito import MojitoL1File
from tqdm import tqdm

from hwilas.filter import (
    apply_bandpass_filter,
    highpass_filter_scipy,
    lowpass_filter_scipy,
)

logger = logging.getLogger(__name__)


CHUNK_SIZE = 2 ** 22

# The lowpass filter is applied per chunk, so its impulse response can leak
# into the first and last samples of each chunk. To avoid edge contamination
# in the decimated output, we load a 1 day overlap on each side of every chunk
# and only keep the decimated samples that fall within the central part of the
# chunk (i.e. after the midpoint of the leading overlap day).
OVERLAP_DURATION = 86400.0

_CHANNEL_MAP = {
    'LISA_A': "a2",
    'LISA_E': "e2",
    'LISA_T': "t2",
    'LISA_X': "x2",
    'LISA_Y': "y2",
    'LISA_Z': "z2",
}


def _apply_edge_taper(data, taper_samples):
    """Apply a raised-cosine taper to the first/last ``taper_samples`` samples.

    This suppresses large edge excursions that otherwise get handed to the
    split/pad inpainting stage in the first and last analysis chunks.
    """
    if taper_samples <= 0:
        return data

    n = len(data)
    if n == 0:
        return data

    taper_samples = int(min(taper_samples, n // 2))
    if taper_samples == 0:
        return data

    ramp = 0.5 * (1.0 - np.cos(np.linspace(0.0, np.pi, taper_samples)))
    data[:taper_samples] *= ramp
    data[-taper_samples:] *= ramp[::-1]
    return data


def _read_laser_frequency(input_path):
    """Read the Mojito file laser frequency root attribute as a float."""
    with h5py.File(input_path, "r") as h5f:
        if "laser_frequency" not in h5f.attrs:
            raise ValueError(
                "Requested scaling by laser frequency, but 'laser_frequency' "
                f"attribute is missing from {input_path}"
            )
        laser_frequency = float(h5f.attrs["laser_frequency"])
    if not np.isfinite(laser_frequency) or laser_frequency <= 0.0:
        raise ValueError(
            "Requested scaling by laser frequency, but found invalid value "
            f"{laser_frequency!r} in {input_path}"
        )
    return laser_frequency

def load_downsample_mojito(
    input_path,
    downsample_delta_t,
    channels=None,
    low_cutoff_freq=1e-5,
    edge_taper_duration=86400.0,
    divide_by_laser_frequency=False,
):
    """Load a Mojito file, bandpass filter it, and downsample to a coarser cadence.

    The data is processed in chunks of CHUNK_SIZE samples with an overlap of
    OVERLAP_DURATION (1 day) on each side so that the edge contamination from
    the per-chunk filtering does not leak into the decimated output. Only
    decimated samples after the midpoint of the leading overlap day are kept,
    which is the changeover point between chunks.

    A zero-phase IIR Butterworth lowpass is applied per chunk before downsampling
    to attenuate frequencies above the Nyquist frequency of the downsampled
    cadence (anti-aliasing). After decimation, a zero-phase IIR Butterworth
    highpass enforces the low-frequency cutoff ``low_cutoff_freq`` on the fully
    assembled data; this IIR stage is required because a per-chunk filter cannot
    resolve a cutoff as low as the default 1e-5 Hz (each chunk is far too short
    for its FFT to span that band).

    Args:
        input_path: Path to the Mojito L1 HDF5 file.
        downsample_delta_t: Target sample spacing in seconds. Must be an integer
            multiple of the file's native delta_t.
        channels: List of TDI channel names to load (e.g. 'LISA_A', 'LISA_E',
            'LISA_T'). Defaults to ['LISA_A', 'LISA_E', 'LISA_T'].
        low_cutoff_freq: Lower passband edge frequency in Hz. Frequencies below
            this are attenuated by the post-decimation Butterworth highpass.
            Default 1e-5 Hz.
        edge_taper_duration: Duration in seconds of the raised-cosine taper
            applied at both start and end of each downsampled output channel.
            Set to 0 to disable. Default 86400 (1 day).
        divide_by_laser_frequency: If True, divide output channels by the
            file-level ``laser_frequency`` attribute. This maps Mojito's TDI
            scale to a strain-like normalization compatible with Sangria.

    Returns:
        dict: Mapping of channel name to a TimeSeries at the downsampled cadence.
    """
    if channels is None:
        channels = ['LISA_A', 'LISA_E', 'LISA_T']

    logger.info(
        'Processing Mojito file %s',
        input_path,
    )

    laser_frequency = None
    if divide_by_laser_frequency:
        laser_frequency = _read_laser_frequency(input_path)
        logger.info(
            "Will divide output TDI channels by laser_frequency=%g",
            laser_frequency,
        )


    logger.debug("Loading Mojito File")
    with MojitoL1File(input_path) as f:
        logger.debug("File Open")
        tdi_sampling = f.tdis.time_sampling
        epoch = float(tdi_sampling.epoch) if hasattr(tdi_sampling, 'epoch') else 0.0
        logger.debug("epoch: %s", epoch)
        delta_t = float(tdi_sampling.dt)
        logger.debug("delta_t: %s", delta_t)
        n_samples = int(tdi_sampling.size)
        logger.debug("n_samples: %s", n_samples)

        if downsample_delta_t is None or downsample_delta_t == delta_t:
            raise ValueError(
                "Cannot load Mojito file without downsampling"
            )
        logger.info("Downsampling to %ss cadence", downsample_delta_t)
        if np.remainder(downsample_delta_t, delta_t) != 0:
            raise ValueError(
                "Downsampled delta_t must be an integer multiple of the original delta_t"
            )
        new_nyquist = 1 / downsample_delta_t / 2
        if low_cutoff_freq <= 0:
            raise ValueError(
                f"low_cutoff_freq must be positive, got {low_cutoff_freq}"
            )
        if low_cutoff_freq >= new_nyquist:
            raise ValueError(
                f"low_cutoff_freq ({low_cutoff_freq} Hz) must be less than the "
                f"downsampled Nyquist frequency ({new_nyquist} Hz)"
            )
        decimation = int(downsample_delta_t / delta_t)
        logger.debug("decimation: %s", decimation)
        post_decimation_size = (n_samples + decimation - 1) // decimation
        logger.debug("post_decimation_size: %s", post_decimation_size)

        output_data = {
            channel: np.zeros(post_decimation_size, dtype=np.float64)
            for channel in channels
        }

        # Number of native samples corresponding to the 1 day overlap. We load
        # this many extra samples on each side of every chunk so that the edge
        # contamination from the per-chunk lowpass filtering is discarded. The
        # changeover between chunks happens at the midpoint of the leading
        # overlap day, i.e. half a day before the start of the central part.
        overlap_samples = int(np.ceil(OVERLAP_DURATION / delta_t))
        logger.debug("Overlap samples: %d", overlap_samples)

        for chunk_i in tqdm(range(0, n_samples, CHUNK_SIZE)):
            logger.debug("Processing chunk %d", chunk_i)
            # Central part of this chunk (the region whose decimated output we keep)
            chunk_start = chunk_i
            chunk_end = min(chunk_i + CHUNK_SIZE, n_samples)
            logger.debug("Chunk start: %d, end: %d", chunk_start, chunk_end)

            # Extended region including the overlap on both sides
            read_start = max(0, chunk_start - overlap_samples)
            read_end = min(n_samples, chunk_end + overlap_samples)
            logger.debug("Read start: %d, end: %d", read_start, read_end)

            output_chunk_start = chunk_start // decimation
            output_chunk_end = chunk_end // decimation
            logger.debug("Output chunk start: %d, end: %d", output_chunk_start, output_chunk_end)
            dlen = max(1, output_chunk_end - output_chunk_start)
            new_nyquist = 1 / downsample_delta_t / 2
            logger.debug("new nyquist: %s", new_nyquist)

            data = {
                channel: getattr(f.tdis, _CHANNEL_MAP[channel])[read_start:read_end]
                for channel in channels
            }
            # Offset of the central chunk start within the extended read region.
            # Decimated samples before this point fall inside the leading overlap
            # day and are discarded; the changeover between chunks is at the
            # midpoint of that day, which (for a 1 day overlap) coincides with
            # the start of the central part when decimation >= 2.
            offset_in_chunk = chunk_start - read_start

            for channel in channels:
                # Anti-alias the data before downsampling: attenuate everything
                # above the downsampled Nyquist frequency so decimation does not
                # fold high frequencies into the band of interest. This is a
                # zero-phase IIR Butterworth lowpass (see
                # :func:`hwilas.filter.lowpass_filter_scipy`); it resolves the
                # anti-aliasing cutoff regardless of the chunk's FFT resolution.
                # The low-frequency cutoff cannot be enforced here because each
                # chunk is far too short to resolve sub-bin frequencies, so it
                # is re-enforced on the fully assembled dataset with an IIR
                # Butterworth highpass after decimation (see below).
                # logger.debug("Anti-aliasing %s", channel)
                filtered = lowpass_filter_scipy(
                    data[channel],
                    high_cutoff_freq=new_nyquist,
                    sample_rate=1.0 / delta_t,
                )

                # Decimate while storing the samples in the chunk data dictionary.
                # logger.debug("Decimating %s", channel)
                decimated = filtered[offset_in_chunk::decimation]
                write_end = output_chunk_start + len(decimated)
                if write_end > post_decimation_size:
                    # The decimated data would go beyond the end of the output data
                    # Truncate the decimated data
                    decimated = decimated[:post_decimation_size - output_chunk_start]
                    # Write up to the end of the data
                    write_end = post_decimation_size
                output_data[channel][output_chunk_start:write_end] = decimated

        # Enforce the low-frequency cutoff on the fully assembled downsampled
        # data using a zero-phase IIR Butterworth highpass. The per-chunk FIR
        # filter above cannot resolve frequencies below 1/chunk_duration, so a
        # very low cutoff (e.g. the default 1e-5 Hz) is not attenuated there;
        # filtering the complete dataset with an IIR prototype removes that
        # limitation because it does not depend on the data's FFT resolution.
        # Anti-aliasing was already handled by the per-chunk filter, so only
        # the low-frequency cutoff needs enforcement here.
        logger.info(
            "Applying post-decimation Butterworth highpass filter: low cutoff %g Hz",
            low_cutoff_freq,
        )
        for channel in channels:
            output_data[channel] = highpass_filter_scipy(
                output_data[channel],
                low_cutoff_freq=low_cutoff_freq,
                sample_rate=1.0 / downsample_delta_t,
            )

        taper_samples = int(np.round(float(edge_taper_duration) / downsample_delta_t))
        if taper_samples > 0:
            logger.info(
                "Applying edge taper: %g s (%d samples at each edge)",
                edge_taper_duration,
                taper_samples,
            )
            for channel in channels:
                _apply_edge_taper(output_data[channel], taper_samples)

        if laser_frequency is not None:
            for channel in channels:
                output_data[channel] = output_data[channel] / laser_frequency

        processed_delta_t = downsample_delta_t
        tdis_results = {
            channel: TimeSeries(
                output_data[channel],
                epoch=epoch,
                delta_t=processed_delta_t,
            )
            for channel in channels
        }

    logger.info(
        'Processed Mojito file %s with %d TDI channels and epoch=%s delta_t=%s',
        input_path,
        len(tdis_results),
        epoch,
        processed_delta_t,
    )

    return tdis_results