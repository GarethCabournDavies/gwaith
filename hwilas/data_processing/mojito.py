"""Mojito file processing helpers for GWAITH ingestion."""
import logging

import numpy as np
from gwaith.types import TimeSeries
from mojito import MojitoL1File
from scipy.signal import filtfilt, firls
from tqdm import tqdm

logger = logging.getLogger(__name__)


def lowpass_filter_scipy(data, cutoff_freq, filter_width, sample_rate, filter_order=129):
    nyquist_rate = sample_rate / 2.0
    desired = (1, 1, 0, 0)
    bands = (0,cutoff_freq - 0.5 * filter_width, cutoff_freq + 0.5 * filter_width, nyquist_rate)
    filter_coefs = firls(filter_order, bands, desired, fs=nyquist_rate*2)

    filtered_data = filtfilt(
        filter_coefs,
        [1],
        data
    )
    return filtered_data

CHUNK_SIZE = 2 ** 22

_CHANNEL_MAP = {
    'LISA_A': "a2",
    'LISA_E': "e2",
    'LISA_T': "t2",
    'LISA_X': "x2",
    'LISA_Y': "y2",
    'LISA_Z': "z2",
}

def load_downsample_mojito(
    input_path,
    downsample_delta_t,
    channels=None
):
    """Process a Mojito file using parsed ingest CLI arguments.

    Args:
        parser: Argument parser used to build the ingest command.
        args: Parsed CLI arguments from ingest_data.

    Returns:
        dict: Mapping containing 'tdis' and 'orbits' output payloads.
    """
    if channels is None:
        channels = ['LISA_A', 'LISA_E', 'LISA_T']

    logger.info(
        'Processing Mojito file %s',
        input_path,
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
                "Downsampled delta_t mst be an integer multiple of the original delta_t"
            )
        decimation = int(downsample_delta_t / delta_t)
        logger.debug("decimation: %s", decimation)
        post_decimation_size = (n_samples + decimation - 1) // decimation
        logger.debug("post_decimation_size: %s", post_decimation_size)

        output_data = {
            channel: np.zeros(post_decimation_size, dtype=np.float64)
            for channel in channels
        }

        for chunk_i in tqdm(range(0, n_samples, CHUNK_SIZE)):
            logger.debug("Processing chunk %d", chunk_i)
            chunk_start = chunk_i
            chunk_end = chunk_i + CHUNK_SIZE
            chunk_end = min(chunk_end, n_samples)
            logger.debug("Chunk start: %d, end: %d", chunk_start, chunk_end)

            output_chunk_start = chunk_start // decimation
            output_chunk_end = chunk_end // decimation
            logger.debug("Output chunk start: %d, end: %d", output_chunk_start, output_chunk_end)
            dlen = max(1, output_chunk_end - output_chunk_start)
            tlen = dlen * downsample_delta_t
            delta_f = 1 / tlen
            logger.debug("delta_f: %s", delta_f)
            new_nyquist = 1 / downsample_delta_t / 2
            logger.debug("new nyquist: %s", new_nyquist)

            data = {
                channel: getattr(f.tdis, _CHANNEL_MAP[channel])[chunk_start:chunk_end]
                for channel in channels
            }
            for channel in channels:
                # We need to lowpass filter the data so there is nothing above the
                # new nyquist frequency before we can downsample it using decimation
                # logger.debug("Lowpass filtering %s", channel)
                filtered = lowpass_filter_scipy(
                    data[channel],
                    cutoff_freq=new_nyquist,
                    filter_width=2 * delta_f,
                    sample_rate=1. / delta_t
                )

                # Decimate while storing the samples in the chunk data dictionary.
                # logger.debug("Decimating %s", channel)
                decimated = filtered[::decimation]
                write_end = output_chunk_start + len(decimated)
                if write_end > post_decimation_size:
                    # The decimated data would go beyond the end of the output data
                    # Truncate the decimated data
                    decimated = decimated[:post_decimation_size - output_chunk_start]
                    # Write up to the end of the data
                    write_end = post_decimation_size
                output_data[channel][output_chunk_start:write_end] = decimated

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