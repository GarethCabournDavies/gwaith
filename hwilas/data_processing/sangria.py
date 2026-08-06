"""Data processing for the Sangria dataset"""

import logging

import ldc.io.hdf5 as hdfio
import numpy as np
from tqdm import tqdm

from gwaith.types import TimeSeries

logger = logging.getLogger(__name__)

_CHANNEL_MAP = {
    'LISA_A': "A",
    'LISA_E': "E",
    'LISA_T': "T",
    'LISA_X': "X",
    'LISA_Y': "X",
    'LISA_Z': "Z",
}

CHUNK_SIZE = 2 ** 22


def linear_AET_conversion(data):
    X = data['X']
    Y = data['Y']
    Z = data['Z']

    return {
        'LISA_A': (Z - X)/np.sqrt(2),
        'LISA_E': (X - 2*Y + Z)/np.sqrt(6),
        'LISA_T': (X + Y + Z)/np.sqrt(3)
    }

def to_timeseries(data, channels, delta_t, epoch):
    return {
        channel: TimeSeries(
            data[channel],
            delta_t=delta_t,
            epoch=epoch
        )
        for channel in channels
    }


def load_downsample_sangria(
    input_path,
    downsample_delta_t,
    channels=None
):
    """
    Process a Sangria file
    """
    logger.info("Processing Sangria File %s", input_path)
    if channels is None:
        channels = ['LISA_A', 'LISA_E', 'LISA_T']

    sangria_data, sangria_descr = hdfio.load_array(input_path, name="obs/tdi")

    delta_t = sangria_descr['dt']
    epoch = sangria_descr['t0']

    data = linear_AET_conversion(sangria_data)

    if downsample_delta_t is None or delta_t == downsample_delta_t:
        return to_timeseries(data, channels, delta_t, epoch)

    logger.info("Downsampling to %s", downsample_delta_t)


    n_samples = len(data[channels[0]])
    logger.info("n_samples: %s", n_samples)

    if np.remainder(downsample_delta_t, delta_t) != 0:
        raise ValueError(
            "Downsampled delta_t mst be an integer multiple of the original delta_t"
        )

    decimation = int(downsample_delta_t / delta_t)
    post_decimation_size = (n_samples + decimation - 1) // decimation
    output_data = {
        channel: np.zeros(post_decimation_size, dtype=np.float64)
        for channel in channels
    }
    logger.info("decimation: %s", decimation)
    for chunk_i in tqdm(range(0, len(data[channels[0]]), CHUNK_SIZE)):
        chunk_start = chunk_i
        chunk_end = chunk_i + CHUNK_SIZE
        chunk_end = min(chunk_end, n_samples)
        
        output_chunk_start = chunk_start // decimation
        output_chunk_end = chunk_end // decimation

        dlen = max(1, output_chunk_end - output_chunk_start)
        tlen = dlen * downsample_delta_t
        delta_f = 1 / tlen
        logger.debug("delta_f: %s", delta_f)
        new_nyquist = 1 / downsample_delta_t / 2
        logger.debug("new nyquist: %s", new_nyquist)
        for channel in channels:
            decimated = data[channel][chunk_start:chunk_end][::decimation]
            write_end = output_chunk_start + len(decimated)
            if write_end > post_decimation_size:
                decimated = decimated[:post_decimation_size - output_chunk_start]
                write_end = post_decimation_size
            output_data[channel][output_chunk_start:write_end] = decimated

    return to_timeseries(output_data, channels, downsample_delta_t, epoch)

def load_sangria_timeseries(
    filename,
    data_group="obs/tdi",
):
    tdi_ts, _ = hdfio.load_array(filename, name=data_group)
    X = tdi_ts['X']
    Y = tdi_ts['Y']
    Z = tdi_ts['Z']

    attrs = hdfio.load_attributes(filename, name=data_group)
    delta_t = attrs['dt']
    epoch = attrs['t0']
    
    data = linear_AET_conversion({'X': X, 'Y': Y, 'Z': Z})

    return to_timeseries(data, data.keys(), delta_t, epoch)
