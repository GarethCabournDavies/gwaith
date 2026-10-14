"""Shared detector sampling and channel validation for waveform adapters."""
import numpy as np

from .constants import AET_CHANNELS, LISA_CHANNELS


def detector_grid(params):
    """Return a uniform grid, retaining its absolute detector epoch."""
    dt = float(params['delta_t'])
    start = float(params.get('start_time', 0.))
    if not np.isfinite(dt) or dt <= 0 or not np.isfinite(start):
        raise ValueError('A finite epoch and positive delta_t are required')
    if 'length' in params:
        n = int(params['length'])
        if n != params['length']:
            raise ValueError('length must be an integer')
    else:
        duration = (float(params['end_time'])-start)/dt
        n = round(duration)
        if not np.isclose(duration, n):
            raise ValueError('The time window must contain an integer number of samples')
    if n < 2:
        raise ValueError('At least two detector samples are required')
    return start+np.arange(n)*dt, dt


def tdi_version(tdi):
    """Normalize public TDI generation names."""
    value = str(tdi)
    if value in ('2', '2.0'):
        return '2'
    if value == '1.5':
        return value
    raise ValueError("tdi must be '1.5', '2', or '2.0'")


def requested_channels(params):
    """Reject unknown channels instead of silently dropping them."""
    channels = tuple(params.get('ifos', AET_CHANNELS))
    if not channels or any(c not in LISA_CHANNELS for c in channels):
        raise ValueError(f'Channels must be selected from {LISA_CHANNELS}')
    return channels
