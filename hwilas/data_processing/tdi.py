"""TDI data stream conversion utilities."""
import numpy as np
from gwaith.types import TimeSeries


def to_timeseries(waveform_dict, delta_t, epoch=0):
    """Converts a dictionary of numpy arrays to a dictionary of gwaith TimeSeries."""
    return_dict = {}
    for channel, array in waveform_dict.items():
        array_ts = TimeSeries(array, delta_t=delta_t, epoch=epoch)
        return_dict[channel] = array_ts
    return return_dict

def linear_AET_conversion(X, Y, Z):
    """Converts X, Y, Z numpy arrays to a dictionary of A, E, T numpy arrays."""
    A = (Z - X)/np.sqrt(2.0)
    E = (X - 2.0*Y + Z)/np.sqrt(6.0)
    T = (X + Y + Z)/np.sqrt(3.0)
    return {
        'LISA_A': A,
        'LISA_E': E,
        'LISA_T': T,
    }

def AET(X, Y, Z, delta_t, epoch=0):
    """Converts X, Y, Z data streams to A, E, T TimeSeries."""
    aet_arrays = linear_AET_conversion(X, Y, Z)
    return to_timeseries(aet_arrays, delta_t, epoch=epoch)