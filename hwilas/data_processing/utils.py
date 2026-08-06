import numpy as np
from pycbc.types import TimeSeries


def to_timeseries(waveform_dict, delta_t, epoch=0):
    return_dict = {}
    for channel, array in waveform_dict.items():
        array_ts = TimeSeries(array, delta_t=delta_t, epoch=epoch)
        return_dict[channel] = array_ts
    return return_dict

def AET(X,Y,Z, delta_t, epoch=0):
    A = (Z - X)/np.sqrt(2.0)
    E = (X - 2.0*Y + Z)/np.sqrt(6.0)
    T = (X + Y + Z)/np.sqrt(3.0)

    AET_ts = to_timeseries(
        {
            'LISA_A': A,
            'LISA_E': E,
            'LISA_T': T,
        },
        delta_t,
        epoch=epoch
    )

    return AET_ts
