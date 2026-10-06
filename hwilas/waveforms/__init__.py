"""
Waveform generation wrapper providing a unified interface for time-domain
waveforms, routing to the LDC FastBHB (lisabeta) generator. The legacy BBHx and
PhenomXPy generators have been removed from the project; FastBHB is the only
supported path.
"""

from .fastbhb import FASTBHB_CHANNELS, get_td_waveform_from_fastbhb
from .constants import AET_CHANNELS, LISA_CHANNELS
import numpy as np
from gwaith.types import FrequencySeries


# Approximant name(s) routed to the time-domain FastBHB (lisabeta) generator.
FASTBHB_APPROXIMANTS = {
    "fastbhb",
}


def get_fd_waveform(
    approximant,
    tdi='2',
    **waveform_params
):
    """
    Generate a detector waveform as a dictionary of ``gwaith.types.TimeSeries``
    objects.

    The LDC FastBHB (lisabeta) time-domain generator is the only supported
    path. The waveform is returned in the time domain so that it can be
    (over)whitened before matched filtering; callers convert back to the
    frequency domain as needed.

    Parameters
    ----------
    approximant : str
        Name of the approximant. Only ``'FastBHB'`` (case-insensitive) is
        recognised.
    tdi : {'1.5', '2'}, optional
        TDI version for the LISA detector channels. Default is ``'2'``.
    **waveform_params : dict
        Waveform parameters (masses, spins, sky coords, delta_f, length, etc.).

    Returns
    -------
    dict
        Dictionary mapping channel names (``'LISA_A'``/``'LISA_E'``/``'LISA_T'``)
        to TimeSeries.
    """
    if approximant.lower() in FASTBHB_APPROXIMANTS:
        # The FastBHB generator needs a detector-frame sampling grid. If the
        # caller did not supply an explicit window (start_time/end_time),
        # derive one from the shared FFT parameters: ``delta_t`` is used
        # directly when present, otherwise recovered as 1 / (delta_f * length)
        # so the grid spans the full data duration [0, tlen).
        if 'end_time' not in waveform_params:
            length = int(waveform_params['length'])
            if 'delta_t' not in waveform_params:
                delta_f = float(waveform_params['delta_f'])
                waveform_params['delta_t'] = 1.0 / (delta_f * length)
            waveform_params.setdefault('start_time', 0.0)
            waveform_params['end_time'] = (
                float(waveform_params['start_time'])
                + length * float(waveform_params['delta_t'])
            )

        return get_td_waveform_from_fastbhb(
            approximant=approximant,
            tdi=tdi,
            **waveform_params
        )

    raise ValueError(f"Approximant {approximant} not recognised")