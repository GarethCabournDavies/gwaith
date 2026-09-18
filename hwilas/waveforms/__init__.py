"""
Waveform generation wrapper providing a unified interface for frequency-domain
waveforms, routing BBHx approximants to the direct BBHx generator and falling
back to PyCBC for standard ground-based/LAL approximants. A separate time-domain
PhenomXPy + TDISpline path is also available for MBHB injections.
"""

from .bbhx_phenom import get_fd_waveform_from_bbhx
from .phenomx_tdi import (
    AET_CHANNELS,
    DEFAULT_HM_MODE_ARRAY_TD,
    LISA_CHANNELS,
    build_phenomx_parameters,
    generate_tdi_waveform,
    get_orbit_interpolant,
    get_td_waveform_from_phenomx,
    xyz_to_aet,
)
from .fastbhb import (
    FASTBHB_CHANNELS,
    get_td_waveform_from_fastbhb,
)
import numpy as np
from gwaith.types import FrequencySeries


BBHX_APPROXIMANTS = {
    "bbhx_phenomd",
    "bbhx_phenomhm",
}

# Approximant name(s) routed to the time-domain PhenomXPy + TDISpline generator.
PHENOMX_TD_APPROXIMANTS = {
    "phenomx_td",
    "phenomx_tdi",
    "imrphenomthm",
}

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

    Two generators are supported:

    * BBHx (frequency domain) for the ``BBHX_PhenomD`` / ``BBHX_PhenomHM``
      approximants, and
    * PhenomXPy + TDISpline (time domain) for the MBHB time-domain path.

    The waveform is returned in the time domain so that it can be (over)whitened
    before matched filtering; callers convert back to the frequency domain as
    needed.

    Parameters
    ----------
    approximant : str
        Name of the approximant. BBHx names (e.g. ``'BBHX_PhenomD'``,
        ``'BBHX_PhenomHM'``) are routed to the BBHx generator; PhenomX time-domain
        names (e.g. ``'PhenomXTD'``) are routed to the PhenomXPy + TDISpline
        generator.
    tdi : {'1.5', '2'}, optional
        TDI version for the BBHx LISA detector channels. Default is ``'2.0'``.
        Ignored by the PhenomX time-domain path (which always uses the XYZ frame).
    **waveform_params : dict
        Waveform parameters (masses, spins, sky coords, delta_f, length, etc.).

    Returns
    -------
    dict
        Dictionary mapping channel names (e.g. ``'LISA_A'`` for BBHx, or
        ``'LISA_X'``/``'LISA_Y'``/``'LISA_Z'`` for PhenomX) to TimeSeries.
    """
    if approximant.lower() in BBHX_APPROXIMANTS:
        return get_fd_waveform_from_bbhx(
            approximant=approximant,
            tdi=tdi,
            **waveform_params
        )

    if approximant.lower() in PHENOMX_TD_APPROXIMANTS:
        # The time-domain generator needs the detector-frame sampling grid. If the
        # caller did not supply it, derive one from the shared parameters that drive
        # the BBHx path so both generators produce waveforms of identical duration
        # and resolution. ``delta_t`` (the data sampling interval in seconds) is used
        # directly when present; otherwise it is recovered as ``1 / (delta_f * length)``
        # from the FFT parameters, which gives a grid spanning the full data duration
        # [0, tlen). (Do NOT use ``delta_f * arange(length)``: that is a frequency-grid
        # artifact spanning only ~2 Hz^-1 and would push the waveform evaluation far
        # outside the response grid.)
        if 'detector_times' not in waveform_params:
            length = int(waveform_params['length'])
            if 'delta_t' in waveform_params:
                delta_t = float(waveform_params['delta_t'])
            else:
                delta_f = float(waveform_params['delta_f'])
                delta_t = 1.0 / (delta_f * length)
            waveform_params['detector_times'] = delta_t * np.arange(length)

        return get_td_waveform_from_phenomx(**waveform_params)

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