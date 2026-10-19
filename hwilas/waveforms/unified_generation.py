"""
Unified waveform generation interface for GW search and analysis.

Provides a single coherent function for generating waveforms using the LDC
FastBHB (lisabeta) time-domain generator, with automatic frame conversion
(XYZ -> AET when needed), optional channel filtering, and consistent parameter
handling. PhenomTHM uses PhenomXPy modes and a TDISpline detector response.
"""

import numpy as np

from hwilas.waveforms import get_fd_waveform, AET_CHANNELS
from hwilas.waveforms.constants import TIME_OFFSET_20_DEGREES
from hwilas.data_processing.tdi import linear_AET_conversion


def generate_waveform(
    signal,
    approximant,
    detector_times=None,
    *,
    delta_f=None,
    delta_t=None,
    channels=None,
    tdi_generation="1.5",
    orbit="analytic",
    t_offset=TIME_OFFSET_20_DEGREES,
    padding_fill=0.0,
):
    """
    Generate a waveform with the unified interface.

    All models return time-domain detector strain on ``detector_times``. Output frame conversion is automatic (XYZ ↔ AET).

    Parameters
    ----------
    signal : dict
        Signal parameters with keys:
        - mass1, mass2 (required)
        - spin1z, spin2z (required, use 0.0 for non-spinning)
        - distance (required, in Mpc)
        - eclipticlongitude, eclipticlatitude (required, sky location in rad)
        - inclination, polarization (required, Euler angles in rad)
        - phi_ref, coa_phase (required, coalescence phase in rad)
        - tc (required, coalescence time in seconds)
        - spin1_polar_angle, spin2_polar_angle (optional, in rad; default 0.0)
    approximant : str
        'FastBHB', 'PhenomTHM' (PhenomXPy + TDISpline), or an external provider name.
    detector_times : array_like, optional
        Time grid for output (required for time-domain generators).
    delta_f : float, optional
        Accepted for API compatibility; detector_times sets the output grid.
    delta_t : float, optional
        Time sampling interval (seconds). Inferred from detector_times if not given.
    channels : list of str, optional
        Which TDI channels to return. If None, returns all generated channels.
        Valid: LISA_A, LISA_E, LISA_T (AET) or LISA_X, LISA_Y, LISA_Z (XYZ).
    tdi_generation : str, default '1.5'
        TDI generation version ('1.5' or '2.0').
    orbit : str or object, default 'analytic'
        PhenomTHM orbit ('analytic', 'oem', or an equatorial orbit object).
        External providers receive this value as their orbit parameter.
    t_offset : float, default TIME_OFFSET_20_DEGREES
        Accepted for API compatibility; model parameters use the SSB frame.
    padding_fill : float, default 0.0
        Accepted for API compatibility.

    Returns
    -------
    dict
        Dictionary mapping channel names to numpy arrays (or TimeSeries if
        generated as time-domain).
        Shape: (n_samples,) for each channel.

    Raises
    ------
    ValueError
        If approximant is unknown, required parameters are missing, or channel
        specifications are incompatible with the chosen approximant.
    """
    from .plugins import normalise_approximant
    approximant = normalise_approximant(approximant)
    # Extract parameters and apply defaults
    mass1 = float(signal['mass1'])
    mass2 = float(signal['mass2'])
    spin1z = float(signal['spin1z'])
    spin2z = float(signal['spin2z'])
    s1_pa = signal.get('spin1_polar_angle')
    spin1_polar_angle = float(s1_pa) if s1_pa is not None else 0.0
    s2_pa = signal.get('spin2_polar_angle')
    spin2_polar_angle = float(s2_pa) if s2_pa is not None else 0.0
    distance = float(signal['distance'])
    eclipticlongitude = float(signal['eclipticlongitude'])
    eclipticlatitude = float(signal['eclipticlatitude'])
    inclination = float(signal['inclination'])
    polarization = float(signal['polarization'])
    phi_ref = float(signal.get('phi_ref', signal.get('coa_phase', 0.)))
    coa = signal.get('coa_phase')
    coa_phase = float(coa) if coa is not None else phi_ref
    tc = float(signal['tc'])

    # Build common parameters dictionary
    common_params = dict(
        mass1=mass1,
        mass2=mass2,
        spin1z=spin1z,
        spin2z=spin2z,
        spin1_polar_angle=spin1_polar_angle,
        spin2_polar_angle=spin2_polar_angle,
        distance=distance,
        eclipticlongitude=eclipticlongitude,
        eclipticlatitude=eclipticlatitude,
        inclination=inclination,
        polarization=polarization,
        phi_ref=phi_ref,
        coa_phase=coa_phase,
        tc=tc,
        tdi=tdi_generation,
    )

    # Keep model-specific settings such as f_ref, modes and orbit epochs.
    common_params.update({key: value for key, value in signal.items() if key not in common_params})
    common_params['tdi'] = tdi_generation
    for key in ('start_time', 'end_time', 'length', 'delta_t', 'delta_f', 'ifos'):
        common_params.pop(key, None)
    if str(approximant).lower() != 'fastbhb':
        common_params['orbit'] = orbit

    if detector_times is None:
        raise ValueError(f"Approximant {approximant!r} requires a detector_times grid")

    detector_times = np.asarray(detector_times, dtype=float).ravel()
    if len(detector_times) < 2:
        raise ValueError("detector_times must have at least 2 samples")

    if delta_t is None:
        delta_t = float(detector_times[1] - detector_times[0])

    if delta_t <= 0 or not np.allclose(np.diff(detector_times), delta_t):
        raise ValueError('detector_times must be uniformly sampled with positive delta_t')

    generated = get_fd_waveform(
        approximant,
        start_time=float(detector_times[0]),
        end_time=float(detector_times[-1])+delta_t,
        delta_t=delta_t,
        **common_params,
    )

    # Convert to AET if needed
    if all(ch in generated for ch in AET_CHANNELS):
        data = {
            ch: np.asarray(generated[ch].data, dtype=float)
            for ch in AET_CHANNELS
        }
    elif all(ch in generated for ch in ['LISA_X', 'LISA_Y', 'LISA_Z']):
        X = np.asarray(generated['LISA_X'].data, dtype=float)
        Y = np.asarray(generated['LISA_Y'].data, dtype=float)
        Z = np.asarray(generated['LISA_Z'].data, dtype=float)
        data = linear_AET_conversion(X, Y, Z)
    else:
        raise ValueError(
            "Generated waveform did not expose known channel sets. "
            "Expected either AET or XYZ channels."
        )

    if channels is not None and any(ch in ('LISA_X', 'LISA_Y', 'LISA_Z') for ch in channels):
        A, E, T = (data[ch] for ch in AET_CHANNELS)
        data.update(LISA_X=T/np.sqrt(3)-A/np.sqrt(2)+E/np.sqrt(6),
                    LISA_Y=T/np.sqrt(3)-2*E/np.sqrt(6),
                    LISA_Z=T/np.sqrt(3)+A/np.sqrt(2)+E/np.sqrt(6))

    # Filter channels if requested
    if channels is not None:
        channels_set = tuple(channels)
        if not channels_set or any(ch not in data for ch in channels_set):
            raise ValueError(
                f"No requested channels found in generated data. "
                f"Requested: {channels}, Generated: {list(data.keys())}"
            )
        data = {ch: data[ch] for ch in channels_set if ch in data}

    return data
