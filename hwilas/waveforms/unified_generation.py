"""
Unified waveform generation interface for GW search and analysis.

Provides a single coherent function for generating waveforms using the LDC
FastBHB (lisabeta) time-domain generator, with automatic frame conversion
(XYZ -> AET when needed), optional channel filtering, and consistent parameter
handling. The legacy BBHx and PhenomXPy generators have been removed from the
project; FastBHB is the only supported path.
"""

import numpy as np

from hwilas.waveforms import get_fd_waveform, AET_CHANNELS, FASTBHB_CHANNELS
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

    The only supported generator is LDC FastBHB (lisabeta), which works in the
    time domain: provide ``detector_times`` (or both ``delta_t`` and
    ``delta_f``). Output frame conversion is automatic (XYZ ↔ AET).

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
        Waveform approximant. Only 'FastBHB' (LDC FastBHB, lisabeta-backed,
        time-domain) is supported.
    detector_times : array_like, optional
        Time grid for output (required for time-domain generators).
        For frequency-domain, provide either this or both delta_t and delta_f.
    delta_f : float, optional
        Frequency resolution (Hz). Required for frequency-domain generators.
    delta_t : float, optional
        Time sampling interval (seconds). Inferred from detector_times if not given.
    channels : list of str, optional
        Which TDI channels to return. If None, returns all generated channels.
        Valid: LISA_A, LISA_E, LISA_T (AET) or LISA_X, LISA_Y, LISA_Z (XYZ).
    tdi_generation : str, default '1.5'
        TDI generation version ('1.5' or '2.0').
    orbit : str, default 'analytic'
        Orbit source ('analytic' or 'oem'). Ignored by the FastBHB generator.
    t_offset : float, default TIME_OFFSET_20_DEGREES
        Time offset for SSB frame conversion (seconds).
    padding_fill : float, default 0.0
        Value to use for padding outside the generated window (if detector_times
        is provided for a frequency-domain generator).

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
    if str(approximant).lower() != 'fastbhb':
        raise ValueError(
            f"Unknown approximant '{approximant}'. Only 'FastBHB' is supported."
        )

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
    phi_ref = float(signal['phi_ref'])
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

    # FastBHB time-domain generator
    if detector_times is None:
        raise ValueError("Approximant 'FastBHB' requires a 'detector_times' grid")

    detector_times = np.asarray(detector_times, dtype=float).ravel()
    if len(detector_times) < 2:
        raise ValueError("detector_times must have at least 2 samples")

    if delta_t is None:
        delta_t = float(detector_times[1] - detector_times[0])

    generated = get_fd_waveform(
        'FastBHB',
        start_time=float(detector_times[0]),
        end_time=float(detector_times[-1]),
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

    # Filter channels if requested
    if channels is not None:
        channels_set = set(ch for ch in channels if ch in data)
        if not channels_set:
            raise ValueError(
                f"No requested channels found in generated data. "
                f"Requested: {channels}, Generated: {list(data.keys())}"
            )
        data = {ch: data[ch] for ch in channels_set if ch in data}

    return data
