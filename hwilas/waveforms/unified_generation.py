"""
Unified waveform generation interface for GW search and analysis.

Provides a single coherent function for generating waveforms that handles:
- Time-domain generators (PhenomX, FastBHB)
- Frequency-domain generators (BBHx)
- Automatic frame conversion (XYZ ↔ AET)
- Optional channel filtering
- Consistent parameter handling across all approximants
"""

import numpy as np

from hwilas.waveforms import get_fd_waveform, AET_CHANNELS, FASTBHB_CHANNELS, LISA_CHANNELS
from hwilas.waveforms.bbhx_utils import TIME_OFFSET_20_DEGREES
from hwilas.data_processing.tdi import linear_AET_conversion


# Time-domain generators that take detector_times
TIME_DOMAIN_DETECTORS = {'phenomx_td', 'fastbhb'}

# Frequency-domain generators that need FD grid parameters
FREQUENCY_DOMAIN_GENERATORS = {'bbhx_phenomd', 'bbhx_phenomhm'}


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
    Generate a waveform with unified interface across all approximants.

    This function handles the complexity of different waveform generators:
    - Time-domain generators (PhenomX_TD, FastBHB) work with detector_times
    - Frequency-domain generators (BBHx) work with delta_f / length
    - Output frame conversion is automatic (XYZ ↔ AET)

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
        Waveform approximant. One of:
        - 'PhenomX_TD': PhenomXPy time-domain with TDI
        - 'FastBHB': LDC FastBHB (lisabeta-backed) time-domain
        - 'BBHX_PhenomD': BBHx frequency-domain (spin-aligned)
        - 'BBHX_PhenomHM': BBHx frequency-domain (higher modes)
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
        Orbit source ('analytic' or 'oem'). Ignored for BBHx generators.
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
    approximant = str(approximant).lower()

    # Validate and normalize approximant name
    if approximant == 'phenomx_td':
        wf_type = 'phenomx_td'
    elif approximant == 'fastbhb':
        wf_type = 'fastbhb'
    elif approximant.startswith('bbhx'):
        wf_type = 'bbhx'
    else:
        raise ValueError(
            f"Unknown approximant '{approximant}'. "
            f"Expected one of: PhenomX_TD, FastBHB, BBHX_PhenomD, BBHX_PhenomHM"
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

    # Time-domain generators (PhenomX_TD, FastBHB)
    if wf_type in TIME_DOMAIN_DETECTORS:
        if detector_times is None:
            raise ValueError(
                f"Approximant '{approximant}' requires 'detector_times' grid"
            )

        detector_times = np.asarray(detector_times, dtype=float).ravel()
        if len(detector_times) < 2:
            raise ValueError("detector_times must have at least 2 samples")

        if delta_t is None:
            delta_t = float(detector_times[1] - detector_times[0])

        if wf_type == 'fastbhb':
            generated = get_fd_waveform(
                'FastBHB',
                start_time=float(detector_times[0]),
                end_time=float(detector_times[-1]),
                delta_t=delta_t,
                **common_params,
            )
        else:  # phenomx_td
            generated = get_fd_waveform(
                'PhenomX_TD',
                detector_times=detector_times,
                orbit=orbit,
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

    # Frequency-domain generators (BBHx)
    else:  # wf_type == 'bbhx'
        if detector_times is not None:
            detector_times = np.asarray(detector_times, dtype=float).ravel()
            n_samples = len(detector_times)
            freq_domain_length = n_samples // 2 + 1
            if delta_t is None:
                if len(detector_times) < 2:
                    raise ValueError("detector_times must have at least 2 samples")
                delta_t = float(detector_times[1] - detector_times[0])
            if delta_f is None:
                delta_f = 1.0 / (n_samples * delta_t)
            t_obs_start = n_samples * delta_t
        else:
            if delta_f is None or delta_t is None:
                raise ValueError(
                    "Frequency-domain generators require either 'detector_times' "
                    "or both 'delta_f' and 'delta_t'"
                )
            delta_f = float(delta_f)
            delta_t = float(delta_t)
            n_samples = int(1.0 / (delta_f * delta_t))
            freq_domain_length = n_samples // 2 + 1
            t_obs_start = n_samples * delta_t
            detector_times = None

        generated = get_fd_waveform(
            approximant,
            delta_f=delta_f,
            length=freq_domain_length,
            t_obs_start=t_obs_start,
            t_offset=t_offset,
            **common_params,
        )

        # BBHx returns AET channels
        data = {
            ch: np.asarray(generated[ch].data, dtype=float)
            for ch in AET_CHANNELS
            if ch in generated
        }

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
