import numpy as np
from astropy import units as u
from scipy.constants import c, pi

from bbhx.utils.constants import YRSID_SI, L_SI
from bbhx.waveformbuild import BBHWaveformFD
from gwaith.types import FrequencySeries

from hwilas.waveforms.bbhx_utils import get_bbhx_parameter_dict, TIME_OFFSET_20_DEGREES

DEFAULT_HM_MODE_ARRAY = [(2, 2), (2, 1), (3, 3), (3, 2), (4, 4), (4, 3)]


def get_fd_waveform_from_bbhx(
    tdi='2.0',
    align_tc_zero=True,
    **params
):
    """
    Generate a frequency domain waveform using BBHx, including TDI projection.

    Parameters
    ----------
    tdi : {'1.5', '2.0'}
        Version of TDI to use. Default is '2.0'.
    align_tc_zero : bool, optional
        If True, cyclically time-shifts the waveform so tc is at t = 0 (useful for
        generating matched-filter templates for SNR time series).
        If False (default), retains the embedded tc phase factor for direct single-point
        overlap / likelihood evaluation.
    **params : dict
        Waveform parameters including mass1, mass2, tc, delta_f, length, etc.

    Returns
    -------
    dict
        A dictionary containing the projected waveforms for each IFO as
        gwaith.types.TimeSeries objects. The waveform is generated in the
        frequency domain and converted to the time domain via ``to_timeseries``
        so that it can be (over)whitened before matched filtering.
    """
    ifos = params.get('ifos', ['LISA_A', 'LISA_E'])
    approximant = params.get('approximant', 'BBHX_PhenomHM')

    df = float(params['delta_f'])
    length = int(params['length'])
    if length <= 0:
        raise ValueError(f"Waveform length must be positive, got {length}")

    # Time offset conversion to SSB frame
    t_offset = params.get("t_offset", TIME_OFFSET_20_DEGREES)
    tc_lisa = float(params["tc"])
    t_ref_lisa_for_conversion = tc_lisa + t_offset

    # 1. Assemble parameters for conversion
    bbhx_input_params = {
        "m1": float(params["mass1"]),
        "m2": float(params["mass2"]),
        "chi1z": float(params.get("spin1z", 0.0)),
        "chi2z": float(params.get("spin2z", 0.0)),
        # BBHx expects the luminosity distance in metres; the bank stores it
        # in Mpc, so convert here (1 Mpc -> m).
        "distance": float(params["distance"]) * float(u.Mpc.to("m")),
        "phi_ref": float(params.get("coa_phase", 0.0)),
        "f_ref": float(params.get("f_ref", params.get("f_lower", 0.0))),
        "inc": float(params.get("inclination", 0.0)),
        "lam": float(params["eclipticlongitude"]),
        "beta": float(params["eclipticlatitude"]),
        "psi": float(params["polarization"]),
        "t_ref": float(t_ref_lisa_for_conversion),
    }

    # 2. Convert to SSB frame for BBHx
    t_obs_start = float(params.get('t_obs_start', 0.0))
    bbhx_params = get_bbhx_parameter_dict(
        bbhx_input_params,
        parameter_frame="LISA",
        t0=t_obs_start / YRSID_SI
    )

    raw_modes = params.get("mode_array", DEFAULT_HM_MODE_ARRAY)
    mode_array = [(int(l), int(m)) for l, m in raw_modes]

    # Build frequency grid
    freqs = df * np.arange(length, dtype=np.float64)

    # 3. Instantiate BBHx generator
    wave_gen = BBHWaveformFD(
        amp_phase_kwargs={
            "run_phenomd": (approximant == 'BBHX_PhenomD')
        }
    )

    # 4. Generate waveform
    wave = wave_gen(
        *[float(v) for v in bbhx_params.values()],
        freqs=freqs,
        modes=mode_array,
        t_obs_start=float(t_obs_start / YRSID_SI),
        length=int(length),
        direct=False,
        fill=True,
        squeeze=True
    )

    if wave.ndim == 3:
        wave = wave[0]

    # 5. Map channels to FrequencySeries and optionally apply cyclic shift
    channel_map = {'LISA_A': 0, 'LISA_E': 1, 'LISA_T': 2}
    output = {}

    length_of_wave = 1.0 / df

    for ifo in ifos:
        if ifo in channel_map:
            idx = channel_map[ifo]
            data_vec = np.asarray(wave[idx], dtype=np.complex128)

            if align_tc_zero:
                # Calculate signal offset within the circular buffer
                loc_of_signal_merger_within_wave = tc_lisa % length_of_wave
                epoch_val = tc_lisa - loc_of_signal_merger_within_wave

                fs = FrequencySeries(
                    data=data_vec,
                    delta_f=df,
                    epoch=epoch_val
                )
                # Cyclically roll merger to t = 0
                shift_amount = length_of_wave - loc_of_signal_merger_within_wave
                output[ifo] = fs.cyclic_time_shift(shift_amount)
            else:
                # Direct FD representation with exact tc phase
                output[ifo] = FrequencySeries(
                    data=data_vec,
                    delta_f=df,
                    epoch=tc_lisa
                )

    # 6. Apply TDI 2.0 transfer factor if requested
    if str(tdi) == '2.0':
        omega_len = 2.0 * pi * freqs * L_SI / c
        rescale = 2.0j * np.sin(2.0 * omega_len) * np.exp(-2.0j * omega_len)
        rescale[0] = 0.0

        for key in output:
            output[key] = output[key] * rescale

    # 7. Convert to the time domain so that the waveform can be (over)whitened
    # before matched filtering. Each FrequencySeries is converted via
    # ``to_timeseries``.
    for key in output:
        output[key] = output[key].to_timeseries()

    return output