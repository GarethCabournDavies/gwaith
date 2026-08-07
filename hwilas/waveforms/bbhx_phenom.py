import numpy as np
from scipy.constants import c, pi

from bbhx.utils.constants import YRSID_SI
from bbhx.waveformbuild import BBHWaveformFD
from gwaith.types import FrequencySeries

from hwilas.waveforms.bbhx_utils import get_bbhx_parameter_dict, TIME_OFFSET_20_DEGREES

DEFAULT_HM_MODE_ARRAY = [(2, 2), (2, 1), (3, 3), (3, 2), (4, 4), (4, 3)]

def get_fd_waveform_from_bbhx(
    tdi='2.0',
    **params
):
    """
    Generate a frequency domain waveform using BBHx, including TDI projection.

    This function is a wrapper around the BBHx waveform generator, intended
    to provide a unified interface for waveform generation, using the utility
    modules for parameter conversion.

    Parameters
    ----------
    tdi : {'1.5', '2.0'}
        Version of TDI to use.
    **params : dict
        A dictionary of waveform parameters.

    Returns
    -------
    dict
        A dictionary containing the projected waveforms for each IFO as
        gwaith.types.FrequencySeries objects.
    """
    ifos = params.get('ifos', ['LISA_A', 'LISA_E'])

    # The pycbc-bbhx plugin applies a time offset before frame conversion
    # to account for LISA's position. We replicate that logic here.
    t_offset = params.get("t_offset", TIME_OFFSET_20_DEGREES)
    t_ref_lisa_for_conversion = params["tc"] + t_offset

    # 1. Assemble parameters for conversion to BBHx frame
    # The input parameters are assumed to be in the LISA frame.
    bbhx_input_params = {
        "m1": params["mass1"],
        "m2": params["mass2"],
        "chi1z": params.get("spin1z", 0.0),
        "chi2z": params.get("spin2z", 0.0),
        "distance": params["distance"],
        "inc": params.get("inclination", 0.0),
        "phi_ref": params.get("coa_phase", 0.0),
        "f_ref": params.get("f_ref", params["f_lower"]),
        "lam": params["eclipticlongitude"],
        "beta": params["eclipticlatitude"],
        "psi": params["polarization"],
        "t_ref": t_ref_lisa_for_conversion,
    }

    # 2. Convert to SSB frame for BBHx
    bbhx_params = get_bbhx_parameter_dict(
        bbhx_input_params,
        parameter_frame="LISA",
        t0=params['t_obs_start'] / YRSID_SI
    )

    t_obs_start = np.float64(params['t_obs_start'])
    mode_array = params.get("mode_array", DEFAULT_HM_MODE_ARRAY)

    df = np.float64(params['delta_f'])
    length = int(params['length'])
    if length <= 0:
        raise ValueError(f"Waveform length must be positive, got {length}")

    # Build an exact FFT grid from requested bin count to avoid off-by-one
    # effects from floating-point arange stop values.
    freqs = df * np.arange(length, dtype=np.float64)

    # 3. Instantiate and call the BBHx waveform generator
    wave_gen = BBHWaveformFD(
        amp_phase_kwargs={
            "run_phenomd": (params['approximant'] == 'BBHX_PhenomD')
        },
    )

    wave = wave_gen(
        bbhx_params["m1"], bbhx_params["m2"], bbhx_params["chi1z"], bbhx_params["chi2z"],
        bbhx_params["distance"], bbhx_params["phi_ref"], bbhx_params["f_ref"],
        bbhx_params["inc"], bbhx_params["lam"], bbhx_params["beta"],
        bbhx_params["psi"], bbhx_params["t_ref"],
        freqs=freqs,
        modes=mode_array,
        t_obs_start=t_obs_start / YRSID_SI,
        compress=True,
        fill=True,
        squeeze=True,
        direct=False,
        length=length,
    )[0]

    # 4. Select the desired TDI channels and wrap in FrequencySeries
    wanted = {name: i for i, name in enumerate(['LISA_A', 'LISA_E', 'LISA_T']) if name in ifos}
    output = {}

    # The coalescence time in the LISA frame is the original 'tc'
    t_ref_lisa = np.float64(params['tc'])
    length_of_wave = 1 / df
    loc_of_signal_merger_within_wave = t_ref_lisa % length_of_wave

    for channel, tdi_num in wanted.items():
        output[channel] = FrequencySeries(
            wave[tdi_num],
            delta_f=df,
            epoch=t_ref_lisa - loc_of_signal_merger_within_wave,
        )
        # move the merge to the end of the vector
        output[channel] = output[channel].cyclic_time_shift(
            length_of_wave - loc_of_signal_merger_within_wave)


    # 5. Apply TDI 2.0 rescaling if requested
    if str(tdi) == '2.0':
        from bbhx.utils.constants import L_SI
        f = np.asarray(output['LISA_A'].sample_frequencies.data)
        omega_len = 2 * pi * f * L_SI / c
        rescale = 2j * np.sin(2 * omega_len) * np.exp(-2j * omega_len)
        for key in output:
            output[key] *= rescale

    return output
