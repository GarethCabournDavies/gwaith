import numpy as np
from bbhx.utils.constants import YRSID_SI
from bbhx.waveformbuild import BBHWaveformFD
from pycbc.coordinates import TIME_OFFSET_20_DEGREES, lisa_to_ssb
from pycbc.psd.analytical_space import omega_length
from pycbc.types import FrequencySeries

DEFAULT_HM_MODE_ARRAY = [(2, 2), (2, 1), (3, 3), (3, 2), (4, 4), (4, 3)]

def get_fd_waveform_from_bbhx(
    tdi='2.0',
    **params
):
    """
    Generate a frequency domain waveform using BBHx, including TDI projection.

    This function is a wrapper around the BBHx waveform generator, intended
    to be a replacement for the custom pycbc `get_fd_det_waveform` call.
    It mimics the logic found in the `gwastro/BBHX-waveform-model` plugin.

    Parameters
    ----------
    tdi : {'1.5', '2.0'}
        Version of TDI to use.
    **params : dict
        A dictionary of waveform parameters.

    Returns
    -------
    dict
        A dictionary containing the projected waveforms for each detector as
        pycbc.types.FrequencySeries objects.
    """
    ifos = params.get('ifos', ['LISA_A', 'LISA_E'])
    
    # 1. Get parameters and convert to types expected by BBHx
    m1 = np.float64(params['mass1'])
    m2 = np.float64(params['mass2'])
    a1 = np.float64(params.get('spin1z', 0.0))
    a2 = np.float64(params.get('spin2z', 0.0))
    inc = np.float64(params.get('inclination', 0.0))
    dist = np.float64(params['distance']) # Already in Mpc
    f_ref = np.float64(params.get('f_ref', params['f_lower']))
    phi_ref = np.float64(params.get('coa_phase', 0.0))
    
    t_offset = TIME_OFFSET_20_DEGREES # Default from BBHX_Phenom
    t_obs_start = np.float64(params['t_obs_start'])
    mode_array = params.get("mode_array", DEFAULT_HM_MODE_ARRAY)
    
    df = np.float64(params['delta_f'])
    length = int(params['length'])
    if length <= 0:
        raise ValueError(f"Waveform length must be positive, got {length}")
    # Build an exact FFT grid from requested bin count to avoid off-by-one
    # effects from floating-point arange stop values.
    freqs = df * np.arange(length, dtype=np.float64)
    
    # 2. Coordinate transformation from LISA frame to SSB
    # This is a critical step from the plugin logic
    t_ref_lisa = np.float64(params['tc']) + t_offset
    lam = np.float64(params['eclipticlongitude'])
    beta = np.float64(params['eclipticlatitude'])
    psi = np.float64(params['polarization'])
    
    t_ref, lam, beta, psi = lisa_to_ssb(
        t_lisa=t_ref_lisa,
        longitude_lisa=lam,
        latitude_lisa=beta,
        polarization_lisa=psi,
        t0=0
    )
    
    # 3. Instantiate the BBHx waveform generator
    # For now, we use a simplified setup. The plugin has more complex caching.
    wave_gen = BBHWaveformFD(
        amp_phase_kwargs={
            "run_phenomd": (params['approximant'] == 'BBHX_PhenomD')
        },
    )
    
    # 4. Call the generator
    wave = wave_gen(
        m1, m2, a1, a2,
        dist, phi_ref, f_ref, inc, lam,
        beta, psi, t_ref,
        freqs=freqs,
        modes=mode_array,
        t_obs_start=t_obs_start / YRSID_SI,
        compress=True,
        fill=True,
        squeeze=True,
        direct=False,
        length=length,
    )[0]

    # 5. Select the desired TDI channels
    wanted = {name: i for i, name in enumerate(['LISA_A', 'LISA_E', 'LISA_T']) if name in ifos}
    output = {}
    
    # 6. Wrap in FrequencySeries and apply shifts/rescaling
    length_of_wave = 1 / df
    loc_of_signal_merger_within_wave = t_ref_lisa % length_of_wave
    
    for channel, tdi_num in wanted.items():
        output[channel] = FrequencySeries(
            wave[tdi_num],
            delta_f=df,
            epoch=t_ref_lisa - loc_of_signal_merger_within_wave,
            copy=False
        )
        # move the merge to the end of the vector
        output[channel] = output[channel].cyclic_time_shift(
            length_of_wave - loc_of_signal_merger_within_wave)
        output[channel].start_time -= t_offset
        
    # 7. Apply TDI 2.0 rescaling if requested
    if str(tdi) == '2.0':
        from bbhx.utils.constants import L_SI
        omega_len = omega_length(f=output['LISA_A'].sample_frequencies, len_arm=L_SI)
        rescale = 2j * np.sin(2 * omega_len) * np.exp(-2j * omega_len)
        for key in output:
            output[key] *= rescale
            
    return output
