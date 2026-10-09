"""Helper functions for creating and using template banks."""
import copy
import numpy


def _first_existing_key(h5_group, candidates):
    """Return the first existing dataset name from candidates, else None."""
    for name in candidates:
        if name in h5_group:
            return name
    return None


def parse_mode_array(input_params):
    """Ensures mode_array argument in a dictionary of input parameters is
    a list of tuples of (l, m), where l and m are ints.

    Accepted formats for the ``mode_array`` argument is a list of tuples of
    ints (e.g., ``[(2, 2), (3, 3), (4, 4)]``), a space-separated string giving
    the modes (e.g., ``22 33 44``), or an array of ints or floats (e.g.,
    ``[22., 33., 44.]``.
    """
    if 'mode_array' in input_params and input_params['mode_array'] is not None:
        mode_array = input_params['mode_array']
        # If it is bytes, convert to a string
        if isinstance(mode_array, bytes):
            mode_array = mode_array.decode('utf-8')
        # If it is a string, make sure that it is split on any spaces
        if isinstance(mode_array, str):
            mode_array = mode_array.split()
        # If it isnt a list, make sure that it is
        if not isinstance(mode_array, (numpy.ndarray, list)):
            mode_array = [mode_array]
        for ii, ma in enumerate(mode_array):
            # if ma is a float or int, convert to str (e.g., 22. -> '22'), so
            # that...
            if isinstance(ma, (float, int)):
                ma = str(int(ma))
            # if ma is a str convert to (int, int) (e.g., '22' -> (2, 2))
            if isinstance(ma, str):
                if len(ma) == 2: # format is "22", presumed m is positive
                    l, m = ma
                elif len(ma) == 3: # format is "2+2", "2-2", signed m
                    l = ma[0]
                    m = ma[1:]
                else:
                    raise ValueError(f"Unknown lm mode string format: {ma}")

                ma = (int(l), int(m))
            mode_array[ii] = ma
        input_params['mode_array'] = mode_array
    return input_params


def waveform_from_bank(bank_file, idx, waveform_params_shared, data_length):
    bank_wf = copy.deepcopy(waveform_params_shared)
    # Update waveform params to use the ones from the bank file
    # We use the tc provided in waveform_params_shared (which is shifted by 1 day)
    # If not provided, fallback to data_length - 1 day
    bank_wf['tc'] = waveform_params_shared.get('tc', data_length - 86400.0)
    bank_wf['mass1'] = bank_file['mass1'][idx]
    bank_wf['mass2'] = bank_file['mass2'][idx]
    bank_wf['inclination'] = bank_file['inclination'][idx]
    bank_wf['polarization'] = bank_file['polarization'][idx]
    bank_wf['spin1z'] = bank_file['spin1z'][idx]
    bank_wf['spin2z'] = bank_file['spin2z'][idx]
    #bank_wf['coa_phase'] = hfile['coa_phase'][idx]
    bank_wf['eclipticlatitude'] = bank_file['eclipticlatitude'][idx]
    bank_wf['eclipticlongitude'] = bank_file['eclipticlongitude'][idx]
    bank_wf['distance'] = bank_file['distance'][idx]

    # Coalescence phase is optional in bank rows; when present, propagate it
    # explicitly so BBHx/FastBHB generators do not silently fall back to 0.
    phase_field = _first_existing_key(
        bank_file,
        ['coa_phase', 'CoaPhase', 'PhaseAtCoalescence', 'phi_ref'],
    )
    if phase_field is not None:
        bank_wf['coa_phase'] = bank_file[phase_field][idx]
        bank_wf['phi_ref'] = bank_file[phase_field][idx]

    # Correctly call parse_mode_array by passing a dictionary
    temp_params = {'mode_array': bank_file["mode_array"][idx]}
    bank_wf["mode_array"] = parse_mode_array(temp_params)['mode_array']

    bank_wf['f_ref'] = max(
        waveform_params_shared.get('f_ref', 0),
        bank_file['f_lower'][idx]
    )

    return bank_wf
