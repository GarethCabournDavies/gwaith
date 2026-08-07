"""
Comparison test for the hwilas waveform generation implementation against the
pycbc implementation.
"""

import logging
import time

import numpy as np
import pytest

# PyCBC imports
from pycbc.waveform import get_fd_det_waveform as get_fd_waveform_pycbc

# Hwilas imports
from hwilas.waveforms import get_fd_waveform as get_fd_waveform_hwilas
from hwilas.waveforms.bbhx_phenom import DEFAULT_HM_MODE_ARRAY
from hwilas.waveforms.bbhx_utils import TIME_OFFSET_20_DEGREES

logger = logging.getLogger(__name__)

# Define a set of test cases to span the parameter space.
# More cases can be added here to increase coverage.
TEST_CASES = [
    # 1. Low mass, equal mass, low spin, face-on
    dict(mass1=5e5, mass2=5e5, spin1z=0.1, spin2z=-0.1, inclination=0.1, eclipticlongitude=0.5, eclipticlatitude=0.2, polarization=0.1, coa_phase=0.0),
    # 2. High mass, unequal mass (q=0.25), high aligned spins, edge-on
    dict(mass1=2e7, mass2=5e6, spin1z=0.9, spin2z=0.8, inclination=np.pi/2, eclipticlongitude=3.0, eclipticlatitude=-1.0, polarization=np.pi/2, coa_phase=1.0),
    # 3. Mid mass, high mass ratio (q=0.1), anti-aligned spins
    dict(mass1=8e6, mass2=8e5, spin1z=-0.7, spin2z=0.6, inclination=np.pi/4, eclipticlongitude=5.0, eclipticlatitude=1.2, polarization=np.pi-0.1, coa_phase=2.0),
    # 4. Another mid mass, equal mass, zero spin
    dict(mass1=1e6, mass2=1e6, spin1z=0.0, spin2z=0.0, inclination=np.pi/3, eclipticlongitude=1.5, eclipticlatitude=-0.5, polarization=0.7, coa_phase=3.0),
    # 5. High mass, equal mass, high anti-aligned spins
    dict(mass1=1.5e7, mass2=1.5e7, spin1z=0.8, spin2z=-0.8, inclination=2*np.pi/3, eclipticlongitude=4.0, eclipticlatitude=0.0, polarization=0.0, coa_phase=4.0),
]

@pytest.mark.parametrize("case_params", TEST_CASES)
@pytest.mark.parametrize("tdi_version", ["1.5", "2.0"])
@pytest.mark.parametrize("approximant", ["BBHX_PhenomD", "BBHX_PhenomHM"])
def test_waveform_generation_replacement(approximant, tdi_version, case_params):
    """
    Compares the hwilas waveform generation implementation to the pycbc
    implementation for equivalent inputs.
    """
    # 1. Define fixed test parameters
    # These are chosen to be representative of a typical MBHB signal.
    tlen = 365 * 86400  # 1 year
    delta_f = 1.0 / tlen
    # Calculate length for a 1/16 Hz sample rate to keep it manageable
    nyquist = 1.0 / 16.0 / 2.0
    flen = int(nyquist / delta_f) + 1

    base_params = {
        "distance": 10000.0,  # Mpc
        "f_lower": 1e-4,
        "delta_f": delta_f,
        "approximant": approximant,
        "t_obs_start": 30 * 86400.0,
        "tc": tlen / 2.0,  # Coalescence time in the middle of the segment
        "ifos": ["LISA_A", "LISA_E"],
        "t_offset": TIME_OFFSET_20_DEGREES,
    }

    params = base_params.copy()
    params.update(case_params)

    # Set modes based on approximant
    if approximant == "BBHX_PhenomD":
        params["mode_array"] = [(2, 2)]
    elif approximant == "BBHX_PhenomHM":
        params["mode_array"] = DEFAULT_HM_MODE_ARRAY

    # 2. Generate waveform using hwilas
    hwilas_params = params.copy()
    hwilas_params["tdi"] = tdi_version
    hwilas_params["length"] = flen
    start_hwilas = time.perf_counter()
    hwilas_wf = get_fd_waveform_hwilas(**hwilas_params)
    end_hwilas = time.perf_counter()
    duration_hwilas = end_hwilas - start_hwilas

    # 3. Generate waveform using pycbc
    pycbc_params = params.copy()
    pycbc_params.pop("length", None) # pycbc uses f_final, not length
    # pycbc uses f_final, not length, to determine array size
    # To ensure pycbc generates `flen` points, we add delta_f / 2 to f_final.
    # This is a common pattern in pycbc to ensure the highest frequency bin
    # is included when f_final is an exact multiple of delta_f.
    # The highest frequency we want corresponds to flen-1 bins.
    f_nyquist = (flen - 1) * delta_f
    pycbc_params["tdi"] = tdi_version
    pycbc_params["f_final"] = f_nyquist + delta_f / 2
    start_pycbc = time.perf_counter()
    pycbc_wf = get_fd_waveform_pycbc(**pycbc_params)
    end_pycbc = time.perf_counter()
    duration_pycbc = end_pycbc - start_pycbc

    # 4. Compare the results
    for channel in params["ifos"]:
        hwilas_series = hwilas_wf[channel]
        pycbc_series = pycbc_wf[channel]

        # Ensure lengths are the same before comparison
        assert len(hwilas_series) == len(pycbc_series)

        # The waveforms should be very close. We check the normalized overlap
        # (match) to quantify similarity.
        # match = |<h1|h2>| / sqrt(<h1|h1> * <h2|h2>)
        norm_hwilas = np.linalg.norm(hwilas_series.data)
        norm_pycbc = np.linalg.norm(pycbc_series.data)

        # A zero-norm waveform is unexpected for these parameters and may indicate an issue.
        # If one waveform is zero (invalid), the other should be as well.
        is_hwilas_zero = np.isclose(norm_hwilas, 0)
        is_pycbc_zero = np.isclose(norm_pycbc, 0)

        if is_hwilas_zero or is_pycbc_zero:
            if is_hwilas_zero and is_pycbc_zero:
                # This case is consistent, but unexpected. Flag it.
                logger.warning(
                    "Both hwilas and pycbc generated a zero-norm waveform for "
                    "approximant %s case %d. This may indicate an issue.",
                    approximant, TEST_CASES.index(case_params) + 1
                )
                # continue
            else:
                # Mismatch: one is zero and the other is not. This is a failure.
                pytest.fail(
                    f"Waveform norm mismatch: hwilas={norm_hwilas}, pycbc={norm_pycbc}. "
                    "One is zero while the other is not."
                )

        dot_product = np.vdot(hwilas_series.data, pycbc_series.data)
        match = np.abs(dot_product) / (norm_hwilas * norm_pycbc)

        assert match > 0.999, (
            f"Waveform match is too low ({match:.6f}) for channel {channel} "
            f"and approximant {approximant}"
        )

    # 5. Compare performance
    logger.info(
        "Waveform generation time for %s case %d: hwilas=%.2fs, pycbc=%.2fs",
        f"{approximant} (TDI {tdi_version})",
        TEST_CASES.index(case_params) + 1,
        duration_hwilas,
        duration_pycbc
    )
    # assert duration_hwilas < 5 * duration_pycbc, (
    #     f"hwilas waveform generation is too slow: {duration_hwilas:.2f}s "
    #     f"vs {duration_pycbc:.2f}s (pycbc)"
    # )
