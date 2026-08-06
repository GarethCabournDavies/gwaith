"""
Comparison test for the hwilas inpainting implementation against the pycbc
implementation.
"""

import pytest
import numpy as np

from gwaith.types import TimeSeries as GwaithTimeSeries, FrequencySeries as GwaithFrequencySeries
from pycbc.types import TimeSeries as PycbcTimeSeries, FrequencySeries as PycbcFrequencySeries
from pycbc.strain.gate import gate_and_paint as pycbc_gate_and_paint

from hwilas.data_processing.inpainting import gate_and_paint as hwilas_gate_and_paint

def test_inpainting_replacement():
    """
    Compares the hwilas inpainting implementation to the pycbc implementation.
    """
    # 1. Define test parameters
    sample_rate = 1024  # Hz
    duration = 4  # seconds
    num_samples = duration * sample_rate

    # 2. Create a synthetic TimeSeries with Gaussian noise
    np.random.seed(0)
    noise = np.random.normal(0, 1, num_samples)
    
    # Create pycbc TimeSeries
    pycbc_ts = PycbcTimeSeries(noise, delta_t=1.0 / sample_rate)

    # 3. Create a flat Power Spectral Density (PSD)
    psd_value = 1.0  # Flat PSD for simplicity
    delta_f = 1.0 / duration
    
    pycbc_psd = PycbcFrequencySeries(np.full(num_samples // 2 + 1, psd_value), delta_f=delta_f)

    # 4. Define a gap to inpaint
    gap_start = int(1.0 * sample_rate)
    gap_end = int(1.5 * sample_rate)

    # 5. Run pycbc inpainting
    pycbc_inv_psd = 1.0 / pycbc_psd
    pycbc_inpainted = pycbc_gate_and_paint(pycbc_ts, gap_start, gap_end, pycbc_inv_psd)

    # 6. Run hwilas inpainting
    # Convert pycbc types to gwaith types / basic types
    gwaith_ts = GwaithTimeSeries(pycbc_ts.data, epoch=pycbc_ts._epoch, delta_t=pycbc_ts.delta_t)
    gwaith_psd = GwaithFrequencySeries(pycbc_psd.data, delta_f=pycbc_psd.delta_f)
    
    hwilas_inpainted_ts = hwilas_gate_and_paint(gwaith_ts, [gap_start], [gap_end], gwaith_psd)

    # 7. Compare the results
    # The data should be very close, but may not be identical due to floating point differences.
    np.testing.assert_allclose(pycbc_inpainted.data, hwilas_inpainted_ts.data, rtol=1e-9, atol=1e-9)
