"""Unit tests for the inpainting module."""

import pytest
import numpy as np
from gwaith.types import TimeSeries, FrequencySeries

from hwilas.data_processing.inpainting import gate_and_paint

def test_inpainting_overwhitened_is_zero():
    """
    Tests that the over-whitened inpainted region of a time series is consistent
    with zero.
    """
    # 1. Define test parameters
    sample_rate = 1024  # Hz
    duration = 4  # seconds
    num_samples = duration * sample_rate
    
    # 2. Create a synthetic TimeSeries with Gaussian noise
    np.random.seed(0)
    noise = np.random.normal(0, 1, num_samples)
    times = np.arange(num_samples) / sample_rate
    
    input_ts = TimeSeries(noise, epoch=times[0], delta_t=1.0 / sample_rate)
    
    # 3. Create a flat Power Spectral Density (PSD)
    psd_value = 1.0  # Flat PSD for simplicity
    delta_f = 1.0 / duration
    freqs = np.fft.rfftfreq(num_samples, d=1.0/sample_rate)
    psd_data = np.full(len(freqs), psd_value)
    
    psd = FrequencySeries(psd_data, delta_f=delta_f)

    # 4. Define a gap to inpaint
    gap_start = int(1.0 * sample_rate)  # 1 second in
    gap_end = int(1.5 * sample_rate)    # 1.5 seconds in
    gap_indices = [gap_start]
    gap_end_indices = [gap_end]

    # Create a copy of the original data in the gap for later comparison if needed
    # original_gap_data = input_ts.data[gap_start:gap_end].copy()
    
    # 5. Apply the inpainting function
    inpainted_ts = gate_and_paint(input_ts, gap_indices, gap_end_indices, psd)

    # 6. Over-whiten the inpainted data
    # The theory of inpainting is that the whitened data should be zero in the gap
    
    # Take the FFT of the inpainted time series
    fft_inpainted = np.fft.rfft(inpainted_ts.data)
    
    # Whiten the FFT data. Handle division by zero for DC component if psd[0] is 0
    whitened_fft = np.zeros_like(fft_inpainted, dtype=np.complex128)
    
    # Check psd at each frequency bin.
    # The psd is a FrequencySeries, so we can interpolate it to the FFT frequencies
    psd_interp = np.interp(freqs, psd.sample_frequencies.data, psd.data)

    non_zero_psd_indices = psd_interp > 0
    whitened_fft[non_zero_psd_indices] = fft_inpainted[non_zero_psd_indices] / np.sqrt(psd_interp[non_zero_psd_indices])

    # Inverse FFT to get the whitened time series
    whitened_ts_data = np.fft.irfft(whitened_fft, n=num_samples)
    
    # 7. Check that the whitened data in the gap is close to zero
    whitened_gap_data = whitened_ts_data[gap_start:gap_end]
    
    # Check if all values in the whitened gap are close to zero
    assert np.all(np.isclose(whitened_gap_data, 0, atol=1e-9))
