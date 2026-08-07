"""Unit tests for the inpainting module."""

import logging
from unittest.mock import patch

import numpy as np
import pytest
from gwaith.types import FrequencySeries, TimeSeries

from hwilas.data_processing.inpainting import gate_and_paint


@pytest.fixture
def synthetic_data():
    """Create synthetic data for testing."""
    sample_rate = 1024
    duration = 4
    num_samples = duration * sample_rate
    np.random.seed(0)
    noise = np.random.normal(0, 1, num_samples)
    input_ts = TimeSeries(noise, delta_t=1.0 / sample_rate, epoch=1234567890)

    psd_value = 1.0
    delta_f = 1.0 / duration
    freqs = np.fft.rfftfreq(num_samples, d=1.0 / sample_rate)
    psd_data = np.full(len(freqs), psd_value)
    psd = FrequencySeries(psd_data, delta_f=delta_f)

    return input_ts, psd


def test_inpainting_overwhitened_is_zero(synthetic_data):
    """
    Tests that the over-whitened inpainted region of a time series is
    consistent with zero.
    """
    input_ts, psd = synthetic_data
    num_samples = len(input_ts)
    sample_rate = int(1 / input_ts.delta_t)
    freqs = np.fft.rfftfreq(num_samples, d=1.0 / sample_rate)

    gap_start = int(1.0 * sample_rate)
    gap_end = int(1.5 * sample_rate)

    inpainted_ts = gate_and_paint(input_ts, [gap_start], [gap_end], psd)

    # Over-whiten the inpainted data
    fft_inpainted = np.fft.rfft(inpainted_ts.data)
    psd_interp = np.interp(freqs, psd.sample_frequencies.data, psd.data)
    whitened_fft = fft_inpainted / np.sqrt(psd_interp)

    whitened_ts_data = np.fft.irfft(whitened_fft, n=num_samples)

    # Check that the whitened data in the gap is close to zero
    whitened_gap_data = whitened_ts_data[gap_start:gap_end]
    assert np.all(np.isclose(whitened_gap_data, 0, atol=1e-9))


def test_no_gaps(synthetic_data):
    """Test that the function returns a copy of the original data when there are no gaps."""
    input_ts, psd = synthetic_data
    inpainted_ts = gate_and_paint(input_ts, [], [], psd)
    assert np.array_equal(input_ts.data, inpainted_ts.data)
    assert input_ts is not inpainted_ts  # Ensure it's a copy
    assert input_ts.epoch == inpainted_ts.epoch
    assert input_ts.delta_t == inpainted_ts.delta_t


def test_multiple_gaps(synthetic_data):
    """Test inpainting with multiple, non-overlapping gaps."""
    input_ts, psd = synthetic_data
    sample_rate = int(1 / input_ts.delta_t)

    gaps = [
        (int(0.5 * sample_rate), int(0.6 * sample_rate)),
        (int(2.1 * sample_rate), int(2.2 * sample_rate)),
    ]
    gap_starts = [g[0] for g in gaps]
    gap_ends = [g[1] for g in gaps]

    inpainted_ts = gate_and_paint(input_ts, gap_starts, gap_ends, psd)

    # A simple check: the data outside the gaps should not change
    mask = np.ones(len(input_ts), dtype=bool)
    for start, end in gaps:
        mask[start:end] = False

    assert np.allclose(
        input_ts.data[mask], inpainted_ts.data[mask], atol=1e-9
    )
    # The main correctness check is done in test_inpainting_overwhitened_is_zero
    # Here we just want to ensure it runs with multiple gaps.


def test_input_output_types_and_properties(synthetic_data):
    """Test that input types are handled and output types/properties are correct."""
    input_ts, psd = synthetic_data
    sample_rate = int(1 / input_ts.delta_t)
    gap_start = int(1.0 * sample_rate)
    gap_end = int(1.5 * sample_rate)

    inpainted_ts = gate_and_paint(input_ts, [gap_start], [gap_end], psd)

    assert isinstance(inpainted_ts, TimeSeries)
    assert inpainted_ts.epoch == input_ts.epoch
    assert inpainted_ts.delta_t == input_ts.delta_t
    assert len(inpainted_ts) == len(input_ts)


@patch("hwilas.data_processing.inpainting.cg")
def test_cg_convergence_warning(mock_cg, synthetic_data, caplog):
    """Test that a warning is logged if CG does not converge."""
    input_ts, psd = synthetic_data
    sample_rate = int(1 / input_ts.delta_t)
    gap_start = int(1.0 * sample_rate)
    gap_end = int(1.5 * sample_rate)

    # Simulate non-convergence
    mock_cg.return_value = (np.zeros(gap_end - gap_start), 1)

    with caplog.at_level(logging.WARNING):
        gate_and_paint(input_ts, [gap_start], [gap_end], psd)
    
    assert "CG did not fully converge" in caplog.text
