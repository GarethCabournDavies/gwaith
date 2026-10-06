import logging
from typing import List, Tuple

import numpy as np
import numpy.typing as npt
from scipy.sparse.linalg import LinearOperator, cg
from scipy.signal import fftconvolve

from gwaith.types import TimeSeries, FrequencySeries

# Initialize a named logger for this module
logger = logging.getLogger(__name__)

def gate_and_paint(
    input_data: TimeSeries,
    gap_start_indices: List[int],
    gap_end_indices: List[int],
    psd: FrequencySeries
) -> TimeSeries:
    """
    Inpaints multiple gaps in stationary time-series data simultaneously
    using FFT-accelerated Conjugate Gradient (Matrix-Free PCG).
    
    Parameters:
    -----------
    input_data : gwaith.TimeSeries
        Time series data (1D array).
    gap_start_indices : list[int]
        List of start indices (inclusive) for each gap.
    gap_end_indices : list[int]
        List of end indices (exclusive) for each gap.
    psd : gwaith.FrequencySeries
        Power Spectral Density of the background noise.
        
    Returns:
    --------
    inpainted_data : gwaith.TimeSeries
        Data array with all gaps filled jointly.
    """
    logger.debug("Starting gate_and_paint function.")
    # Convert data to standard numpy array if needed
    time_series_array: np.ndarray = np.asarray(input_data.data, dtype=np.float64)
    num_samples: int = len(input_data)
    sampling_rate_hz: float = float(1. / input_data.delta_t)

    logger.debug(
        "Initial parameters: num_samples=%d, sampling_rate_hz=%.2f, "
        "num_gaps=%d", num_samples, sampling_rate_hz, len(gap_start_indices)
    )

    # 1. Build composite multi-gap mask
    composite_gap_mask: npt.NDArray[np.bool_] = np.zeros(num_samples, dtype=bool)
    for start_idx, end_idx in zip(gap_start_indices, gap_end_indices):
        composite_gap_mask[start_idx:end_idx] = True
        
    missing_indices: npt.NDArray[np.int_] = np.where(composite_gap_mask)[0]
    num_missing_samples: int = len(missing_indices)
    
    logger.debug(
        "Composite gap mask built. Total samples: %d, Missing samples: %d (%.2f%%)",
        num_samples, num_missing_samples, (num_missing_samples / num_samples) * 100
    )

    if num_missing_samples == 0:
        logger.info("No gaps found, returning original data.")
        # Ensure we return a numpy array copy, even if input_data was a TimeSeries
        return input_data.copy()

    # 2. Prepare for whitening operation (C^-1)
    # The whitening operator in Fourier domain is 1/PSD.
    # We handle potential infinities in the PSD (where 1/inf -> 0).
    #
    # The FFT of a length-`num_samples` real signal has exactly
    # ``num_samples // 2 + 1`` bins. The supplied PSD may not match that length
    # (e.g. when the caller truncated the padded data length independently of the
    # PSD grid), which would cause a broadcast error below. Rebuild the inverse
    # PSD at exactly the required number of bins, interpolating from whatever
    # frequency grid the supplied PSD uses so the shapes always agree.
    rfft_len: int = num_samples // 2 + 1
    psd_data: np.ndarray = np.asarray(psd.data, dtype=np.float64)
    if len(psd_data) != rfft_len:
        logger.warning(
            "PSD length (%d) does not match the required FFT bin count (%d) for "
            "%d time samples; interpolating the PSD onto a matching grid.",
            len(psd_data), rfft_len, num_samples,
        )
        n_src = max(len(psd_data), 2)
        # Source frequency axis of the supplied PSD (Hz).
        src_freqs: np.ndarray = (np.arange(n_src) * psd.delta_f)[:len(psd_data)]
        # Target frequency axis covering [0, Nyquist] with rfft_len bins.
        target_freqs: npt.NDArray[np.float64] = (
            np.linspace(0.0, sampling_rate_hz / 2.0, rfft_len)
        )
        psd_data = np.interp(target_freqs, src_freqs, psd_data)

    inv_psd_values: npt.NDArray[np.float64] = 1.0 / (psd_data ** 2)
    inv_psd_values[~np.isfinite(inv_psd_values)] = 0.0

    def whiten_and_mask(input_vector: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        """Applies the C^-1 operator and returns the result in the gaps."""
        input_fft = np.fft.rfft(input_vector, n=num_samples)
        whitened_fft = input_fft * inv_psd_values
        whitened_vector = np.fft.irfft(whitened_fft, n=num_samples)
        return whitened_vector[missing_indices]

    # 3. Define matrix-free (C^-1)_MM product via FFT
    def matvec_K_MM(vector_missing_section: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        """Calculates (C^-1)_MM @ v_M in O(N log N) using FFT."""
        # Step A: Zero-pad vector v_M back to full data length num_samples
        full_vector_padded: npt.NDArray[np.float64] = np.zeros(num_samples, dtype=np.float64)
        full_vector_padded[missing_indices] = vector_missing_section
        
        # Step B: Whiten the padded vector and extract the missing indices
        return whiten_and_mask(full_vector_padded)

    # Wrap as SciPy LinearOperator
    logger.debug("Creating LinearOperator for (C^-1)_MM product.")
    covariance_inverse_operator: LinearOperator = LinearOperator(
        (num_missing_samples, num_missing_samples),
        matvec=matvec_K_MM,
        dtype=np.float64
    )

    # 4. Construct Right-Hand Side: RHS = -(C^-1)_MO @ x_O
    logger.debug("Constructing Right-Hand Side vector.")
    # Evaluate full C^-1 @ x_O where x_O has zeros in gaps
    observed_data_full_array: npt.NDArray[np.float64] = time_series_array.copy()
    observed_data_full_array[missing_indices] = 0.0  # Zero out gaps
    
    right_hand_side_vector: npt.NDArray[np.float64] = -whiten_and_mask(observed_data_full_array)
    logger.debug("Finished constructing Right-Hand Side vector.")

    # 5. Solve (C^-1)_MM @ x_M_hat = RHS using Conjugate Gradient
    logger.debug("Starting Conjugate Gradient solver.")
    # Preconditioner: The diagonal of K = C^-1 is constant for a stationary
    # process. K_ii = integral(1/S(f)) df.
    delta_f = psd.delta_f
    diag_K = np.sum(inv_psd_values) * delta_f
    
    if np.isclose(diag_K, 0):
        preconditioner_inverse = None
        logger.warning("Could not compute a valid diagonal for the preconditioner.")
    else:
        preconditioner_inverse = LinearOperator(
            (num_missing_samples, num_missing_samples),
            matvec=lambda v: v / diag_K
        )
    
    estimated_missing_data_vector: npt.NDArray[np.float64]
    cg_convergence_info: int
    estimated_missing_data_vector, cg_convergence_info = cg(
        covariance_inverse_operator, right_hand_side_vector, M=preconditioner_inverse, maxiter=1000, rtol=1e-12
    )
    
    if cg_convergence_info > 0:
        logger.warning(f"CG did not fully converge after {cg_convergence_info} iterations.")
    elif cg_convergence_info < 0:
        logger.error("CG failed with error code: %d", cg_convergence_info)
    else:
        logger.debug("CG solver completed.")

    # 6. Assemble complete inpainted output
    logger.debug("Assembling final inpainted data.")
    final_inpainted_data: npt.NDArray[np.float64] = time_series_array.copy()
    final_inpainted_data[missing_indices] = estimated_missing_data_vector
    
    logger.debug("gate_and_paint function finished. Returning gwaith.types.TimeSeries.")
    return TimeSeries(final_inpainted_data, epoch=input_data.epoch, delta_t=input_data.delta_t)