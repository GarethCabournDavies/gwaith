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
        return time_series_array.copy()

    # 2. Extract Autocorrelation Function (ACF) from PSD via Inverse FFT
    logger.debug("Extracting Autocorrelation Function (ACF) from PSD.")
    psd_values: np.ndarray = psd.data
    psd_frequencies: np.ndarray = psd.sample_frequencies.data
    logger.debug("Using gwaith.FrequencySeries for PSD input.")

    fft_frequencies: np.ndarray = np.fft.rfftfreq(num_samples, d=1.0 / sampling_rate_hz)
    
    # We assume the input PSD is already correctly aligned or will be handled externally
    # If the user wishes to interpolate, it should be done before calling this function
    interpolated_psd: np.ndarray = psd_values


    # Wiener-Khinchin: ACF is the real IFFT of single-sided PSD
    logger.debug("Calculating Autocorrelation function (ACF).")
    autocorrelation_function: npt.NDArray[np.float64] = np.fft.irfft(interpolated_psd, n=num_samples) * (sampling_rate_hz / 2.0)

    # 3. Define matrix-free C_MM product via FFT
    # Embed acf into a symmetric circulant kernel for fast 1D convolution
    logger.debug("Building autocorrelation kernel for FFT convolution.")
    autocorrelation_kernel: npt.NDArray[np.float64] = np.concatenate(
        [autocorrelation_function, autocorrelation_function[1:-1][::-1]]
    )

    def matvec_C_MM(vector_missing_section: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        """Calculates C_MM @ v_M in O(N log N) using FFT convolution."""
        # logger.debug("Inside matvec_C_MM: vector_missing_section shape %s", vector_missing_section.shape)
        # Step A: Zero-pad vector v_M back to full data length num_samples
        full_vector_padded: npt.NDArray[np.float64] = np.zeros(num_samples, dtype=np.float64)
        full_vector_padded[missing_indices] = vector_missing_section
        
        # Step B: Fast Toeplitz matrix-vector multiply using FFT
        convolution_result: npt.NDArray[np.float64] = fftconvolve(
            full_vector_padded, autocorrelation_function, mode='full'
        )[:num_samples]
        
        # Step C: Extract only the missing gap indices
        return convolution_result[missing_indices]

    # Wrap as SciPy LinearOperator
    logger.debug("Creating LinearOperator for C_MM product.")
    covariance_matrix_operator: LinearOperator = LinearOperator(
        (num_missing_samples, num_missing_samples),
        matvec=matvec_C_MM,
        dtype=np.float64
    )

    # 4. Construct Right-Hand Side: RHS = -C_MO @ x_O
    logger.debug("Constructing Right-Hand Side vector.")
    # Evaluate full C @ x_O where x_O has zeros in gaps
    observed_data_full_array: npt.NDArray[np.float64] = time_series_array.copy()
    observed_data_full_array[missing_indices] = 0.0  # Zero out gaps
    
    full_covariance_observed_product: npt.NDArray[np.float64] = fftconvolve(
        observed_data_full_array, autocorrelation_function, mode='full'
    )[:num_samples]
    right_hand_side_vector: npt.NDArray[np.float64] = -full_covariance_observed_product[missing_indices]
    logger.debug("Finished constructing Right-Hand Side vector.")

    # 5. Solve C_MM @ x_M_hat = RHS using Conjugate Gradient
    logger.debug("Starting Conjugate Gradient solver.")
    # Diagonal Jacobi Preconditioner (ACF[0] is variance along main diagonal)
    preconditioner_inverse: LinearOperator = LinearOperator(
        (num_missing_samples, num_missing_samples),
        matvec=lambda v: v / autocorrelation_function[0]
    )
    
    estimated_missing_data_vector: npt.NDArray[np.float64]
    cg_convergence_info: int
    estimated_missing_data_vector, cg_convergence_info = cg(
        covariance_matrix_operator, right_hand_side_vector, M=preconditioner_inverse, maxiter=500
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