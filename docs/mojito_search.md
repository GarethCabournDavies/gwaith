# Mojito search workflow

Run `examples/search/mojito_basic_search.sh` with bash from the repository checkout. This script executes a joint A/E matched-filtering search on preprocessed data.

## Requirements

*   **Software**: Installed `hwilas`.
*   **Data Inputs**: Preprocessed A/E/T HDF5 files, template bank, and smoothed PSDs.
*   **Diagnostics**: Mojito catalogue required for truth plots and diagnostics (not used for filtering).

## Environment Configuration
Override defaults using these variables:

*   `DATA_FILE`: Input data path.
*   `TRUTH_SOURCE`: Catalogue for validation.
*   `BANK_FILE`: Template bank location.
*   `PSD_DIR`: Directory for smoothed PSDs.
*   `OUTPUT_DIR`: Default is `examples/search/output/mojito`.
*   `N_PROCESSES`: Parallel workers (default 4).
*   `DATA_TIME_ORIGIN`: Required absolute timestamp of the first input sample; obtain it from the input metadata.

## Search Workflow

1.  **Modeling**: Uses `fastbhb` with analytic orbits by default.
2.  **Filtering**: Joint A/E matched filtering partitioned across the template bank.
3.  **Clustering**: Network clustering of candidates.
4.  **Refinement**: Physical waveform fitting and subtraction using `BBHx` with ESA orbits (configurable via `SEARCH_APPROXIMANT`, `SEARCH_ORBITS`, `OPTIMISATION_MODEL`, etc.).
5.  **Residuals**: Second search on residuals.
6.  **Output**: Final catalogue, PE exports, candidate CSVs, plots, and residual diagnostics.

## Preprocessing & Validation
Raw data preprocessing is handled by `examples/data_processing/prepare_mojito_hpc.sh` (accepting `MOJITO_RAW`, `MOJITO_CATALOGUE`, etc.), which generates 4-second A/E/T data and median PSDs via `hwilas_prepare_mojito`. These products are distinct from the specific PSD/bank inputs required for search matching.

**Input checks**:

*   Ensure matched sampling, units, channel names, and waveform/orbit conventions align between data, bank, and search configuration.
*   A detected event does not guarantee clean subtraction.
*   Validate results using noiseless injection data when available.

## Minimal Invocation Example

```bash
DATA_FILE=/path/to/data.h5 \
TRUTH_SOURCE=/path/to/catalogue.hdf5 \
BANK_FILE=/path/to/bank.hdf5 \
PSD_DIR=/path/to/psds \
OUTPUT_DIR=/path/to/output \
DATA_TIME_ORIGIN=12345678.0 \
N_PROCESSES=4 \
bash examples/search/mojito_basic_search.sh
```

The timestamp in the example is synthetic; replace it with the input data origin.

## References

*   [Mojito Joint Search](mojito_joint_search.md)
*   [MBHBS PE Export](mbhbs_pe_export.md)
*   [Residual Statistics](residual_statistics.md)
