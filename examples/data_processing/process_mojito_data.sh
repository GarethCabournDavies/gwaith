#!/bin/bash
set -e

# All search outputs for this workflow go under output/mojito.
OUTPUT_DIR="output/mojito"

# Mojito dataset (731 days of light data at 4 s cadence after downsampling).
mojito_data_file="/path/to/data.h5"
mojito_catalogue_file="/path/to/catalogue.hdf5"

# Total data duration: 731 days in seconds.
total_data_duration=$((731 * 86400))
chunk_duration=$((122 * 86400))      # just over 4 months in seconds

current_analysis_start_time=0
chunk_idx=0
downsampled_files=()

# =============================================================================
# Step 1 (per chunk): Downsample the raw Mojito data to AET channels.
# One downsampled HDF5 per chunk; all of them are pooled together in Step 2
# for a smoother PSD estimate.
# =============================================================================
mkdir -p "${OUTPUT_DIR}"

while [ $current_analysis_start_time -lt $total_data_duration ]; do
    current_analysis_end_time=$((current_analysis_start_time + $chunk_duration))
    if [ $current_analysis_end_time -gt $total_data_duration ]; then
        current_analysis_end_time=$total_data_duration
    fi

    downsampled_file=${OUTPUT_DIR}/chunk_${chunk_idx}.hdf

    hwilas_process_mojito_data_chunk \
        --data-file "${mojito_data_file}" \
        --output-file "${downsampled_file}" \
        --low-cutoff-frequency 1e-5 \
        --channels LISA_A LISA_E LISA_T \
        --delta-t 4 \
        --analysis-start-time "${current_analysis_start_time}" \
        --analysis-end-time "${current_analysis_end_time}" \
        --verbose

    downsampled_files+=("${downsampled_file}")

    current_analysis_start_time=$current_analysis_end_time
    chunk_idx=$((chunk_idx + 1))
done

recombined_file=${OUTPUT_DIR}/mojito_all_downsampled.hdf

hwilas_process_data_combine \
    --data-file "${downsampled_files[@]}" \
    --output-file "${recombined_file}" \
    --channels LISA_A LISA_E LISA_T \
    --delta-t 4 \
    --verbose
