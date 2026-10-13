#!/usr/bin/env bash
set -euo pipefail

# Run with the updated hwilas package installed in the active HPC environment.
mojito_clean_dir="${1:-/path/to/source_slices}"
mojito_catalogue_file="${2:-/path/to/catalogue.hdf5}"
OUTPUT_DIR="${3:-output/mojito_clean}"
total_data_duration=$((731 * 86400))
chunk_duration=$((122 * 86400))
before_merger_days=40
after_merger_days=20
catalogue_time_offset=800

mkdir -p "${OUTPUT_DIR}"
windows_file=$(mktemp)
trap 'rm -f "$windows_file"' EXIT

# Catalogue merger times are shifted onto the data axis before selecting chunks.
hwilas_mojito_clean_windows \
    --data-directory "$mojito_clean_dir" \
    --catalogue-file "$mojito_catalogue_file" \
    --output-directory "${OUTPUT_DIR}/data_processing" \
    --start-time 0 --end-time "$total_data_duration" \
    --chunk-duration "$chunk_duration" \
    --before-merger-days "$before_merger_days" \
    --after-merger-days "$after_merger_days" \
    --catalogue-time-offset "$catalogue_time_offset" > "$windows_file"

downsampled_files=()
while IFS=$'\t' read -r input_file downsampled_file start_time end_time; do
    echo "Downsampling ${input_file}: ${start_time} to ${end_time} -> ${downsampled_file}"
    hwilas_process_mojito_data_chunk \
        --data-file "$input_file" \
        --output-file "$downsampled_file" \
        --low-cutoff-frequency 1e-5 \
        --channels LISA_A LISA_E LISA_T \
        --delta-t 4 \
        --analysis-start-time "$start_time" \
        --analysis-end-time "$end_time" \
        --verbose
    downsampled_files+=("$downsampled_file")
done < "$windows_file"

if [ ${#downsampled_files[@]} -eq 0 ]; then
    echo "No clean source chunks were selected" >&2
    exit 1
fi

# Add each segment into its timestamps; uncovered intervals remain zero.
summed_file="${OUTPUT_DIR}/mojito_clean_summed.hdf"
hwilas_process_data_combine \
    --data-file "${downsampled_files[@]}" \
    --output-file "$summed_file" \
    --channels LISA_A LISA_E LISA_T \
    --start-time 0 --end-time "$total_data_duration" --delta-t 4 \
    --verbose
echo "Clean summed data: ${summed_file}"
