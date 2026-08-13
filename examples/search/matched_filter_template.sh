#!/bin/bash
# This script runs a single matched filter task.
# It is designed to be called by sangria_basic_search.sh either directly (bash mode)
# or via sbatch (slurm mode).

CHUNK_OUTPUT_DIR="$1"
CHANNEL="$2"
TASK_ID="$3"
TEMPLATE_PARALLELIZATION="$4"
PSD_FILE_A="../psd_estimation/output/A_sangria_hm_SMOOTHED_PSD.txt"
PSD_FILE_E="../psd_estimation/output/E_sangria_hm_SMOOTHED_PSD.txt"
BANK_FILE="../template_banks/output/sangria_bank.hdf"
INPUT_DATA_FILE="${CHUNK_OUTPUT_DIR}/data_processing/LDC2_sangria_hm_training_padded.hdf"

echo "Running matched filter for chunk ${CHUNK_OUTPUT_DIR}, channel ${CHANNEL}, task ${TASK_ID}"

OUTPUT_FILE="${CHUNK_OUTPUT_DIR}/matched_filter/matched_filter_snr_peaks_${CHANNEL}_${TASK_ID}.hdf"

# Determine which PSD file to use based on the channel
if [ "$CHANNEL" == "LISA_A" ]; then
  CURRENT_PSD_FILE="${PSD_FILE_A}"
elif [ "$CHANNEL" == "LISA_E" ]; then
  CURRENT_PSD_FILE="${PSD_FILE_E}"
else
  echo "Error: Unknown channel ${CHANNEL}"
  exit 1
fi

hwilas_matched_filter_search \
  --verbose \
  --bank-file "${BANK_FILE}" \
  --psd-file "${CURRENT_PSD_FILE}" \
  --input-data-file "${INPUT_DATA_FILE}" \
  --channel "${CHANNEL}" \
  --output-file "${OUTPUT_FILE}" \
  --template-fraction-range "${TASK_ID}/${TEMPLATE_PARALLELIZATION}" \
  --testing-plots "${CHUNK_OUTPUT_DIR}/matched_filter"
