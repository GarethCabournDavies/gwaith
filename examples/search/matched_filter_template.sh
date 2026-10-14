#!/bin/bash
# This script runs a single matched filter task.
# It invokes the matched-filter command directly.

script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)

CHUNK_OUTPUT_DIR="$1"
CHANNEL="$2"
TASK_ID="$3"
TEMPLATE_PARALLELIZATION="$4"
WAVEFORM_TYPE="${5:-fastbhb}"
TDI_GENERATION="${6:-1.5}"
BANK_FILE="${7:-../template_banks/output/sangria_bank.hdf}"
PSD_FILE_A="${8:-../psd_estimation/output/A_sangria_hm_SMOOTHED_PSD.txt}"
PSD_FILE_E="${9:-../psd_estimation/output/E_sangria_hm_SMOOTHED_PSD.txt}"
INPUT_DATA_FILE="${10:-${CHUNK_OUTPUT_DIR}/data_processing/LDC2_sangria_hm_training_padded.hdf}"

echo "Running matched filter for chunk ${CHUNK_OUTPUT_DIR}, channel ${CHANNEL}, task ${TASK_ID} (waveform: ${WAVEFORM_TYPE}, TDI gen: ${TDI_GENERATION})"

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
  --start-time "${11:-0}" \
  --end-time "${12:-31536000}" \
  --verbose \
  --bank-file "${BANK_FILE}" \
  --psd-file "${CURRENT_PSD_FILE}" \
  --input-data-file "${INPUT_DATA_FILE}" \
  --channel "${CHANNEL}" \
  --output-file "${OUTPUT_FILE}" \
  --template-fraction-range "${TASK_ID}/${TEMPLATE_PARALLELIZATION}" \
  --waveform-type "${WAVEFORM_TYPE}" \
  --tdi-generation "${TDI_GENERATION}"
