# Estimate PSD from processed Sangria data
# NOTE: This assumes you have first processed the raw Sangria data using
# hwilas_process_sangria_data or similar to generate a processed HDF5 file
# with root-level LISA_A/E/T channels and epoch/delta_t attributes.

SANGRIA_PROCESSED_DATA=output/sangria_data_aet.hdf

mkdir -p output

hwilas_estimate_sangria_psd \
  --data-file "$SANGRIA_PROCESSED_DATA" \
  --output-file \
    LISA_A:output/A_sangria_SMOOTHED_PSD.txt \
    LISA_E:output/E_sangria_SMOOTHED_PSD.txt \
    LISA_T:output/T_sangria_SMOOTHED_PSD.txt \
  --segment-duration 1576800 \
  --smooth-regions 0.03,0.09 \
  --testing-plots output/ \
  --verbose