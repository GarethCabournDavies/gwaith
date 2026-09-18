
# Estimate PSD from processed Mojito data
# This uses the output of hwilas_process_mojito_data which provides root-level
# LISA_A/E/T channels with epoch/delta_t attributes.

MOJITO_PROCESSED_DATA=../plots/mojito/mojito_data_aet.hdf

mkdir -p output

hwilas_estimate_mojito_psd \
  --data-file "$MOJITO_PROCESSED_DATA" \
  --output-file \
    LISA_A:output/A_mojito_SMOOTHED_PSD.hdf \
    LISA_E:output/E_mojito_SMOOTHED_PSD.hdf \
    LISA_T:output/T_mojito_SMOOTHED_PSD.hdf \
  --segment-duration 1576800 \
  --smooth-regions 0.03,0.09 \
  --testing-plots output/ \
  --verbose