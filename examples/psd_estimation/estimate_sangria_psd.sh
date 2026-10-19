# Estimate PSD from processed Sangria data
# NOTE: This assumes you have first processed the raw Sangria data using
# hwilas_process_sangria_data or similar to generate a processed HDF5 file
# with root-level LISA_A/E/T channels and epoch/delta_t attributes.
set -e

SANGRIA_PROCESSED_DATA=../data_processing/output/sangria/sangria_data_aet.hdf

mkdir -p output/sangria

hwilas_estimate_psd \
  --data-file "$SANGRIA_PROCESSED_DATA" \
  --output-file \
    LISA_A:output/sangria/A_sangria_SMOOTHED_PSD.txt \
    LISA_E:output/sangria/E_sangria_SMOOTHED_PSD.txt \
    LISA_T:output/sangria/T_sangria_SMOOTHED_PSD.txt \
  --unsmoothed-output-file \
    LISA_A:output/sangria/A_sangria_UNSMOOTHED_PSD.txt \
    LISA_E:output/sangria/E_sangria_UNSMOOTHED_PSD.txt \
    LISA_T:output/sangria/T_sangria_UNSMOOTHED_PSD.txt \
  --segment-duration 1576800 \
  --smooth-regions 0.03,0.09 \
  --verbose

# Plot the analysis PSDs alongside all three unsmoothed channels.
hwilas_plot_psd \
  --verbose \
  --psd-files \
    "T unsmoothed:output/sangria/T_sangria_UNSMOOTHED_PSD.txt" \
    "A unsmoothed:output/sangria/A_sangria_UNSMOOTHED_PSD.txt" \
    "E unsmoothed:output/sangria/E_sangria_UNSMOOTHED_PSD.txt" \
    "T smoothed:output/sangria/T_sangria_SMOOTHED_PSD.txt" \
    "A smoothed:output/sangria/A_sangria_SMOOTHED_PSD.txt" \
    "E smoothed:output/sangria/E_sangria_SMOOTHED_PSD.txt" \
  --output-file output/sangria/sangria_psd.png
