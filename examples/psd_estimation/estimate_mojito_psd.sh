
# Estimate PSD from processed Mojito data
# This uses the output of hwilas_process_mojito_data which provides root-level
# LISA_A/E/T channels with epoch/delta_t attributes.
#
# Multiple chunk files can be given; each channel is concatenated across all
# chunks before Welch's method is applied, giving a much smoother estimate.

MOJITO_PROCESSED_DATA="../data_processing/mojito_all_downsampled.hdf"

mkdir -p output/mojito/

hwilas_estimate_psd \
  --data-file "${MOJITO_PROCESSED_DATA[@]}" \
  --output-file \
    LISA_A:output/mojito/A_mojito_SMOOTHED_PSD.txt \
    LISA_E:output/mojito/E_mojito_SMOOTHED_PSD.txt \
    LISA_T:output/mojito/T_mojito_SMOOTHED_PSD.txt \
  --unsmoothed-output-file \
    LISA_A:output/mojito/A_mojito_UNSMOOTHED_PSD.txt \
    LISA_E:output/mojito/E_mojito_UNSMOOTHED_PSD.txt \
    LISA_T:output/mojito/T_mojito_UNSMOOTHED_PSD.txt \
  --segment-duration 1576800 \
  --smooth-regions 0.02,0.04 0.05,0.07 0.08,0.1 \
  --verbose

# Plot the analysis PSDs alongside all three unsmoothed channels.
hwilas_plot_psd \
  --verbose \
  --psd-files \
    "T unsmoothed:output/mojito/T_mojito_UNSMOOTHED_PSD.txt" \
    "A unsmoothed:output/mojito/A_mojito_UNSMOOTHED_PSD.txt" \
    "E unsmoothed:output/mojito/E_mojito_UNSMOOTHED_PSD.txt" \
    "T smoothed:output/mojito/T_mojito_SMOOTHED_PSD.txt" \
    "A smoothed:output/mojito/A_mojito_SMOOTHED_PSD.txt" \
    "E smoothed:output/mojito/E_mojito_SMOOTHED_PSD.txt" \
  --f-min 1e-5 --f-max 0.1 \
  --y-min 1e-47 --y-max 1e-36 \
  --output-file output/mojito/mojito_psd.png
