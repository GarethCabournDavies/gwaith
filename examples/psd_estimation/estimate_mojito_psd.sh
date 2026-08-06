

hwilas_estimate_mojito_psd \
  --data-file \
    /path/to/data.h5 \
  --psd-files \
    LISA_A:output/A_mojito_SMOOTHED_PSD.txt \
    LISA_E:output/E_mojito_SMOOTHED_PSD.txt \
    LISA_T:output/T_mojito_SMOOTHED_PSD.txt \
  --downsample-delta-t 4 \
  --smooth-regions \
    0.03,0.09 \
  --output-plots output \
  --verbose