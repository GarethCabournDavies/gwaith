hwilas_estimate_sangria_psd \
  --data-file ../LDC2_sangria_hm_training.hdf \
  --output-file \
    LISA_A:output/A_sangria_hm_SMOOTHED_PSD.txt \
    LISA_E:output/E_sangria_hm_SMOOTHED_PSD.txt \
    LISA_T:output/T_sangria_hm_SMOOTHED_PSD.txt \
  --smooth-regions \
    0.03,0.09 \
  --testing-plots output/ \
  --verbose