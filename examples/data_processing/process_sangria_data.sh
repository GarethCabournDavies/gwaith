# Step 1: Load Sangria data, convert to AET channels (optionally downsample)
hwilas_process_sangria_data \
  --verbose \
  --data-file \
    ../datasets/LDC2_sangria_hm_training.hdf \
  --output-file \
    sangria_data_aet.hdf \
  --channels \
    LISA_A LISA_E LISA_T

# Step 2: Split to analysis window, pad, and inpaint (common for both sangria and mojito)
hwilas_split_pad_data \
  --verbose \
  --data-file \
    sangria_data_aet.hdf \
  --output-file \
    sangria_data_processed.hdf \
  --channels \
    LISA_A LISA_E \
  --psd-file \
    ../psd_estimation/output/A_sangria_hm_SMOOTHED_PSD.txt \
  --padding-duration \
    $((30 * 86400)) \
  --analysis-start-time 0 \
  --analysis-end-time $((122 * 86400))