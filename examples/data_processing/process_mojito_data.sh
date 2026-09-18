# Step 1: Load and downsample Mojito data to AET channels
hwilas_process_mojito_data \
  --verbose \
  --data-file \
    /path/to/data.h5 \
  --output-file \
    mojito_data_aet.hdf \
  --channels \
    LISA_A LISA_E LISA_T \
  --delta-t 4

# Step 2: Split to analysis window, pad, and inpaint (common for both sangria and mojito)
hwilas_split_pad_data \
  --verbose \
  --data-file \
    mojito_data_aet.hdf \
  --output-file \
    mojito_data_processed.hdf \
  --channels \
    LISA_A LISA_E \
  --psd-file \
    ../psd_estimation/output/A_mojito_SMOOTHED_PSD.txt \
  --padding-duration \
    $((30 * 86400))