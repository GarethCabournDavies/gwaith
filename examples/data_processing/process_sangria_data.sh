set -e

mkdir -p output/sangria

# Step 1: Load Sangria data, convert to AET channels (optionally downsample)
hwilas_process_sangria_data \
  --verbose \
  --data-file \
    ../datasets/output/LDC2_sangria_hm_training.hdf \
  --output-file \
    output/sangria/sangria_data_aet.hdf \
  --channels \
    LISA_A LISA_E LISA_T