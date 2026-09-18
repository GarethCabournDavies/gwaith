# Example MOJITO waveform plotting workflow.
#
# 1) Extract MBHB truth metadata from the MOJITO catalogue.
# 2) Convert MOJITO L1 data to AET channels (scaled by laser frequency by
#    default to match Sangria-like strain normalization).
# 3) Plot noisy vs regenerated waveforms with FastBHB (default).

MOJITO_DATA_FILE=/path/to/data.h5
MOJITO_CATALOG_FILE=/path/to/catalogue.hdf5

mkdir -p plots/mojito

# hwilas_extract_truth \
#   --data-file "$MOJITO_CATALOG_FILE" \
#   --type mojito \
#   --verbose \
#   --output-file plots/mojito/truth_mbhb.h5

# hwilas_process_mojito_data \
#   --verbose \
#   --data-file "$MOJITO_DATA_FILE" \
#   --output-file plots/mojito/mojito_data_aet.hdf \
#   --channels LISA_A LISA_E LISA_T \
#   --delta-t 4

hwilas_plot_mojito_waveforms \
  --verbose \
  --truth-file plots/mojito/truth_mbhb.h5 \
  --input-file plots/mojito/mojito_data_aet.hdf \
  --output-dir plots/mojito \
  --channels LISA_A LISA_E \
  --waveform FastBHB \
  --orbit oem \
  --tdi-generation 1.5 \
  --window-before-days 0.05 \
  --window-after-days 0.025 