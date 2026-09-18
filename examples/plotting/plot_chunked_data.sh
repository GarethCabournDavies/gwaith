#!/bin/bash
set -e


# ---------------------------------------------------------------------------
# Mojito
# ---------------------------------------------------------------------------
MOJITO_OUTPUT_DIR="../search/output/mojito/round_0"

data_files=$(ls ${MOJITO_OUTPUT_DIR}/chunk_*/data_processing/*.hdf)

echo $data_files

psd_a=../psd_estimation/output/A_mojito_SMOOTHED_PSD.txt
psd_e=../psd_estimation/output/E_mojito_SMOOTHED_PSD.txt

hwilas_plot_chunked_data \
    --data-files $data_files \
    --psd-file "${psd_a}" \
    --channel LISA_A \
    --output output/mojito_LISA_A.png \
    --verbose

hwilas_plot_chunked_data \
    --data-files $data_files \
    --psd-file "${psd_e}" \
    --channel LISA_E \
    --output output/mojito_LISA_E.png \
    --verbose


# ---------------------------------------------------------------------------
# Sangria
# ---------------------------------------------------------------------------
SANGRIA_OUTPUT_DIR="../search/output/sangria/round_0"

psd_a=../psd_estimation/output/A_sangria_hm_SMOOTHED_PSD.txt
psd_e=../psd_estimation/output/E_sangria_hm_SMOOTHED_PSD.txt

s_data_files=$(ls ${SANGRIA_OUTPUT_DIR}/chunk_*/data_processing/*.hdf)

hwilas_plot_chunked_data \
    --data-files $s_data_files \
    --psd-file "${psd_a}" \
    --channel LISA_A \
    --output output/sangria_LISA_A.png \
    --verbose

hwilas_plot_chunked_data \
    --data-files $s_data_files \
    --psd-file "${psd_e}" \
    --channel LISA_E \
    --output output/sangria_LISA_E.png \
    --verbose


echo "Done! Plots written under ${PLOTS_ROOT}/"
