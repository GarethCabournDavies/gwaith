hwilas_extract_truth \
    --data-file datasets/LDC2_sangria_hm_training.hdf \
    --source-group mbhb \
    --output-file plots/sangria/truth_mbhb.h5

hwilas_plot_sangria_waveforms \
    --verbose \
    --truth-file plots/sangria/truth_mbhb.h5 \
    --input-file datasets/LDC2_sangria_hm_training.hdf \
    --output-dir plots/sangria \
    --channels LISA_A LISA_E \
    --waveform FastBHB \
    --orbit analytic \
    --tdi-generation 1.5 \
    --window-before-days 0.05 \
    --window-after-days 0.025