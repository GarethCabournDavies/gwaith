#!/usr/bin/env bash
set -euo pipefail

mkdir -p output

hwilas_make_stochastic_bank \
    --config sangria_bank.toml \
    --nproc 4 \
    --seed 1 \
    --verbose

hwilas_plot_bank \
  --bank-file output/sangria_bank.hdf \
  --output-plot output/sangria_bank.png \
  --signal-definition-file ../datasets/LDC2_sangria_hm_training.hdf
