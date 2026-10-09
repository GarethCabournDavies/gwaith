#!/usr/bin/env bash
set -euo pipefail

mkdir -p output

hwilas_make_stochastic_bank \
    --config mojito_bank.toml \
    --nproc 4 \
    --seed 1 \
    --verbose

hwilas_plot_bank \
    --bank-file output/mojito_bank.hdf \
    --output-plot output/mojito_bank.png \
    --verbose
