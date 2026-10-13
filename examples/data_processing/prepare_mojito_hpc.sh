#!/usr/bin/env bash
# Run directly with bash after activating the HPC Python environment.
set -euo pipefail
: "${HWILAS_REPO:?Set HWILAS_REPO to the HPC checkout containing these changes}"
: "${MOJITO_RAW:?Set MOJITO_RAW to the raw 0.25 s L1 HDF5 file}"
: "${MOJITO_CATALOGUE:?Set MOJITO_CATALOGUE to the catalogue HDF5 file}"
: "${MOJITO_TRANSFER:?Set MOJITO_TRANSFER to a NEW output directory}"
cd "$HWILAS_REPO"
# Activate env_hwilas_search before running (or supply its Python explicitly).
export MPLCONFIGDIR=${MPLCONFIGDIR:-${TMPDIR:-/tmp}/hwilas-matplotlib}
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
printf 'host=%s cwd=%s pid=%s output=%s\n' "$(hostname)" "$PWD" "$$" "$MOJITO_TRANSFER"
printf 'command: %s -m tools.data_processing.prepare_mojito --raw %s --catalogue %s --output-dir %s\n' "${PYTHON:-python}" "$MOJITO_RAW" "$MOJITO_CATALOGUE" "$MOJITO_TRANSFER"
"${PYTHON:-python}" -u -m tools.data_processing.prepare_mojito \
    --raw "$MOJITO_RAW" --catalogue "$MOJITO_CATALOGUE" --output-dir "$MOJITO_TRANSFER"
printf 'Complete. Copy this directory and verify SHA256SUMS on the laptop.\n'
