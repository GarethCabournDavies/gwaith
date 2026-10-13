#!/bin/bash
set -euo pipefail

OUTPUT_DIR=$1
number_of_injections=$2
injection_batches=$3
workers_per_batch=$4
batch_idx=$SLURM_ARRAY_TASK_ID

# Each Python worker uses one CPU for numerical libraries.
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1

echo "Recovering injection batch ${batch_idx}..."
hwilas_verify_bank run \
    --cache-dir "${OUTPUT_DIR}/cache" \
    --output-file "${OUTPUT_DIR}/shard-${batch_idx}.json" \
    --n "${number_of_injections}" \
    --shard-count "${injection_batches}" \
    --shard-index "${batch_idx}" \
    --nproc "${workers_per_batch}"
