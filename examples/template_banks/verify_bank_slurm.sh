#!/bin/bash
set -euo pipefail

RUN_MODE="slurm" # Options: "slurm" or "bash"

# Run from examples/template_banks. Edit these paths for another bank.
CONFIG_FILE="mojito_bank.toml"
BANK_FILE="output/mojito_bank.hdf"
OUTPUT_DIR="output/mojito_bank_verification"

number_of_injections=200000
injection_batches=200
workers_per_batch=4
max_parallel_batches=20

# Optional command-line paths: CONFIG BANK OUTPUT_DIR.
CONFIG_FILE="${1:-$CONFIG_FILE}"
BANK_FILE="${2:-$BANK_FILE}"
OUTPUT_DIR="${3:-$OUTPUT_DIR}"

if [ "$RUN_MODE" != "slurm" ] && [ "$RUN_MODE" != "bash" ]; then
    echo 'RUN_MODE must be "slurm" or "bash".' >&2
    exit 1
fi

# Step 1: Generate the bank waveforms once for all injection batches.
echo "Preparing the bank waveform cache..."
mkdir -p "${OUTPUT_DIR}/logs"
hwilas_verify_bank prepare \
    --config "${CONFIG_FILE}" \
    --bank-file "${BANK_FILE}" \
    --cache-dir "${OUTPUT_DIR}/cache"

# Step 2: Recover injections in independent batches.
echo "Running ${number_of_injections} injections in ${injection_batches} batches..."
if [ "$RUN_MODE" == "slurm" ]; then
    echo "Batch logs: ${OUTPUT_DIR}/logs/injections-JOB_ID_TASK_ID.log"
    echo "Slurm prints the job ID below. Monitor: squeue -j JOB_ID; stop: scancel JOB_ID"
    echo "Waiting for all injection jobs to finish before collating and plotting..."

    sbatch --wait \
        --job-name=bank-injections \
        --array="0-$((injection_batches - 1))%${max_parallel_batches}" \
        --cpus-per-task="${workers_per_batch}" \
        --output="${OUTPUT_DIR}/logs/injections-%A_%a.log" \
        "$(dirname "$0")/verify_bank_injections.sbatch" \
        "${OUTPUT_DIR}" "${number_of_injections}" "${injection_batches}" "${workers_per_batch}"
else
    # Bash mode runs one batch at a time, using its configured Python workers.
    for ((batch_idx=0; batch_idx<injection_batches; batch_idx++)); do
        echo "Recovering injection batch ${batch_idx}..."
        hwilas_verify_bank run \
            --cache-dir "${OUTPUT_DIR}/cache" \
            --output-file "${OUTPUT_DIR}/shard-${batch_idx}.json" \
            --n "${number_of_injections}" \
            --shard-count "${injection_batches}" \
            --shard-index "${batch_idx}" \
            --nproc "${workers_per_batch}"
    done
fi

# Step 3: Collate the completed batches and check that every injection is present.
echo "Collating injection results..."
hwilas_verify_bank merge \
    --output-file "${OUTPUT_DIR}/coverage.json" \
    "${OUTPUT_DIR}"/shard-*.json

# Step 4: Plot match histograms for the complete campaign and individual batches.
echo "Plotting bank verification results..."
hwilas_plot_bank_verification \
    --output-dir "${OUTPUT_DIR}/plots" \
    "${OUTPUT_DIR}/coverage.json" \
    "${OUTPUT_DIR}"/shard-*.json

echo "Verification complete. Reports: ${OUTPUT_DIR}; plots: ${OUTPUT_DIR}/plots"
