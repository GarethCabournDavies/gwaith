#!/bin/bash
#SBATCH --job-name="mf_@@CHANNEL@@_chunk@@CHUNK_IDX@@"
#SBATCH --output="@@CHUNK_OUTPUT_DIR@@/logs/mf_@@CHANNEL@@_%a_%j.out"
#SBATCH --error="@@CHUNK_OUTPUT_DIR@@/logs/mf_@@CHANNEL@@_%a_%j.err"
#SBATCH --array=0-@@TEMPLATE_PARALLELIZATION_MINUS_1@@

run_task() {
    local task_id=$1
    echo "Running task ${task_id} for channel @@CHANNEL@@ in chunk @@CHUNK_IDX@@"
    "@@TEMPLATE_SCRIPT_PATH@@" \
        "@@CHUNK_OUTPUT_DIR@@" "@@CHANNEL@@" "${task_id}" "@@TEMPLATE_PARALLELIZATION@@"
}

if [ -n "${SLURM_ARRAY_TASK_ID}" ]; then
    # Slurm mode: SLURM_ARRAY_TASK_ID is set, so run a single task.
    run_task "${SLURM_ARRAY_TASK_ID}"
else
    # Bash mode: SLURM_ARRAY_TASK_ID is not set.
    echo "Running all @@TEMPLATE_PARALLELIZATION@@ tasks in parallel (bash mode)..."
    for i in $(seq 0 @@TEMPLATE_PARALLELIZATION_MINUS_1@@); do
        run_task "$i" &
    done
    wait
    echo "All bash tasks for channel @@CHANNEL@@ complete."
fi