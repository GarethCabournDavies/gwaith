#!/bin/bash
set -e

RUN_MODE="slurm" # Options: "slurm" or "bash"

# Ensure the matched filter template script is executable
chmod +x "$(dirname "$0")/matched_filter_template.sh"

template_parallelization=15

total_data_duration=$((365 * 86400)) # 1 year in seconds
chunk_duration=$((122 * 86400))      # just over 4 months in seconds
padding_for_process_data=$((30 * 86400)) # Padding for hwilas_process_sangria_data, e.g., 30 days

current_analysis_start_time=0
chunk_idx=0

matched_filter_job_ids=()

# This function creates a self-contained, executable script that can either be
# submitted to Slurm as a job array or run directly with bash to execute all
# tasks in parallel.
handle_matched_filter_jobs() {
    local chunk_output_dir=$1
    local channel=$2
    local chunk_idx=$3
    local template_parallelization=$4

    echo "Handling Matched Filter jobs for Channel ${channel} (Chunk ${chunk_idx})"
    local sbatch_script="${chunk_output_dir}/logs/mf_${channel}_chunk${chunk_idx}.sbatch"
    local sbatch_template_path
    sbatch_template_path="$(dirname "$0")/sbatch_mf_template.sh"
    # Use an absolute path for the template script to make the sbatch script portable
    local template_script_path
    template_script_path=$(readlink -f "$(dirname "$0")/matched_filter_template.sh")
    local template_parallelization_minus_1=$((template_parallelization - 1))

    # Create the executable sbatch script from the template
    sed -e "s|@@CHUNK_OUTPUT_DIR@@|${chunk_output_dir}|g" \
        -e "s|@@CHANNEL@@|${channel}|g" \
        -e "s|@@CHUNK_IDX@@|${chunk_idx}|g" \
        -e "s|@@TEMPLATE_PARALLELIZATION_MINUS_1@@|${template_parallelization_minus_1}|g" \
        -e "s|@@TEMPLATE_PARALLELIZATION@@|${template_parallelization}|g" \
        -e "s|@@TEMPLATE_SCRIPT_PATH@@|${template_script_path}|g" \
        "${sbatch_template_path}" > "${sbatch_script}"

    chmod +x "${sbatch_script}"

    if [ "$RUN_MODE" == "slurm" ]; then
        job_id=$(sbatch --parsable "${sbatch_script}")
        matched_filter_job_ids+=("$job_id")
    else # RUN_MODE == "bash"
        "${sbatch_script}"
    fi
}

while (( $(echo "$current_analysis_start_time < $total_data_duration" | bc -l) )); do
  analysis_end_time=$(echo "$current_analysis_start_time + $chunk_duration" | bc -l)
  # Ensure the last chunk doesn't go beyond the total data duration
  if (( $(echo "$analysis_end_time > $total_data_duration" | bc -l) )); then
    analysis_end_time=$total_data_duration
  fi

  echo "--- Processing Chunk ${chunk_idx} (Analysis Window: ${current_analysis_start_time}s to ${analysis_end_time}s) ---"

  chunk_output_dir="output/round_0/chunk_${chunk_idx}"
  mkdir -p "${chunk_output_dir}/data_processing"
  mkdir -p "${chunk_output_dir}/matched_filter"
  mkdir -p "${chunk_output_dir}/logs"

#   # 1. Process Sangria Data for this chunk
#   echo "Running hwilas_process_sangria_data for chunk ${chunk_idx}"
#   hwilas_process_sangria_data \
#     --data-file ../datasets/LDC2_sangria_hm_training.hdf \
#     --psd-file ../psd_estimation/output/A_sangria_hm_SMOOTHED_PSD.txt \
#     --output-file "${chunk_output_dir}/data_processing/LDC2_sangria_hm_training_padded.hdf" \
#     --padding-duration "${padding_for_process_data}" \
#     --analysis-start-time "${current_analysis_start_time}" \
#     --analysis-end-time "${analysis_end_time}" \
#     --verbose --verbose \
#     --testing-plots "${chunk_output_dir}/data_processing"

  handle_matched_filter_jobs "${chunk_output_dir}" "LISA_A" "${chunk_idx}" "${template_parallelization}"
  handle_matched_filter_jobs "${chunk_output_dir}" "LISA_E" "${chunk_idx}" "${template_parallelization}"

  # Update for next chunk
  current_analysis_start_time=$(echo "$current_analysis_start_time + $chunk_duration" | bc -l)
  chunk_idx=$((chunk_idx + 1))
done

if [ "$RUN_MODE" == "slurm" ]; then
  if [ ${#matched_filter_job_ids[@]} -eq 0 ]; then
    echo "No matched-filter jobs were submitted."
    exit 1
  fi

  joined_job_ids=$(IFS=,; echo "${matched_filter_job_ids[*]}") # Join all job IDs with commas
  echo "Monitoring $((${#matched_filter_job_ids[@]} * ${template_parallelization})) tasks across job arrays: ${joined_job_ids}"
  echo "Waiting for matched-filter Slurm jobs to finish: ${joined_job_ids}"
  while true; do
    active_jobs=0
    for job_id in "${matched_filter_job_ids[@]}"; do
      # squeue -h -j "$job_id" will list all tasks for that specific job array
      active_jobs=$((active_jobs + $(squeue -h -r -j "$job_id" | wc -l)))
    done
    if [ "${active_jobs}" -eq 0 ]; then
      break
    fi
    echo "${active_jobs} total matched-filter jobs (from all chunks and channels) still running/pending..."
    sleep 30
  done

  echo "All matched-filter Slurm jobs have finished."
fi

shopt -s nullglob
files_A=(output/round_0/chunk_*/matched_filter/matched_filter_snr_peaks_A_*.hdf)
files_E=(output/round_0/chunk_*/matched_filter/matched_filter_snr_peaks_E_*.hdf)
shopt -u nullglob

echo $files_A
echo $files_E


if [ ${#files_A[@]} -eq 0 ] || [ ${#files_E[@]} -eq 0 ]; then
  echo "Missing matched-filter output files after Slurm completion."
  echo "Check logs in output/round_0/logs"
  exit 1
fi

mkdir -p output/round_0/trigger_merge
mkdir -p output/round_0/trigger_merge
hwilas_trigger_merge \
  --verbose \
  --trigger-files "${files_A[@]}" \
  --output-file output/round_0/trigger_merge/trigger_merge_A.hdf \
  --cluster-window 2592000 \
  --testing-plot output/round_0/trigger_merge/trigger_merge_A.png 

hwilas_trigger_merge \
  --verbose \
  --trigger-files "${files_E[@]}" \
  --output-file output/round_0/trigger_merge/trigger_merge_E.hdf \
  --cluster-window 2592000 \
  --testing-plot output/round_0/trigger_merge/trigger_merge_E.png 

mkdir -p output/round_0/coincidence

hwilas_coincidence \
  --verbose \
  --trigger-merge-files \
    LISA_A:output/round_0/trigger_merge/trigger_merge_A.hdf \
    LISA_E:output/round_0/trigger_merge/trigger_merge_E.hdf \
  --output-file output/round_0/coincidence/coincidence.hdf \
  --coincidence-window 3600 \
  --testing-plots output/round_0/coincidence \
  --expected-snr-file optimal_snr/optimal_snr_fitting_factors.hdf

# The rest of the script (optimise, subtract, round_1) would follow a similar chunking logic
# if it were to be enabled. For now, I'll leave them commented out as they were.

# mkdir -p output/round_0/optimise

# hwilas_optimise_snr_peaks \
#   --input-file output/round_0/coincidence/coincidence.hdf \
#   --signal-output-file output/round_0/optimise/optimise.hdf \
#   --signal-parameter-file output/round_0/optimise/optimise_parameters.hdf \
#   --data-file output/data_processing/LDC2_sangria_hm_training_padded.hdf \
#   --testing-plots output/round_0/optimise \
#   --psd-file ../psd_estimation/output/A_sangria_hm_SMOOTHED_PSD.txt \
#   --bank-file ../template_banks/output/sangria_bank.hdf \
#   --verbose \
#   --minimum-bounds \
#     mchirp:4e5 \
#     eta:0.16 \
#     spin1z:-0.98 \
#     spin2z:-0.98 \
#     eclipticlongitude:0 \
#     eclipticlatitude:-1.5707 \
#     inclination:0 \
#     polarization:0 \
#   --maximum-bounds \
#     mchirp:9e6 \
#     eta:0.25 \
#     spin1z:0.98 \
#     spin2z:0.98 \
#     eclipticlongitude:6.283 \
#     eclipticlatitude:1.5707 \
#     inclination:3.1416 \
#     polarization:3.1416 \
#   --bounds-window-relative-to-initial \
#     log_mchirp:0.1 \
#     tc:3600

# mkdir -p output/round_0/subtract
# gwaith_subtract_signals \
#   --channels \
#     LISA_A LISA_E LISA_T \
#   --verbose \
#   --data-file \
#     output/data_processing/LDC2_sangria_hm_training_padded.hdf \
#   --signal-files \
#     output/round_0/optimise/optimise.hdf \
#   --output-file \
#     output/round_0/subtract/sangria_data_subtracted.hdf

# mkdir -p output/round_1/matched_filter
# for template_range in $(seq 0 $((${template_parallelization} - 1))) ; do
#   echo ${template_range}/${template_parallelization}
#   echo "Channel A"
#   hwilas_matched_filter \
#     --start-time 0 \
#     --end-time 6280000 \
#     --bank-file ../template_banks/output/sangria_bank.hdf \
#     --psd-file ../psd_estimation/A_sangria_hm_SMOOTHED_PSD.txt \
#     --input-data-file output/round_0/subtract/sangria_data_subtracted.hdf \
#     --channel LISA_A \
#     --output-file output/round_1/matched_filter/matched_filter_snr_peaks_A_${template_range}.hdf \
#     --template-fraction-range ${template_range}/${template_parallelization} \
#     --testing-plots output/round_1/matched_filter

#   echo "Channel E"
#   hwilas_matched_filter \
#   --start-time 0 \
#   --end-time 6280000 \
#   --bank-file ../template_banks/output/sangria_bank.hdf \
#   --psd-file ../psd_estimation/A_sangria_hm_SMOOTHED_PSD.txt \
#   --input-data-file output/round_0/subtract/sangria_data_subtracted.hdf \
#   --channel LISA_E \
#   --output-file output/round_1/matched_filter/matched_filter_snr_peaks_E_${template_range}.hdf \
#   --template-fraction-range ${template_range}/${template_parallelization} \
#   --testing-plots output/round_1/matched_filter

# done

# files_A=$( ls $(pwd)/output/round_1/matched_filter/matched_filter_snr_peaks_A_*.hdf )
# files_E=$( ls $(pwd)/output/round_1/matched_filter/matched_filter_snr_peaks_E_*.hdf )

# mkdir -p output/round_1/trigger_merge
# hwilas_trigger_merge \
#   --verbose \
#   --trigger-files $files_A \
#   --output-file output/round_1/trigger_merge/trigger_merge_A.hdf \
#   --testing-plot output/round_1/trigger_merge/trigger_merge_A.png 

# hwilas_trigger_merge \
#   --verbose \
#   --trigger-files $files_E \
#   --output-file output/round_1/trigger_merge_E.hdf \
#   --testing-plot output/round_1/trigger_merge_E.png 

# mkdir -p output/round_1/coincidence

# hwilas_coincidence \
#   --trigger-merge-files \
#     LISA_A:output/round_1/trigger_merge/trigger_merge_A.hdf \
#     LISA_E:output/round_1/trigger_merge/trigger_merge_E.hdf \
#   --output-file output/round_1/coincidence/coincidence.hdf \
#   --coincidence-window 3600 \
#   --testing-plots output/round_1/coincidence \
#   --expected-snr-file output/optimal_snr/optimal_snr_fitting_factors.hdf