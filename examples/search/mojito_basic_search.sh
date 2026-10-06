#!/bin/bash
set -e

# Basic MOJITO search workflow using the FastBHB (lisabeta) waveform generator
# with second-generation TDI (--tdi-generation 2), reusing the SANGRIA template
# bank for this initial test.
#
# Pipeline:
#   Step 1 (per chunk): Downsample the raw Mojito data to AET channels for that
#           chunk's time window.
#   Step 2 (once): Estimate the smoothed mojito PSD by pooling all downsampled
#           chunks together, so the Welch estimate uses far more independent
#           segments than a single-chunk estimate.
#   Step 3 (per chunk): Pad and inpaint using that PSD, then run matched
#           filtering over LISA_A / LISA_E.

RUN_MODE="slurm" # Options: "slurm" or "bash"

# All search outputs for this workflow go under output/mojito.
OUTPUT_DIR="output/mojito"

# Ensure the matched filter template script is executable
chmod +x "$(dirname "$0")/matched_filter_template.sh"

template_parallelization=15

# Mojito dataset (731 days of light data at 4 s cadence after downsampling).
mojito_data_file="/path/to/data.h5"
mojito_catalogue_file="/path/to/catalogue.hdf5"


# Total data duration: 731 days in seconds.
total_data_duration=$((731 * 86400))
chunk_duration=$((122 * 86400))      # just over 4 months in seconds
padding_for_process_data=$((30 * 86400)) # Padding for hwilas_split_pad_data, e.g., 30 days

# Waveform generator + TDI generation for matched filtering. FastBHB returns AET
# channels (LISA_A/E/T), so we search LISA_A and LISA_E using the mojito PSDs.
waveform_type="fastbhb"
tdi_generation="2"

# Template bank: for this initial test we reuse the SANGRIA bank.
bank_file="../template_banks/output/sangria_bank.hdf"

# Mojito smoothed PSD files, one per AET channel. These are estimated in Step 2
# by pooling all downsampled chunks together (hwilas_estimate_mojito_psd).
psd_dir="${OUTPUT_DIR}/round_0/psd"
psd_file_a="${psd_dir}/A_mojito_SMOOTHED_PSD.txt"
psd_file_e="${psd_dir}/E_mojito_SMOOTHED_PSD.txt"

# Welch segment duration for the PSD estimate (seconds). Default 18.25 days,
# so each ~4-month chunk contributes several independent segments.
psd_segment_duration=1576800

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

    # Input data file (mojito, padded) for this chunk.
    local input_data_file="${chunk_output_dir}/data_processing/mojito_padded_chunk${chunk_idx}.hdf"

    # Create the executable sbatch script from the template
    sed -e "s|@@CHUNK_OUTPUT_DIR@@|${chunk_output_dir}|g" \
        -e "s|@@CHANNEL@@|${channel}|g" \
        -e "s|@@CHUNK_IDX@@|${chunk_idx}|g" \
        -e "s|@@TEMPLATE_PARALLELIZATION_MINUS_1@@|${template_parallelization_minus_1}|g" \
        -e "s|@@TEMPLATE_PARALLELIZATION@@|${template_parallelization}|g" \
        -e "s|@@TEMPLATE_SCRIPT_PATH@@|${template_script_path}|g" \
        -e "s|@@WAVEFORM_TYPE@@|${waveform_type}|g" \
        -e "s|@@TDI_GENERATION@@|${tdi_generation}|g" \
        -e "s|@@BANK_FILE@@|${bank_file}|g" \
        -e "s|@@PSD_FILE_A@@|${psd_file_a}|g" \
        -e "s|@@PSD_FILE_E@@|${psd_file_e}|g" \
        -e "s|@@INPUT_DATA_FILE@@|${input_data_file}|g" \
        "${sbatch_template_path}" > "${sbatch_script}"

    chmod +x "${sbatch_script}"

    if [ "$RUN_MODE" == "slurm" ]; then
        job_id=$(sbatch --parsable "${sbatch_script}")
        matched_filter_job_ids+=("$job_id")
    else # RUN_MODE == "bash"
        "${sbatch_script}"
    fi
}

# =============================================================================
# Step 1 (per chunk): Downsample the raw Mojito data to AET channels.
# One downsampled HDF5 per chunk; all of them are pooled together in Step 2
# for a smoother PSD estimate.
# =============================================================================
mkdir -p "${OUTPUT_DIR}/round_0/data_processing"
downsampled_files=()

while [ $current_analysis_start_time -lt $total_data_duration ]; do
    current_analysis_end_time=$((current_analysis_start_time + chunk_duration))
    if [ $current_analysis_end_time -gt $total_data_duration ]; then
        current_analysis_end_time=$total_data_duration
    fi

    downsampled_file="${OUTPUT_DIR}/round_0/data_processing/mojito_downsampled_chunk${chunk_idx}.hdf"
    echo "=========================================================="
    echo "1. Downsampling Chunk ${chunk_idx}: [${current_analysis_start_time}, ${current_analysis_end_time})"
    echo "=========================================================="

    hwilas_process_mojito_data_chunk \
        --data-file "${mojito_data_file}" \
        --output-file "${downsampled_file}" \
        --low-cutoff-frequency 1e-5 \
        --channels LISA_A LISA_E LISA_T \
        --delta-t 4 \
        --analysis-start-time "${current_analysis_start_time}" \
        --analysis-end-time "${current_analysis_end_time}" \
        --verbose

    downsampled_files+=("${downsampled_file}")

    current_analysis_start_time=$current_analysis_end_time
    chunk_idx=$((chunk_idx + 1))
done

# =============================================================================
# Step 1b (once): Recombine all downsampled chunks into a single contiguous
# file. The per-chunk files are concatenated in order, so the result is one
# continuous AET time series spanning the whole dataset. This recombined file
# is used by the downstream truth-extraction / plotting steps.
# =============================================================================
recombined_file="${OUTPUT_DIR}/round_0/data_processing/mojito_downsampled.hdf"
echo "=========================================================="
echo "1b. Recombining ${#downsampled_files[@]} downsampled chunks into ${recombined_file}"
echo "=========================================================="

hwilas_process_mojito_data_recombine \
    --data-file "${downsampled_files[@]}" \
    --output-file "${recombined_file}" \
    --verbose

# =============================================================================
# Step 2 (once): Estimate the smoothed mojito PSD by pooling all downsampled
# chunks together. Each channel is concatenated across chunks before Welch's
# method is applied, so the estimate uses far more independent segments than a
# single-chunk estimate and is much smoother.
# =============================================================================
echo "=========================================================="
echo "2. Estimating mojito PSD from all ${#downsampled_files[@]} downsampled chunks"
echo "=========================================================="

hwilas_estimate_mojito_psd \
    --data-file "${downsampled_files[@]}" \
    --output-file \
        LISA_A:"${psd_file_a}" \
        LISA_E:"${psd_file_e}" \
        LISA_T:"${psd_dir}/T_mojito_SMOOTHED_PSD.txt" \
    --segment-duration "${psd_segment_duration}" \
    --smooth-regions 0.03,0.09 \
    --testing-plots "${psd_dir}" \
    --verbose

# =============================================================================
# Step 3 (per chunk): Pad + inpaint using the combined PSD, then run matched
# filtering over LISA_A / LISA_E.
# =============================================================================
current_analysis_start_time=0
chunk_idx=0

# Step 1: Extract the injected-signal ("truth") catalog once, up front, so it
# can be overlaid on the plotting steps below (coincidence, subtracted signals).
echo "Extracting truth data..."
mkdir -p "${OUTPUT_DIR}/round_0"
hwilas_extract_truth \
  --data-file "${mojito_catalogue_file}" \
  --output-file "${OUTPUT_DIR}/round_0/truth.hdf" \
  --channels LISA_A LISA_E LISA_T \
  --verbose \
  --type mojito

while [ $current_analysis_start_time -lt $total_data_duration ]; do
    current_analysis_end_time=$((current_analysis_start_time + chunk_duration))
    if [ $current_analysis_end_time -gt $total_data_duration ]; then
        current_analysis_end_time=$total_data_duration
    fi

    # Per-chunk output directory for this round of the search.
    chunk_output_dir="${OUTPUT_DIR}/round_0/chunk_${chunk_idx}"
    mkdir -p "${chunk_output_dir}/data_processing"
    mkdir -p "${chunk_output_dir}/matched_filter"
    mkdir -p "${chunk_output_dir}/logs"

    # Reuse the downsampled file produced in Step 1 for this chunk.
    downsampled_file="${OUTPUT_DIR}/round_0/data_processing/mojito_downsampled_chunk${chunk_idx}.hdf"
    padded_file="${chunk_output_dir}/data_processing/mojito_padded_chunk${chunk_idx}.hdf"

    # Pad + inpaint the downsampled chunk using the combined mojito PSD.
    echo "Running hwilas_split_pad_data for chunk ${chunk_idx}"
    hwilas_split_pad_data \
        --data-file "${downsampled_file}" \
        --output-file "${padded_file}" \
        --psd-file "${psd_file_a}" \
        --channels LISA_A LISA_E \
        --padding-duration "${padding_for_process_data}" \
        --analysis-start-time "${current_analysis_start_time}" \
        --analysis-end-time "${current_analysis_end_time}" \
        --verbose

    # Plot the padded + inpainted data (and whitened / overwhitened diagnostics)
    # for a quick visual sanity check.
    hwilas_plot_split_pad_data \
        --verbose \
        --data-file "${padded_file}" \
        --psd-file "${psd_file_a}" \
        --channels LISA_A LISA_E \
        --analysis-start-time "${current_analysis_start_time}" \
        --analysis-end-time "${current_analysis_end_time}" \
        --truth-file "${OUTPUT_DIR}/round_0/truth.hdf" \
        --output-file "${chunk_output_dir}/data_processing/plot_split_pad_data.png"

    # Submit matched filter jobs for this chunk, one job array per channel.
    for channel in "LISA_A" "LISA_E"; do
        if [ "${channel}" == "LISA_A" ]; then
            psd_file="${psd_file_a}"
        else
            psd_file="${psd_file_e}"
        fi

        echo "=========================================================="
        echo "4. Submitting Matched Filter jobs for Channel ${channel} (Chunk ${chunk_idx})"
        echo "=========================================================="

        if [ "${RUN_MODE}" == "slurm" ]; then
            job_id=$(handle_matched_filter_jobs "${chunk_output_dir}" "${channel}" "${chunk_idx}" "${template_parallelization}")
            matched_filter_job_ids+=("${job_id}")
        else
            handle_matched_filter_jobs_bash "${chunk_output_dir}" "${channel}" "${chunk_idx}" "${template_parallelization}"
        fi
    done

    # Update for next chunk
    current_analysis_start_time=$current_analysis_end_time
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
      active_jobs=$((active_jobs + $(squeue -h -r -j "$job_id" 2>/dev/null | wc -l)))
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
files_A=(output/mojito/round_0/chunk_*/matched_filter/matched_filter_snr_peaks_LISA_A_*.hdf)
files_E=(output/mojito/round_0/chunk_*/matched_filter/matched_filter_snr_peaks_LISA_E_*.hdf)
shopt -u nullglob

# echo ${files_A[@]}
# echo ${files_E[@]}


if [ ${#files_A[@]} -eq 0 ] || [ ${#files_E[@]} -eq 0 ]; then
  echo "Missing matched-filter output files after Slurm completion."
  echo "Check logs in output/mojito/round_0/logs"
  exit 1
fi

mkdir -p output/mojito/round_0/trigger_merge
mkdir -p output/mojito/round_0/trigger_merge
echo "Running Trigger Merge Jobs"
hwilas_trigger_merge \
  --verbose \
  --trigger-files "${files_A[@]}" \
  --output-file output/mojito/round_0/trigger_merge/trigger_merge_A.hdf \
  --cluster-window 2592000

hwilas_plot_trigger_merge \
  --verbose \
  --trigger-file output/mojito/round_0/trigger_merge/trigger_merge_A.hdf \
  --channel LISA_A \
  --output output/mojito/round_0/trigger_merge/trigger_merge_LISA_A.png

hwilas_trigger_merge \
  --verbose \
  --trigger-files "${files_E[@]}" \
  --output-file output/mojito/round_0/trigger_merge/trigger_merge_E.hdf \
  --cluster-window 2592000

hwilas_plot_trigger_merge \
  --verbose \
  --trigger-file output/mojito/round_0/trigger_merge/trigger_merge_E.hdf \
  --channel LISA_E \
  --output output/mojito/round_0/trigger_merge/trigger_merge_LISA_E.png

mkdir -p output/mojito/round_0/coincidence

hwilas_coincidence \
  --verbose \
  --trigger-merge-files \
    LISA_A:output/mojito/round_0/trigger_merge/trigger_merge_A.hdf \
    LISA_E:output/mojito/round_0/trigger_merge/trigger_merge_E.hdf \
  --output-file output/mojito/round_0/coincidence/coincidence.hdf \
  --coincidence-window 3600

hwilas_plot_coincidence \
  --verbose \
  --input-file output/mojito/round_0/coincidence/coincidence.hdf \
  --channels LISA_A LISA_E \
  --coincidence-window 3600 \
  --truth-file "${OUTPUT_DIR}/round_0/truth.hdf" \
  --output-directory output/mojito/round_0/coincidence


mkdir -p output/mojito/round_0/optimise

hwilas_optimise_snr_peaks \
  --input-file output/mojito/round_0/coincidence/coincidence.hdf \
  --output-timeseries-file \
    output/mojito/round_0/optimise/optimise.hdf \
  --output-parameter-file \
    output/mojito/round_0/optimise/optimise_parameters.hdf \
  --data-files \
    output/mojito/round_0/chunk_0/data_processing/LDC2_sangria_hm_training_padded.hdf \
    output/mojito/round_0/chunk_1/data_processing/LDC2_sangria_hm_training_padded.hdf \
    output/mojito/round_0/chunk_2/data_processing/LDC2_sangria_hm_training_padded.hdf \
  --psd-file ../psd_estimation/output/A_sangria_hm_SMOOTHED_PSD.txt \
  --bank-file ../template_banks/output/sangria_bank.hdf \
  --verbose \
  --minimum-bounds \
    mchirp:4e5 \
    eta:0.16 \
    spin1z:-0.98 \
    spin2z:-0.98 \
    eclipticlongitude:0 \
    eclipticlatitude:-1.5707 \
    inclination:0 \
    polarization:0 \
  --maximum-bounds \
    mchirp:9e6 \
    eta:0.25 \
    spin1z:0.98 \
    spin2z:0.98 \
    eclipticlongitude:6.2832 \
    eclipticlatitude:1.5707 \
    inclination:3.1416 \
    polarization:6.2832 \
  --bounds-window-relative-to-initial \
    log_mchirp:0.1 \
    tc:3600 \
  --f-lower 1e-5

mkdir -p output/mojito/round_0/optimise/plots
hwilas_plot_optimise_waveforms \
  --parameter-file output/mojito/round_0/optimise/optimise_parameters.hdf \
  --input-file output/mojito/round_0/coincidence/coincidence.hdf \
  --bank-file ../template_banks/output/sangria_bank.hdf \
  --data-files \
    output/mojito/round_0/chunk_0/data_processing/LDC2_sangria_hm_training_padded.hdf \
    output/mojito/round_0/chunk_1/data_processing/LDC2_sangria_hm_training_padded.hdf \
    output/mojito/round_0/chunk_2/data_processing/LDC2_sangria_hm_training_padded.hdf \
  --psd-file ../psd_estimation/output/A_sangria_hm_SMOOTHED_PSD.txt \
  --output-directory output/mojito/round_0/optimise/plots \
  --f-lower 1e-5 \
  --verbose

rm output/mojito/round_0/subtract/sangria_data_subtracted.hdf
mkdir -p output/mojito/round_0/subtract
gwaith_subtract_signals \
  --channels \
    LISA_A LISA_E LISA_T \
  --verbose \
  --data \
    output/mojito/round_0/data_processing/sangria_aet.hdf \
  --signal-files \
    output/mojito/round_0/optimise/optimise.hdf \
  --output \
    output/mojito/round_0/subtract/sangria_data_subtracted.hdf

mkdir -p output/mojito/round_0/subtract/plots
hwilas_plot_subtracted_signals \
  --data-file \
    output/mojito/round_0/data_processing/sangria_aet.hdf \
  --signal-file \
    output/mojito/round_0/optimise/optimise.hdf \
  --subtracted-file \
    output/mojito/round_0/subtract/sangria_data_subtracted.hdf \
  --parameter-file \
    output/mojito/round_0/optimise/optimise_parameters.hdf \
  --truth-file \
    "${OUTPUT_DIR}/round_0/truth.hdf" \
  --output-directory \
    output/mojito/round_0/subtract/plots \
  --channels LISA_A LISA_E LISA_T \
  --event-window 0.05,0.025 \
  --verbose

# mkdir -p output/mojito/round_1/matched_filter
for template_range in $(seq 0 $((${template_parallelization} - 1))) ; do
  echo ${template_range}/${template_parallelization}
  echo "Channel A"
  hwilas_matched_filter \
    --start-time 0 \
    --end-time 6280000 \
    --bank-file ../template_banks/output/sangria_bank.hdf \
    --psd-file ../psd_estimation/A_sangria_hm_SMOOTHED_PSD.txt \
    --input-data-file output/mojito/round_0/subtract/sangria_data_subtracted.hdf \
    --channel LISA_A \
    --output-file output/mojito/round_1/matched_filter/matched_filter_snr_peaks_A_${template_range}.hdf \
    --template-fraction-range ${template_range}/${template_parallelization} \
    --testing-plots output/mojito/round_1/matched_filter

  echo "Channel E"
  hwilas_matched_filter \
  --start-time 0 \
  --end-time 6280000 \
  --bank-file ../template_banks/output/sangria_bank.hdf \
  --psd-file ../psd_estimation/A_sangria_hm_SMOOTHED_PSD.txt \
  --input-data-file output/mojito/round_0/subtract/sangria_data_subtracted.hdf \
  --channel LISA_E \
  --output-file output/mojito/round_1/matched_filter/matched_filter_snr_peaks_E_${template_range}.hdf \
  --template-fraction-range ${template_range}/${template_parallelization} \
  --testing-plots output/mojito/round_1/matched_filter

done

files_A=$( ls $(pwd)/output/mojito/round_1/matched_filter/matched_filter_snr_peaks_A_*.hdf )
files_E=$( ls $(pwd)/output/mojito/round_1/matched_filter/matched_filter_snr_peaks_E_*.hdf )

mkdir -p output/mojito/round_1/trigger_merge
hwilas_trigger_merge \
  --verbose \
  --trigger-files $files_A \
  --output-file output/mojito/round_1/trigger_merge/trigger_merge_A.hdf

hwilas_plot_trigger_merge \
  --trigger-file output/mojito/round_1/trigger_merge/trigger_merge_A.hdf \
  --channel LISA_A \
  --output-directory output/mojito/round_1/trigger_merge

hwilas_trigger_merge \
  --verbose \
  --trigger-files $files_E \
  --output-file output/mojito/round_1/trigger_merge_E.hdf

hwilas_plot_trigger_merge \
  --trigger-file output/mojito/round_1/trigger_merge_E.hdf \
  --channel LISA_E \
  --output-directory output/mojito/round_1/trigger_merge

mkdir -p output/mojito/round_1/coincidence

hwilas_coincidence \
  --trigger-merge-files \
    LISA_A:output/mojito/round_1/trigger_merge/trigger_merge_A.hdf \
    LISA_E:output/mojito/round_1/trigger_merge/trigger_merge_E.hdf \
  --output-file output/mojito/round_1/coincidence/coincidence.hdf \
  --coincidence-window 3600 \
  --expected-snr-file output/optimal_snr/optimal_snr_fitting_factors.hdf

hwilas_plot_coincidence \
  --input-file output/mojito/round_1/coincidence/coincidence.hdf \
  --channels LISA_A LISA_E \
  --coincidence-window 3600 \
  --expected-snr-file output/optimal_snr/optimal_snr_fitting_factors.hdf \
  --output-directory output/mojito/round_1/coincidence