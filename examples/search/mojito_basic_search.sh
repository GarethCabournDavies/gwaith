#!/usr/bin/env bash
set -euo pipefail
# Disable OpenMP runtime notices; set KMP_WARNINGS=TRUE to enable diagnostics.
export KMP_WARNINGS="${KMP_WARNINGS:-FALSE}"
# Each waveform worker uses one numerical thread to avoid CPU oversubscription.
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
cd "$script_dir"
OUTPUT_DIR="${OUTPUT_DIR:-output/mojito}"
data_file="${DATA_FILE:-${script_dir}/../data_processing/mojito_all_downsampled.hdf}"
truth_source="${TRUTH_SOURCE:-${script_dir}/catalogue.hdf5}"
bank_file="${BANK_FILE:-../template_banks/output/mojito_bank.hdf}"
psd_dir="${PSD_DIR:-../psd_estimation/output/mojito}"
psd_file_a="${psd_dir}/A_mojito_SMOOTHED_PSD.txt"
psd_file_e="${psd_dir}/E_mojito_SMOOTHED_PSD.txt"
psd_file_t="${psd_dir}/T_mojito_UNSMOOTHED_PSD.txt"
psd_fit_t="${psd_dir}/T_mojito_SMOOTHED_PSD.txt"
search_model="${SEARCH_APPROXIMANT:-fastbhb}"
search_orbits="${SEARCH_ORBITS:-analytic}"
tdi_generation=2
optimisation_model="${OPTIMISATION_MODEL:-BBHx}"
refinement_model="${REFINEMENT_MODEL:-BBHx}"
optimisation_orbit="${OPTIMISATION_ORBIT:-esa}"
refinement_orbit="${REFINEMENT_ORBIT:-esa}"
# Phase is fitted at a fixed reference frequency, independent of the bank phase.
optimisation_f_ref="${OPTIMISATION_F_REF:-0}"
refinement_f_ref="${REFINEMENT_F_REF:-0}"
# Absolute timestamp of the first input sample, supplied with the dataset.
data_time_origin="${DATA_TIME_ORIGIN:?Set DATA_TIME_ORIGIN to the absolute timestamp of the first input sample}"
# Export every retained CSV row with starting parameters and provenance.
pe_time_origin="${PE_TIME_ORIGIN:-$data_time_origin}"
pe_phase_convention="${PE_NATIVE_PHASE_CONVENTION:-pi-over-2-minus-phase}"
pe_export_dir="${PE_EXPORT_DIR:-${OUTPUT_DIR}/final/pe}"
truth_file="${OUTPUT_DIR}/mojito_truth.hdf"
n_processes="${N_PROCESSES:-4}"

mkdir -p "$OUTPUT_DIR"
hwilas_extract_truth --data-file "$truth_source" --type mojito --no-timeseries --output-file "$truth_file" --verbose
search_dir="${OUTPUT_DIR}/round_0/matched_filter"
template_parallelization=$n_processes
network_statistic_threshold=${NETWORK_STATISTIC_THRESHOLD:-6}
residual_statistic_threshold=${RESIDUAL_STATISTIC_THRESHOLD:-4}

# Plot original data before the first search round.
hwilas_plot_processed_data \
    --data-file "$data_file" \
    --truth-file "$truth_file" \
    --end-exclusion 259200 \
    --output-plot "${OUTPUT_DIR}/round_0/data.png" \
    --output-whitened-plot "${OUTPUT_DIR}/round_0/data_whitened.png" \
    --output-overwhitened-plot "${OUTPUT_DIR}/round_0/data_overwhitened.png" \
    --psd-files LISA_A:"$psd_file_a" LISA_E:"$psd_file_e" LISA_T:"$psd_file_t" \
    --verbose

# Combine A/E at every matched-filter sample; cluster network peaks across the bank.
candidate_dir="${OUTPUT_DIR}/round_0/candidates"
mkdir -p "$search_dir" "$candidate_dir"
status=0
pids=()
trigger_files=()
for ((part=0; part<template_parallelization; part++)); do
    trigger_files+=("${search_dir}/matched_filter_AE_${part}.hdf")
    hwilas_matched_filter_search \
        --input-data-file "$data_file" \
        --bank-file "$bank_file" \
        --channels LISA_A LISA_E \
        --psd-files LISA_A:"$psd_file_a" LISA_E:"$psd_file_e" \
        --approximant "$search_model" --orbits "$search_orbits" --data-time-origin "$data_time_origin" \
        --tdi-generation "$tdi_generation" \
        --padding-time 129600 \
        --end-exclusion 259200 \
        --f-upper 0.09 \
        --statistic-threshold "$network_statistic_threshold" \
        --template-fraction-range "${part}/${template_parallelization}" \
        --output-file "${search_dir}/matched_filter_AE_${part}.hdf" \
        --verbose &
    pids+=("$!")
done
for pid in "${pids[@]}"; do
    wait "$pid" || status=1
done
[ "$status" -eq 0 ] || exit "$status"

hwilas_trigger_merge \
    --trigger-files "${trigger_files[@]}" \
    --output-file "${candidate_dir}/candidates.hdf" \
    --unclustered-output-file "${candidate_dir}/candidates_unclustered.hdf" \
    --cluster-across-templates \
    --cluster-window 86400 \
    --verbose

hwilas_candidate_hdf_to_csv \
    --input-file "${candidate_dir}/candidates.hdf" \
    --output-file "${candidate_dir}/candidates.csv" \
    --bank-file "$bank_file" \
    --verbose

hwilas_plot_trigger_merge \
    --trigger-file "${candidate_dir}/candidates_unclustered.hdf" \
    --clustered-trigger-file "${candidate_dir}/candidates.hdf" \
    --network-statistic \
    --channel "LISA_A+LISA_E" \
    --output "${candidate_dir}/candidates.png" \
    --truth-file "$truth_file" \
    --verbose


# # Fit coincident candidates and write their summed signal on the original data grid.
optimise_dir="${OUTPUT_DIR}/round_0/optimise"
subtract_dir="${OUTPUT_DIR}/round_0/subtract"
mkdir -p "$optimise_dir" "$subtract_dir"

hwilas_optimise_snr_peaks \
    --approximant "$optimisation_model" \
    --refinement-approximant "$refinement_model" \
    --orbits "$optimisation_orbit" --refinement-orbits "$refinement_orbit" --data-time-origin "$data_time_origin" \
    --f-ref "$optimisation_f_ref" --refinement-f-ref "$refinement_f_ref" --mass-coordinate log \
    --refinement-generator-keywords response_nodes:4096 \
    --input-file "${candidate_dir}/candidates.hdf" \
    --bank-file "$bank_file" \
    --data-files "$data_file" \
    --psd-files LISA_A:"$psd_file_a" LISA_E:"$psd_file_e" LISA_T:"$psd_fit_t" \
    --fit-channels LISA_A LISA_E \
    --optimisation-channels LISA_A LISA_E \
    --output-timeseries-channels LISA_A LISA_E LISA_T \
    --output-timeseries-file "${optimise_dir}/optimise.hdf" \
    --output-parameter-file "${optimise_dir}/optimise_parameters.hdf" \
    --tdi-generation "$tdi_generation" \
    --statistic-threshold 40 \
    --minimum-peak-separation 259200 \
    --optimisation-window-before 86400 \
    --optimisation-window-after 14400 \
    --n-particles 32 \
    --n-iterations 100 \
    --n-processes "$n_processes" \
    --polish-evaluations 1000 \
    --seed 606 \
    --minimum-bounds \
        mchirp:5e4 eta:0.0826446281 spin1z:-0.99 spin2z:-0.99 \
        eclipticlongitude:0 eclipticlatitude:-1.57079632679 \
        inclination:0 polarization:0 \
    --maximum-bounds \
        mchirp:3e7 eta:0.25 spin1z:0.99 spin2z:0.99 \
        eclipticlongitude:6.28318530718 eclipticlatitude:1.57079632679 \
        inclination:3.14159265359 polarization:3.14159265359 \
    --bounds-window-relative-to-initial log_mchirp:0.65 tc:3600 \
    --f-lower 0.00003 \
    --verbose

hwilas_candidate_hdf_to_csv \
    --input-file "${optimise_dir}/optimise_parameters.hdf" \
    --output-file "${optimise_dir}/optimised_events.csv" \
    --verbose

# Subtract the first-round model before searching for weaker or overlapping signals.
hwilas_subtract_signals \
    --data "${data_file}" \
    --signal-files "${optimise_dir}/optimise.hdf" \
    --output "${subtract_dir}/data_subtracted.hdf" \
    --channels LISA_A LISA_E LISA_T \
    --verbose

# Plot the first residual with markers for the signals already removed.
hwilas_plot_processed_data \
    --data-file "${subtract_dir}/data_subtracted.hdf" \
    --truth-file "$truth_file" \
    --removed-parameter-files "${optimise_dir}/optimise_parameters.hdf" \
    --end-exclusion 259200 \
    --output-plot "${OUTPUT_DIR}/round_1/data.png" \
    --output-whitened-plot "${OUTPUT_DIR}/round_1/data_whitened.png" \
    --output-overwhitened-plot "${OUTPUT_DIR}/round_1/data_overwhitened.png" \
    --psd-files LISA_A:"$psd_file_a" LISA_E:"$psd_file_e" LISA_T:"$psd_file_t" \
    --verbose

# Second round: search A/E jointly on the residual and cluster the network candidates.
round1_search_dir="${OUTPUT_DIR}/round_1/matched_filter"
round1_candidate_dir="${OUTPUT_DIR}/round_1/candidates"

mkdir -p "$round1_search_dir" "$round1_candidate_dir"
status=0
pids=()
trigger_files=()
for ((part=0; part<template_parallelization; part++)); do
    trigger_files+=("${round1_search_dir}/matched_filter_AE_${part}.hdf")
    hwilas_matched_filter_search \
        --input-data-file "${subtract_dir}/data_subtracted.hdf" \
        --bank-file "$bank_file" \
        --channels LISA_A LISA_E \
        --psd-files LISA_A:"$psd_file_a" LISA_E:"$psd_file_e" \
        --approximant "$search_model" --orbits "$search_orbits" --data-time-origin "$data_time_origin" \
        --tdi-generation "$tdi_generation" \
        --padding-time 129600 \
        --end-exclusion 259200 \
        --f-upper 0.09 \
        --statistic-threshold "$network_statistic_threshold" \
        --template-fraction-range "${part}/${template_parallelization}" \
        --output-file "${round1_search_dir}/matched_filter_AE_${part}.hdf" \
        --verbose &
    pids+=("$!")
done
for pid in "${pids[@]}"; do
    wait "$pid" || status=1
done
[ "$status" -eq 0 ] || exit "$status"

hwilas_trigger_merge \
    --trigger-files "${trigger_files[@]}" \
    --output-file "${round1_candidate_dir}/candidates.hdf" \
    --unclustered-output-file "${round1_candidate_dir}/candidates_unclustered.hdf" \
    --cluster-across-templates \
    --cluster-window 86400 \
    --verbose

hwilas_candidate_hdf_to_csv \
    --input-file "${round1_candidate_dir}/candidates.hdf" \
    --output-file "${round1_candidate_dir}/candidates.csv" \
    --previous-event-files "${candidate_dir}/candidates.hdf" \
    --exclusion-window 3600 \
    --bank-file "$bank_file" \
    --verbose

hwilas_plot_trigger_merge \
    --trigger-file "${round1_candidate_dir}/candidates_unclustered.hdf" \
    --clustered-trigger-file "${round1_candidate_dir}/candidates.hdf" \
    --network-statistic \
    --channel "LISA_A+LISA_E" \
    --output "${round1_candidate_dir}/candidates.png" \
    --truth-file "$truth_file" \
    --verbose


# Collect cross-round seeds, keeping earlier-round events within the exclusion window.
hwilas_collect_search_candidates \
    --input-files "${candidate_dir}/candidates.hdf" "${round1_candidate_dir}/candidates.hdf" \
    --output-file "${OUTPUT_DIR}/final/all_candidates.hdf" \
    --csv-file "${OUTPUT_DIR}/final/all_candidates.csv" \
    --bank-file "$bank_file" \
    --close-event-window 3600 \
    --exclude-later-close-events \
    --verbose

# Fit every surviving second-round candidate; retain earlier-window flags.
round1_optimise_dir="${OUTPUT_DIR}/round_1/optimise"
mkdir -p "$round1_optimise_dir"

hwilas_optimise_snr_peaks \
    --approximant "$optimisation_model" \
    --refinement-approximant "$refinement_model" \
    --orbits "$optimisation_orbit" --refinement-orbits "$refinement_orbit" --data-time-origin "$data_time_origin" \
    --f-ref "$optimisation_f_ref" --refinement-f-ref "$refinement_f_ref" --mass-coordinate log \
    --refinement-generator-keywords response_nodes:4096 \
    --input-file "${round1_candidate_dir}/candidates.hdf" \
    --bank-file "$bank_file" \
    --data-files "${subtract_dir}/data_subtracted.hdf" \
    --psd-files LISA_A:"$psd_file_a" LISA_E:"$psd_file_e" LISA_T:"$psd_fit_t" \
    --fit-channels LISA_A LISA_E \
    --optimisation-channels LISA_A LISA_E \
    --minimum-peak-separation 3600 \
    --output-timeseries-channels LISA_A LISA_E LISA_T \
    --output-timeseries-file "${round1_optimise_dir}/optimise.hdf" \
    --output-parameter-file "${round1_optimise_dir}/optimise_parameters.hdf" \
    --tdi-generation "$tdi_generation" \
    --statistic-threshold 20 \
    --previous-parameter-file "${optimise_dir}/optimise_parameters.hdf" \
    --optimisation-window-before 86400 \
    --optimisation-window-after 14400 \
    --n-particles 32 \
    --n-iterations 100 \
    --n-processes "$n_processes" \
    --polish-evaluations 1000 \
    --seed 607 \
    --minimum-bounds \
        mchirp:5e4 eta:0.0826446281 spin1z:-0.99 spin2z:-0.99 \
        eclipticlongitude:0 eclipticlatitude:-1.57079632679 \
        inclination:0 polarization:0 \
    --maximum-bounds \
        mchirp:3e7 eta:0.25 spin1z:0.99 spin2z:0.99 \
        eclipticlongitude:6.28318530718 eclipticlatitude:1.57079632679 \
        inclination:3.14159265359 polarization:3.14159265359 \
    --bounds-window-relative-to-initial log_mchirp:0.65 tc:3600 \
    --f-lower 0.00003 \
    --verbose

hwilas_candidate_hdf_to_csv \
    --input-file "${round1_optimise_dir}/optimise_parameters.hdf" \
    --output-file "${round1_optimise_dir}/optimised_events.csv" \
    --previous-event-files "${optimise_dir}/optimise_parameters.hdf" \
    --exclusion-window 3600 \
    --verbose

# Keep the full fitted catalogue and subtract both models once from the original.
final_dir="${OUTPUT_DIR}/final"
all_parameters_file="${final_dir}/all_optimised_parameters.hdf"
final_residual_file="${final_dir}/residual.hdf"
all_signals_file="${final_dir}/all_optimised_signals.hdf"
mkdir -p "$final_dir"
hwilas_combine_search_parameters \
    --input-files "${optimise_dir}/optimise_parameters.hdf" "${round1_optimise_dir}/optimise_parameters.hdf" \
    --output-file "$all_parameters_file" \
    --csv-file "${final_dir}/all_optimised_parameters.csv" \
    --data-file "$data_file" \
    --output-timeseries-file "$all_signals_file" \
    --channels LISA_A LISA_E LISA_T \
    --close-event-window 3600 \
    --exclude-later-close-events \
    --verbose

# Subtract the retained-event model once from the original strain.
hwilas_subtract_signals \
    --data "$data_file" \
    --signal-files "$all_signals_file" \
    --output "$final_residual_file" \
    --channels LISA_A LISA_E LISA_T \
    --verbose

# Plot the final residual with removal markers from both optimisation rounds.
hwilas_plot_processed_data \
    --data-file "$final_residual_file" \
    --truth-file "$truth_file" \
    --removed-parameter-files "$all_parameters_file" \
    --end-exclusion 259200 \
    --output-plot "${final_dir}/residual_test/data.png" \
    --output-whitened-plot "${final_dir}/residual_test/data_whitened.png" \
    --output-overwhitened-plot "${final_dir}/residual_test/data_overwhitened.png" \
    --psd-files LISA_A:"$psd_file_a" LISA_E:"$psd_file_e" LISA_T:"$psd_file_t" \
    --verbose

# Final residual test: repeat the bank search without fitting further candidates.
residual_search_dir="${OUTPUT_DIR}/final/residual_test/matched_filter"
residual_candidate_dir="${OUTPUT_DIR}/final/residual_test/candidates"

mkdir -p "$residual_search_dir" "$residual_candidate_dir"
status=0
pids=()
trigger_files=()
for ((part=0; part<template_parallelization; part++)); do
    trigger_files+=("${residual_search_dir}/matched_filter_AE_${part}.hdf")
    hwilas_matched_filter_search \
        --input-data-file "$final_residual_file" \
        --bank-file "$bank_file" \
        --channels LISA_A LISA_E \
        --psd-files LISA_A:"$psd_file_a" LISA_E:"$psd_file_e" \
        --approximant "$search_model" \
        --orbits "$search_orbits" \
        --data-time-origin "$data_time_origin" \
        --tdi-generation "$tdi_generation" \
        --padding-time 129600 \
        --end-exclusion 259200 \
        --f-upper 0.09 \
        --statistic-threshold "$residual_statistic_threshold" \
        --template-fraction-range "${part}/${template_parallelization}" \
        --output-file "${residual_search_dir}/matched_filter_AE_${part}.hdf" \
        --verbose &
    pids+=("$!")
done
for pid in "${pids[@]}"; do
    wait "$pid" || status=1
done
[ "$status" -eq 0 ] || exit "$status"

hwilas_trigger_merge \
    --trigger-files "${trigger_files[@]}" \
    --output-file "${residual_candidate_dir}/candidates.hdf" \
    --unclustered-output-file "${residual_candidate_dir}/candidates_unclustered.hdf" \
    --cluster-across-templates \
    --cluster-window 86400 \
    --verbose

hwilas_candidate_hdf_to_csv \
    --input-file "${residual_candidate_dir}/candidates.hdf" \
    --output-file "${residual_candidate_dir}/candidates.csv" \
    --bank-file "$bank_file" \
    --verbose

hwilas_plot_trigger_merge \
    --trigger-file "${residual_candidate_dir}/candidates_unclustered.hdf" \
    --clustered-trigger-file "${residual_candidate_dir}/candidates.hdf" \
    --network-statistic \
    --channel "LISA_A+LISA_E" \
    --output "${residual_candidate_dir}/candidates.png" \
    --truth-file "$truth_file" \
    --verbose

# Check subtraction and plot cumulative residual ranks outside fitted signal windows.
hwilas_check_search_residuals \
    --data-file "$data_file" \
    --residual-file "$final_residual_file" \
    --end-exclusion 259200 \
    --signal-files "$all_signals_file" \
    --parameter-file "$all_parameters_file" \
    --candidate-file "${residual_candidate_dir}/candidates.hdf" \
    --signal-exclusion "${RESIDUAL_SIGNAL_EXCLUSION:-129600}" \
    --output-statistic-plot "${final_dir}/residual_statistics.png" \
    --output-file "${final_dir}/residual_report.json" \
    --verbose

# Enumerate CSV records in their saved statistic order, excluding the header.
pe_row_indices=$(python - "${final_dir}/all_optimised_parameters.csv" <<'PYCSV'
import csv
import sys
with open(sys.argv[1], newline='') as stream:
    for row_index, row in enumerate(csv.DictReader(stream)):
        print(row_index)
PYCSV
)
# Write one named starting-parameter YAML per row; MBHBS owns data and PSDs.
while IFS= read -r row_index; do
    [ -n "$row_index" ] || continue
    hwilas_export_pe_seed \
        --parameters "$all_parameters_file" \
        --row-index "$row_index" \
        --target-time-origin "$pe_time_origin" \
        --native-phase-convention "$pe_phase_convention" \
        --out "${pe_export_dir}/seed_${row_index}.yaml"
done <<< "$pe_row_indices"
