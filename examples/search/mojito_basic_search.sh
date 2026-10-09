#!/usr/bin/env bash
set -euo pipefail

# Search in two rounds using saved PSDs, then test the final residual.
script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
cd "$script_dir"
mojito_data_file="${script_dir}/../data_processing/mojito_all_downsampled.hdf"
OUTPUT_DIR="output/mojito"
psd_dir="${OUTPUT_DIR}/round_0/psd"
psd_file_a="${psd_dir}/A_mojito_SMOOTHED_PSD.txt"
psd_file_e="${psd_dir}/E_mojito_SMOOTHED_PSD.txt"
psd_segment_duration=1576800
search_dir="${OUTPUT_DIR}/round_0/matched_filter"
template_parallelization=4
mojito_truth_file="${OUTPUT_DIR}/mojito_truth.hdf"

# # Extract catalogue-only truth for injection-time overlays.
# mkdir -p "$OUTPUT_DIR"
# hwilas_extract_truth \
#     --data-file "${script_dir}/catalogue.hdf5" \
#     --type mojito \
#     --no-timeseries \
#     --output-file "$mojito_truth_file" \
#     --verbose

# hwilas_estimate_mojito_psd \
#     --data-file "${mojito_data_file}" \
#     --output-file \
#         LISA_A:"${psd_file_a}" \
#         LISA_E:"${psd_file_e}" \
#         LISA_T:"${psd_dir}/T_mojito_SMOOTHED_PSD.txt" \
#     --unsmoothed-output-file \
#         LISA_A:"${psd_dir}/A_mojito_UNSMOOTHED_PSD.txt" \
#         LISA_E:"${psd_dir}/E_mojito_UNSMOOTHED_PSD.txt" \
#         LISA_T:"${psd_dir}/T_mojito_UNSMOOTHED_PSD.txt" \
#     --segment-duration "${psd_segment_duration}" \
#     --smooth-regions 0.02,0.04 0.05,0.07 0.08,0.1 \
#     --verbose

# # Plot the saved A/E/T PSDs together for comparison.
# hwilas_plot_estimate_mojito_psd \
#     --psd-files \
#         LISA_A:"${psd_file_a}" \
#         LISA_E:"${psd_file_e}" \
#         LISA_T:"${psd_dir}/T_mojito_SMOOTHED_PSD.txt" \
#     --unsmoothed-psd-files \
#         LISA_A:"${psd_dir}/A_mojito_UNSMOOTHED_PSD.txt" \
#         LISA_E:"${psd_dir}/E_mojito_UNSMOOTHED_PSD.txt" \
#         LISA_T:"${psd_dir}/T_mojito_UNSMOOTHED_PSD.txt" \
#     --output-file "${psd_dir}/mojito_psd.png" \
#     --verbose


# Plot original data before the first search round.
hwilas_plot_downsampled_mojito \
    --data-file "$mojito_data_file" \
    --truth-file "$mojito_truth_file" \
    --end-exclusion 259200 \
    --output-file "${OUTPUT_DIR}/round_0/data.png" \
    --verbose

# Search each channel using the standard matched-filter executable and trigger format.
mkdir -p "$search_dir"
# for channel in LISA_A LISA_E; do
#     if [ "$channel" = LISA_A ]; then
#         psd_file="$psd_file_a"
#     else
#         psd_file="$psd_file_e"
#     fi
#     pids=()
#     for ((part=0; part<template_parallelization; part++)); do
#         hwilas_matched_filter_search \
#             --input-data-file "$mojito_data_file" \
#             --bank-file ../template_banks/output/mojito_bank.hdf \
#             --psd-file "$psd_file" \
#             --channel "$channel" \
#             --waveform-type fastbhb \
#             --tdi-generation 2 \
#             --padding-time 129600 \
#             --end-exclusion 259200 \
#             --f-upper 0.09 \
#             --snr-threshold 4 \
#             --template-fraction-range "${part}/${template_parallelization}" \
#             --output-file "${search_dir}/matched_filter_snr_peaks_${channel}_${part}.hdf" \
#             --verbose &
#         pids+=("$!")
#     done
#     status=0
#     for pid in "${pids[@]}"; do
#         wait "$pid" || status=1
#     done
#     [ "$status" -eq 0 ] || exit "$status"
# done

# Merge template partitions per channel before finding coincidences.
merge_dir="${OUTPUT_DIR}/round_0/trigger_merge"
coincidence_dir="${OUTPUT_DIR}/round_0/coincidence"
mkdir -p "$merge_dir" "$coincidence_dir"

for channel in LISA_A LISA_E; do
    trigger_files=()
    for ((part=0; part<template_parallelization; part++)); do
        trigger_files+=("${search_dir}/matched_filter_snr_peaks_${channel}_${part}.hdf")
    done
    hwilas_trigger_merge \
        --trigger-files "${trigger_files[@]}" \
        --output-file "${merge_dir}/trigger_merge_${channel}.hdf" \
        --cluster-window 7200 \
        --verbose

    hwilas_plot_trigger_merge \
        --trigger-file "${merge_dir}/trigger_merge_${channel}.hdf" \
        --channel "$channel" \
        --output "${merge_dir}/trigger_merge_${channel}.png" \
        --truth-file "$mojito_truth_file" \
        --verbose

done

hwilas_coincidence \
    --trigger-merge-files \
        LISA_A:"${merge_dir}/trigger_merge_LISA_A.hdf" \
        LISA_E:"${merge_dir}/trigger_merge_LISA_E.hdf" \
    --output-file "${coincidence_dir}/coincidence.hdf" \
    --unclustered-output-file "${coincidence_dir}/coincidence_unclustered.hdf" \
    --coincidence-window 3600 \
    --cluster-window 86400 \
    --verbose

hwilas_plot_coincidence \
    --input-file "${coincidence_dir}/coincidence.hdf" \
    --unclustered-input-file "${coincidence_dir}/coincidence_unclustered.hdf" \
    --data-file "$mojito_data_file" \
    --edge-exclusion 172800 \
    --end-exclusion 259200 \
    --foreground-veto-window 86400 \
    --trigger-merge-files \
        LISA_A:"${merge_dir}/trigger_merge_LISA_A.hdf" \
        LISA_E:"${merge_dir}/trigger_merge_LISA_E.hdf" \
    --channels LISA_A LISA_E \
    --cluster-window 86400 \
    --output-directory "$coincidence_dir" \
    --truth-file "$mojito_truth_file" \
    --verbose


# Fit coincident candidates and write their summed signal on the original data grid.
optimise_dir="${OUTPUT_DIR}/round_0/optimise"
subtract_dir="${OUTPUT_DIR}/round_0/subtract"
mkdir -p "$optimise_dir" "$subtract_dir"

hwilas_optimise_snr_peaks \
    --input-file "${coincidence_dir}/coincidence.hdf" \
    --bank-file ../template_banks/output/mojito_bank.hdf \
    --data-files "$mojito_data_file" \
    --psd-files LISA_A:"$psd_file_a" LISA_E:"$psd_file_e" LISA_T:"${psd_dir}/T_mojito_SMOOTHED_PSD.txt" \
    --fit-channels LISA_A LISA_E LISA_T \
    --optimisation-channels LISA_A LISA_E \
    --output-timeseries-channels LISA_A LISA_E LISA_T \
    --output-timeseries-file "${optimise_dir}/optimise.hdf" \
    --output-parameter-file "${optimise_dir}/optimise_parameters.hdf" \
    --tdi-generation 2 \
    --statistic-threshold 40 \
    --minimum-peak-separation 259200 \
    --optimisation-window-before 691200 \
    --optimisation-window-after 86400 \
    --n-particles 256 \
    --n-iterations 75 \
    --n-processes 4 \
    --polish-evaluations 5000 \
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

# Subtract the first-round model before searching for weaker or overlapping signals.
hwilas_subtract_signals \
    --data "$mojito_data_file" \
    --signal-files "${optimise_dir}/optimise.hdf" \
    --output "${subtract_dir}/mojito_data_subtracted.hdf" \
    --channels LISA_A LISA_E LISA_T \
    --verbose

# Plot the first residual with markers for the signals already removed.
hwilas_plot_downsampled_mojito \
    --data-file "${subtract_dir}/mojito_data_subtracted.hdf" \
    --truth-file "$mojito_truth_file" \
    --removed-parameter-files "${optimise_dir}/optimise_parameters.hdf" \
    --end-exclusion 259200 \
    --output-file "${OUTPUT_DIR}/round_1/data.png" \
    --verbose

# Second round: search the residual, merge partitions, then find coincidences.
round1_search_dir="${OUTPUT_DIR}/round_1/matched_filter"
round1_merge_dir="${OUTPUT_DIR}/round_1/trigger_merge"
round1_coincidence_dir="${OUTPUT_DIR}/round_1/coincidence"

mkdir -p "$round1_search_dir" "$round1_merge_dir" "$round1_coincidence_dir"

for channel in LISA_A LISA_E; do
    if [ "$channel" = LISA_A ]; then
        psd_file="$psd_file_a"
    else
        psd_file="$psd_file_e"
    fi

    pids=()
    trigger_files=()
    for ((part=0; part<template_parallelization; part++)); do
        trigger_file="${round1_search_dir}/matched_filter_snr_peaks_${channel}_${part}.hdf"
        trigger_files+=("$trigger_file")
        hwilas_matched_filter_search \
            --input-data-file "${subtract_dir}/mojito_data_subtracted.hdf" \
            --bank-file ../template_banks/output/mojito_bank.hdf \
            --psd-file "$psd_file" \
            --channel "$channel" \
            --waveform-type fastbhb \
            --tdi-generation 2 \
            --padding-time 129600 \
            --end-exclusion 259200 \
            --f-upper 0.09 \
            --snr-threshold 4 \
            --template-fraction-range "${part}/${template_parallelization}" \
            --output-file "$trigger_file" \
            --verbose &
        pids+=("$!")
    done

    status=0
    for pid in "${pids[@]}"; do
        wait "$pid" || status=1
    done

    [ "$status" -eq 0 ] || exit "$status"
    hwilas_trigger_merge \
        --trigger-files "${trigger_files[@]}" \
        --output-file "${round1_merge_dir}/trigger_merge_${channel}.hdf" \
        --cluster-window 7200 \
        --verbose
    hwilas_plot_trigger_merge \
        --trigger-file "${round1_merge_dir}/trigger_merge_${channel}.hdf" \
        --channel "$channel" \
        --output "${round1_merge_dir}/trigger_merge_${channel}.png" \
        --truth-file "$mojito_truth_file" \
        --verbose
done

hwilas_coincidence \
    --trigger-merge-files \
        LISA_A:"${round1_merge_dir}/trigger_merge_LISA_A.hdf" \
        LISA_E:"${round1_merge_dir}/trigger_merge_LISA_E.hdf" \
    --output-file "${round1_coincidence_dir}/coincidence.hdf" \
    --unclustered-output-file "${round1_coincidence_dir}/coincidence_unclustered.hdf" \
    --coincidence-window 3600 \
    --previous-parameter-file "${optimise_dir}/optimise_parameters.hdf" \
    --cluster-window 86400 \
    --verbose

hwilas_plot_coincidence \
    --input-file "${round1_coincidence_dir}/coincidence.hdf" \
    --unclustered-input-file "${round1_coincidence_dir}/coincidence_unclustered.hdf" \
    --data-file "${subtract_dir}/mojito_data_subtracted.hdf" \
    --edge-exclusion 172800 \
    --end-exclusion 259200 \
    --foreground-veto-window 86400 \
    --trigger-merge-files \
        LISA_A:"${round1_merge_dir}/trigger_merge_LISA_A.hdf" \
        LISA_E:"${round1_merge_dir}/trigger_merge_LISA_E.hdf" \
    --channels LISA_A LISA_E \
    --cluster-window 86400 \
    --output-directory "$round1_coincidence_dir" \
    --truth-file "$mojito_truth_file" \
    --verbose

# Fit every surviving second-round coincidence; retain earlier-window flags.
round1_optimise_dir="${OUTPUT_DIR}/round_1/optimise"
mkdir -p "$round1_optimise_dir"

exit

hwilas_optimise_snr_peaks \
    --input-file "${round1_coincidence_dir}/coincidence.hdf" \
    --bank-file ../template_banks/output/mojito_bank.hdf \
    --data-files "${subtract_dir}/mojito_data_subtracted.hdf" \
    --psd-files LISA_A:"$psd_file_a" LISA_E:"$psd_file_e" LISA_T:"${psd_dir}/T_mojito_SMOOTHED_PSD.txt" \
    --fit-channels LISA_A LISA_E LISA_T \
    --optimisation-channels LISA_A LISA_E \
    --output-timeseries-channels LISA_A LISA_E LISA_T \
    --output-timeseries-file "${round1_optimise_dir}/optimise.hdf" \
    --output-parameter-file "${round1_optimise_dir}/optimise_parameters.hdf" \
    --tdi-generation 2 \
    --statistic-threshold 0 \
    --previous-parameter-file "${optimise_dir}/optimise_parameters.hdf" \
    --optimisation-window-before 172800 \
    --optimisation-window-after 172800 \
    --n-particles 256 \
    --n-iterations 75 \
    --n-processes 4 \
    --polish-evaluations 5000 \
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

# Keep the full fitted catalogue and subtract both models once from the original.
final_dir="${OUTPUT_DIR}/final"
all_parameters_file="${final_dir}/all_optimised_parameters.hdf"
final_residual_file="${final_dir}/mojito_residual.hdf"
mkdir -p "$final_dir"
hwilas_combine_search_parameters \
    --input-files "${optimise_dir}/optimise_parameters.hdf" "${round1_optimise_dir}/optimise_parameters.hdf" \
    --output-file "$all_parameters_file" \
    --csv-file "${final_dir}/all_optimised_parameters.csv" \
    --verbose

hwilas_subtract_signals \
    --data "$mojito_data_file" \
    --signal-files "${optimise_dir}/optimise.hdf" "${round1_optimise_dir}/optimise.hdf" \
    --output "$final_residual_file" \
    --channels LISA_A LISA_E LISA_T \
    --verbose

# Plot the final residual with removal markers from both optimisation rounds.
hwilas_plot_downsampled_mojito \
    --data-file "$final_residual_file" \
    --truth-file "$mojito_truth_file" \
    --removed-parameter-files "$all_parameters_file" \
    --end-exclusion 259200 \
    --output-file "${final_dir}/residual_test/data.png" \
    --verbose

# Final residual test: repeat the bank search without fitting further candidates.
residual_search_dir="${OUTPUT_DIR}/final/residual_test/matched_filter"
residual_merge_dir="${OUTPUT_DIR}/final/residual_test/trigger_merge"
residual_coincidence_dir="${OUTPUT_DIR}/final/residual_test/coincidence"

mkdir -p "$residual_search_dir" "$residual_merge_dir" "$residual_coincidence_dir"

for channel in LISA_A LISA_E; do
    if [ "$channel" = LISA_A ]; then
        psd_file="$psd_file_a"
    else
        psd_file="$psd_file_e"
    fi
    pids=()
    trigger_files=()
    for ((part=0; part<template_parallelization; part++)); do
        trigger_file="${residual_search_dir}/matched_filter_snr_peaks_${channel}_${part}.hdf"
        trigger_files+=("$trigger_file")
        hwilas_matched_filter_search \
            --input-data-file "$final_residual_file" \
            --bank-file ../template_banks/output/mojito_bank.hdf \
            --psd-file "$psd_file" \
            --channel "$channel" \
            --waveform-type fastbhb \
            --tdi-generation 2 \
            --padding-time 129600 \
            --end-exclusion 259200 \
            --f-upper 0.09 \
            --snr-threshold 4 \
            --template-fraction-range "${part}/${template_parallelization}" \
            --output-file "$trigger_file" \
            --verbose &
        pids+=("$!")
    done
    status=0
    for pid in "${pids[@]}"; do
        wait "$pid" || status=1
    done
    [ "$status" -eq 0 ] || exit "$status"
    hwilas_trigger_merge \
        --trigger-files "${trigger_files[@]}" \
        --output-file "${residual_merge_dir}/trigger_merge_${channel}.hdf" \
        --cluster-window 7200 \
        --verbose
    hwilas_plot_trigger_merge \
        --trigger-file "${residual_merge_dir}/trigger_merge_${channel}.hdf" \
        --channel "$channel" \
        --output "${residual_merge_dir}/trigger_merge_${channel}.png" \
        --truth-file "$mojito_truth_file" \
        --verbose
done
hwilas_coincidence \
    --trigger-merge-files \
        LISA_A:"${residual_merge_dir}/trigger_merge_LISA_A.hdf" \
        LISA_E:"${residual_merge_dir}/trigger_merge_LISA_E.hdf" \
    --output-file "${residual_coincidence_dir}/coincidence.hdf" \
    --unclustered-output-file "${residual_coincidence_dir}/coincidence_unclustered.hdf" \
    --coincidence-window 3600 \
    --previous-parameter-file "$all_parameters_file" \
    --cluster-window 86400 \
    --verbose
hwilas_plot_coincidence \
    --input-file "${residual_coincidence_dir}/coincidence.hdf" \
    --unclustered-input-file "${residual_coincidence_dir}/coincidence_unclustered.hdf" \
    --data-file "$final_residual_file" \
    --edge-exclusion 172800 \
    --end-exclusion 259200 \
    --foreground-veto-window 86400 \
    --trigger-merge-files \
        LISA_A:"${residual_merge_dir}/trigger_merge_LISA_A.hdf" \
        LISA_E:"${residual_merge_dir}/trigger_merge_LISA_E.hdf" \
    --channels LISA_A LISA_E \
    --cluster-window 86400 \
    --output-directory "$residual_coincidence_dir" \
    --truth-file "$mojito_truth_file" \
    --verbose

# Check subtraction consistency and candidates remaining in the final residual.
hwilas_check_search_residuals \
    --data-file "$mojito_data_file" \
    --residual-file "$final_residual_file" \
    --end-exclusion 259200 \
    --signal-files "${optimise_dir}/optimise.hdf" "${round1_optimise_dir}/optimise.hdf" \
    --parameter-file "$all_parameters_file" \
    --coincidence-file "${residual_coincidence_dir}/coincidence.hdf" \
    --output-file "${final_dir}/residual_report.json" \
    --verbose
