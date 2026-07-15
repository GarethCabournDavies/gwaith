set -e

# mkdir -p output/data_processing 

# hwilas_process_sangria_data \
#   --data-file ../datasets/LDC2_sangria_hm_training.hdf \
#   --psd-file ../psd_estimation/A_sangria_hm_SMOOTHED_PSD.txt \
#   --output-file output/data_processing/LDC2_sangria_hm_training_padded.hdf \
#   --padding-duration 2592000 \
#   --verbose \
#   --testing-plots output/data_processing 

# template_parallelization=15
# mkdir -p output/round_0/matched_filter
# for template_range in $(seq 0 $((${template_parallelization} - 1))) ; do
#   echo ${template_range}/${template_parallelization}
#   echo "Channel A"
#   hwilas_matched_filter \
#     --start-time 0 \
#     --end-time 6280000 \
#     --bank-file ../template_banks/output/sangria_bank.hdf \
#     --psd-file ../psd_estimation/output/A_sangria_hm_SMOOTHED_PSD.txt \
#     --input-data-file output/data_processing/LDC2_sangria_hm_training_padded.hdf \
#     --channel LISA_A \
#     --output-file output/round_0/matched_filter/matched_filter_snr_peaks_A_${template_range}.hdf \
#     --template-fraction-range ${template_range}/${template_parallelization} \
#     --testing-plots output/round_0/matched_filter

#   echo "Channel E"
#   hwilas_matched_filter \
#   --start-time 0 \
#   --end-time 6280000 \
#   --bank-file ../template_banks/output/sangria_bank.hdf \
#   --psd-file ../psd_estimation/A_sangria_hm_SMOOTHED_PSD.txt \
#   --input-data-file output/data_processing/LDC2_sangria_hm_training_padded.hdf \
#   --channel LISA_E \
#   --output-file output/round_0/matched_filter/matched_filter_snr_peaks_E_${template_range}.hdf \
#   --template-fraction-range ${template_range}/${template_parallelization} \
#   --testing-plots output/round_0/matched_filter

# done

# files_A=$( ls $(pwd)/output/round_0/matched_filter/matched_filter_snr_peaks_A_*.hdf )
# files_E=$( ls $(pwd)/output/round_0/matched_filter/matched_filter_snr_peaks_E_*.hdf )

# mkdir -p output/round_0/trigger_merge
# hwilas_trigger_merge \
#   --verbose \
#   --trigger-files $files_A \
#   --output-file output/round_0/trigger_merge/trigger_merge_A.hdf \
#   --testing-plot output/round_0/trigger_merge/trigger_merge_A.png 

# hwilas_trigger_merge \
#   --verbose \
#   --trigger-files $files_E \
#   --output-file output/round_0/trigger_merge/trigger_merge_E.hdf \
#   --testing-plot output/round_0/trigger_merge/trigger_merge_E.png 

# mkdir -p output/round_0/coincidence

# hwilas_coincidence \
#   --trigger-merge-files \
#     LISA_A:output/round_0/trigger_merge/trigger_merge_A.hdf \
#     LISA_E:output/round_0/trigger_merge/trigger_merge_E.hdf \
#   --output-file output/round_0/coincidence/coincidence.hdf \
#   --coincidence-window 3600 \
#   --testing-plots output/round_0/coincidence \
#   --expected-snr-file optimal_snr/optimal_snr_fitting_factors.hdf

mkdir -p output/round_0/optimise

hwilas_optimise_snr_peaks \
  --input-file output/round_0/coincidence/coincidence.hdf \
  --signal-output-file output/round_0/optimise/optimise.hdf \
  --signal-parameter-file output/round_0/optimise/optimise_parameters.hdf \
  --data-file output/data_processing/LDC2_sangria_hm_training_padded.hdf \
  --testing-plots output/round_0/optimise \
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
    eclipticlongitude:6.283 \
    eclipticlatitude:1.5707 \
    inclination:3.1416 \
    polarization:3.1416 \
  --bounds-window-relative-to-initial \
    log_mchirp:0.1 \
    tc:3600

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