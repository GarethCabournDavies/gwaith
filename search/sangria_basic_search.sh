set -e

# mkdir -p output/data_processing 

# python process_sangria_data.py \
#   --data-file ../datasets/LDC2_sangria_hm_training.hdf \
#   --psd-file ../psd_estimation/A_sangria_hm_SMOOTHED_PSD.txt \
#   --output-file output/data_processing/LDC2_sangria_hm_training_padded.hdf \
#   --padding-duration 2592000 \
#   --verbose \
#   --testing-plots output/data_processing 

# template_parallelization=15
# mkdir -p output/matched_filter
# for template_range in $(seq 0 $((${template_parallelization} - 1))) ; do
#   echo ${template_range}/${template_parallelization}
#   echo "Channel A"
#   python matched_filter.py \
#     --start-time 0 \
#     --end-time 6280000 \
#     --bank-file ../template_banks/output/sangria_bank.hdf \
#     --psd-file ../psd_estimation/A_sangria_hm_SMOOTHED_PSD.txt \
#     --input-data-file output/data_processing/LDC2_sangria_hm_training_padded.hdf \
#     --channel LISA_A \
#     --output-file output/matched_filter/matched_filter_snr_peaks_A_${template_range}.hdf \
#     --template-fraction-range ${template_range}/${template_parallelization} \
#     --testing-plots output/matched_filter

#   echo "Channel E"
#   python matched_filter.py \
#   --start-time 0 \
#   --end-time 6280000 \
#   --bank-file ../template_banks/output/sangria_bank.hdf \
#   --psd-file ../psd_estimation/A_sangria_hm_SMOOTHED_PSD.txt \
#   --input-data-file output/data_processing/LDC2_sangria_hm_training_padded.hdf \
#   --channel LISA_E \
#   --output-file output/matched_filter/matched_filter_snr_peaks_E_${template_range}.hdf \
#   --template-fraction-range ${template_range}/${template_parallelization} \
#   --testing-plots output/matched_filter

# done

# files_A=$( ls $(pwd)/output/matched_filter/matched_filter_snr_peaks_A_*.hdf )
# files_E=$( ls $(pwd)/output/matched_filter/matched_filter_snr_peaks_E_*.hdf )

# mkdir -p output/trigger_merge
# python trigger_merge.py \
#   --verbose \
#   --trigger-files $files_A \
#   --output-file output/trigger_merge/trigger_merge_A.hdf \
#   --testing-plot output/trigger_merge/trigger_merge_A.png 

# python trigger_merge.py \
#   --verbose \
#   --trigger-files $files_E \
#   --output-file output/trigger_merge_E.hdf \
#   --testing-plot output/trigger_merge_E.png 

# mkdir -p output/coincidence

# python coincidence.py \
#   --trigger-merge-files \
#     LISA_A:output/trigger_merge/trigger_merge_A.hdf \
#     LISA_E:output/trigger_merge/trigger_merge_E.hdf \
#   --output-file output/coincidence/coincidence.hdf \
#   --coincidence-window 3600 \
#   --testing-plots output/coincidence \
#   --expected-snr-file optimal_snr/optimal_snr_fitting_factors.hdf

# mkdir -p output/optimise

python optimise_snr_peaks.py \
  --input-file output/coincidence/coincidence.hdf \
  --output-file output/optimise/optimise.hdf \
  --data-file output/data_processing/LDC2_sangria_hm_training_padded.hdf \
  --testing-plots output/optimise \
  --psd-file ../psd_estimation/A_sangria_hm_SMOOTHED_PSD.txt \
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