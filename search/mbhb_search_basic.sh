set -e

python inpaint_gaps_and_pads.py \
  --data-file ../datasets/LDC2_sangria_hm_training.hdf \
  --psd-file ../psd_estimation/A_sangria_hm_SMOOTHED_PSD.txt \
  --output-file output/LDC2_sangria_hm_training_padded.hdf \
  --padding-duration 2592000 \
  --verbose \
  --testing-plots

# template_parallelization=10
# for template_range in $(seq 0 $((${template_parallelization} - 1))) ; do
#   echo ${template_range}/${template_parallelization}
#   echo "Channel A"
#   python matched_filter.py \
#     --start-time 0 \
#     --end-time 6280000 \
#     --bank-file ../template_banks/output/lisa_bank_full.hdf \
#     --psd-file ../psd_estimation/A_sangria_hm_SMOOTHED_PSD.txt \
#     --input-data-file ../datasets/LDC2_sangria_hm_training.hdf \
#     --channel LISA_A \
#     --output-file output/matched_filter_snr_peaks_A_${template_range}.hdf \
#     --template-fraction-range ${template_range}/${template_parallelization} \
#     --testing-plots

#   echo "Channel E"
#   python matched_filter.py \
#   --start-time 0 \
#   --end-time 6280000 \
#   --bank-file ../template_banks/output/lisa_bank_full.hdf \
#   --psd-file ../psd_estimation/A_sangria_hm_SMOOTHED_PSD.txt \
#   --input-data-file ../datasets/LDC2_sangria_hm_training.hdf \
#   --channel LISA_E \
#   --output-file output/matched_filter_snr_peaks_E_${template_range}.hdf \
#   --template-fraction-range ${template_range}/${template_parallelization} \
#     --testing-plots


# done

# files_A=$( ls $(pwd)/output/matched_filter_snr_peaks_A_*.hdf )
# files_E=$( ls $(pwd)/output/matched_filter_snr_peaks_E_*.hdf )

# python trigger_merge.py \
#   --verbose \
#   --trigger-files $files_A \
#   --output-file output/trigger_merge_A.hdf \
#   --testing-plot output/trigger_merge_A.png 

# python trigger_merge.py \
#   --verbose \
#   --trigger-files $files_E \
#   --output-file output/trigger_merge_E.hdf \
#   --testing-plot output/trigger_merge_E.png 

# python coincidence.py \
#   --trigger-merge-files \
#     LISA_A:output/trigger_merge_A.hdf \
#     LISA_E:output/trigger_merge_E.hdf \
#   --output-file output/coincidence.hdf \
#   --coincidence-window 3600 \
#   --testing-plots \
#   --expected-snr-file optimal_snr/optimal_snr_fitting_factors.hdf \
#   --num-timeslides 10