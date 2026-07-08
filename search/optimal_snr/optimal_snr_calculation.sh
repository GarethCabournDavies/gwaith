python optimal_snr_calculation.py \
  --data-file ../../datasets/LDC2_sangria_hm_training.hdf \
  --bank-file ../../template_banks/output/lisa_bank_full.hdf \
  --psd-files \
   LISA_A:../../psd_estimation/A_sangria_hm_SMOOTHED_PSD.txt \
   LISA_E:../../psd_estimation/A_sangria_hm_SMOOTHED_PSD.txt \
  --output-file optimal_snr_fitting_factors.hdf \
  --verbose