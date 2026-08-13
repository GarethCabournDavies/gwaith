hwilas_optimal_snr \
  --data-file ../../datasets/LDC2_sangria_hm_training.hdf \
  --bank-file ../../template_banks/output/sangria_bank.hdf \
  --psd-files \
   LISA_A:../../psd_estimation/output/A_sangria_hm_SMOOTHED_PSD.txt \
   LISA_E:../../psd_estimation/output/E_sangria_hm_SMOOTHED_PSD.txt \
  --output-file optimal_snr_fitting_factors.hdf \
  --verbose