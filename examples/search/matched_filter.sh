mkdir -p output/TEST/round_0/chunk_0/matched_filter/

templates=(45 46)
channels=(LISA_A LISA_E)

for template in ${templates[@]} ; do
for channel in ${channels[@]} ; do

hwilas_matched_filter_search \
  --verbose --verbose \
  --bank-file ../template_banks/output/sangria_bank.hdf \
  --psd-file ../psd_estimation/output/A_sangria_hm_SMOOTHED_PSD.txt \
  --input-data-file output/round_0/chunk_0/data_processing/LDC2_sangria_hm_training_padded.hdf \
  --channel $channel \
  --template-fraction-range $template/161 \
  --low-frequency-cutoff \
  --waveform-type phenomx_td \
  

  done
  done
