set -e

mkdir -p output

python ./pycbc_brute_bank \
    --verbose \
    --minimal-match 0.9 \
    --tolerance .05 \
    --buffer-length 2592000 \
    --sample-rate .2 \
    --approximant BBHX_PhenomD \
    --tau0-threshold 100000 \
    --tau0-crawl 20000000 \
    --tau0-start 0 \
    --tau0-end  20000000 \
    --tau0-cutoff-frequency 0.0001 \
    --input-config config.ini \
    --seed 1 \
    --output-file output/sangria_bank.hdf \
    --psd-file ../psd_estimation/output/A_sangria_hm_SMOOTHED_PSD.txt \
    --low-frequency-cutoff .000001

hwilas_plot_bank \
  --bank-file output/sangria_bank.hdf \
  --output-plot output/sangria_bank.png \
  --signal-definition-file ../datasets/LDC2_sangria_hm_training.hdf