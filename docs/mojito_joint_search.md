# Joint A/E matched-filter search

`examples/search/mojito_basic_search.sh` filters LISA_A and LISA_E in the same
template pass, using their individual PSDs. It computes
`sqrt(abs(rho_A(t))**2 + abs(rho_E(t))**2)` before selecting peaks. Both channels
refer to the same template and sample time. There is no separate one-hour
peak-pairing tolerance and no requirement that each channel cross a threshold.
The statistic maximises channel phases independently; it is not a coherent
common-phase likelihood.

The Bash variable `network_statistic_threshold=6` sets the network peak threshold.
The first-round optimisation cut remains 40. Per-template peak separation
remains two hours; `hwilas_trigger_merge --cluster-across-templates
--cluster-window 86400` then selects the loudest event across templates within
a day. The unclustered file preserves all selected network peaks before that
cross-template clustering.

Example matched-filter arguments:

```bash
--channels LISA_A LISA_E \
--psd-files LISA_A:A_mojito_SMOOTHED_PSD.txt LISA_E:E_mojito_SMOOTHED_PSD.txt
```

The single-channel `--channel` interface remains available. Joint files contain
`template_id`, `end_time`, network `statistic`, and per-channel `statistic_*`, `time_*`,
`complex_snr_*`, and `trigger_id_*` columns. The complex channel values retain
the phase information at each selected network peak. Full SNR timeseries are
not saved. Merging preserves these columns and makes the trigger IDs unique
across partitions. The optimiser accepts this schema directly.

All three search calls (initial, first residual, final residual) use the joint
path. The existing `exit` before second-round optimisation remains in the Bash
script; remove it when ready to execute the later stages. Existing PSD and
preprocessing steps retain their commented status.

## Significance

Candidate SNR is not a calibrated false-alarm probability. The current workflow
plots unclustered network peaks with clustered candidates highlighted in one
figure. It does not produce channel foreground/background SNR histograms.

A future calibration can form the same network statistic after shifting one
channel's SNR series by many nonphysical time offsets, then apply the same
thresholds and clustering and measure the resulting event rate. That requires
accounting for usable overlap time, signal contamination, channel-noise
correlations, and the template trials. Alternatively, simulated noise or
signal-removed data can be run through the same pipeline. Per-channel values
also leave room for a later consistency-weighted statistic. No time-slide
calibration or false-alarm rates are implemented by this change.

The combined plot is produced by `hwilas_plot_trigger_merge --trigger-file
candidates_unclustered.hdf --clustered-trigger-file candidates.hdf
--network-statistic --output candidates.png`. Truth lines remain behind the data.

## Sharing candidate results

Each trigger-merge call writes `candidates.csv` beside the clustered HDF5 file,
sorted by decreasing SNR. Columns include candidate rank, time in seconds/days,
SNR, template ID, channel SNRs/times and channel phases. Supplying `--bank-file`
adds explicitly labelled template masses, chirp mass, spins, effective spin,
angles and start frequency. These are bank parameters, not optimised estimates.
Use the separate `hwilas_candidate_hdf_to_csv --input-file HDF --output-file CSV`
job after trigger merging or optimisation. `--bank-file BANK` optionally adds
template parameters to a candidate export. Optimisation exports preserve fitted
parameters, initial/final SNRs, convergence flags and original fit-row identity.
All clustered candidates are exported;
the first-round optimisation SNR-40 cut is a separate selection.

Ranking datasets and CSV columns use `statistic` and `statistic_CHANNEL`. The
CLI uses `--statistic-threshold`; old `--snr-threshold` flags and SNR-named
HDF5 files remain readable. Raw `complex_snr_CHANNEL` measurements retain their
physical meaning. Optimisation saves `initial_stat`, `initial_fit_stat` and
`optimised_stat`. No numeric statistic formula is changed by this naming migration.
