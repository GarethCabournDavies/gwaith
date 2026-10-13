# Mojito two-round search and residual validation

Run `examples/search/mojito_basic_search.sh` from any directory after installing the local package. The script reuses preprocessed data and existing PSDs. Its search stages now use the [joint A/E matched-filter path](mojito_joint_search.md). The current explicit `exit` stops execution before second-round optimisation; later fitting and residual-test stages remain available below it. Existing fitted models must be regenerated to use the physical-distance fit.

## Stages

1. First-round candidates: same-time A/E quadrature SNR selected during matched filtering, then clustered across templates with separation 86400 seconds. Optimisation selects network coincidence SNR ≥40 and saves each actual fit window.
2. Subtract the first-round model with `hwilas_subtract_signals`, preserving cadence, epoch and fractional-frequency metadata.
3. Search the first residual jointly in A/E and cluster the network candidates. The optimiser can flag membership in a first-round optimisation window using `--previous-parameter-file`. This is not a rejection or foreground veto.
4. Optimise second-round coincidences in decreasing coincidence SNR, with `--statistic-threshold 0`; the matched-filter network threshold is set by `network_snr_threshold` (default 6). The first-round SNR-40 selection cut is not applied. Flags are retained in the fitted catalogue and logs.
5. Concatenate both parameter files into `output/mojito/final/all_optimised_parameters.hdf` and `.csv`. `search_round` is zero-based and `round_event_index` identifies the source row. No flagged events or duplicate-looking times are silently discarded.
6. Subtract both round models exactly once from the original data to produce `output/mojito/final/mojito_residual.hdf`.
7. Repeat the bank search on the final residual, without further optimisation, under `output/mojito/final/residual_test`. Produce trigger/coincidence plots and `residual_report.json`, reusing the existing search PSDs.

## Saved fit information

Each optimiser parameter file includes the initial coincidence statistic, initial higher-mode fit SNR, optimised SNR, masses, spins, sky/orientation, coalescence time/phase, estimated distance, amplitude phase, per-event start frequency, original candidate time, actual optimisation-window bounds, and previous-window flags. Mode, TDI, band, orbit and taper settings are stored as attributes; the combined catalogue preserves each round's attributes under `round_metadata`.

The fit solves a positive real amplitude at a reference distance, then saves `distance = reference_distance / amplitude`. It optimises the physical coalescence phase and regenerates A, E and T at that same distance for subtraction. No independent T gain or arbitrary complex waveform rotation is applied; the legacy `amplitude_phase` column is zero. The example uses all three channels in the fit, while coincidence selection uses A/E. Merger time is profiled, and a bounded local refinement follows the swarm; its evaluation count and convergence flag are saved.

Both rounds use 256 particles for 75 iterations and `--n-processes 4`. PySwarms splits particle objective evaluations between four processes. Events run serially in decreasing coincidence SNR order. Phase seeding and the Nelder–Mead refinement remain serial; refinement allows a maximum of 5,000 evaluations and stops early on convergence.

Optional event parallelism remains available through `--event-workers N`, with particle workers set to one. The CLI rejects simultaneous event and particle multiprocessing to avoid nested pools. Event workers use independent HDF5 handles and deterministic per-event seeds; parameter rows and summed models are combined in the original SNR order. Full-length summed models and output writes are handled only by the parent process. A fit already running retains its original arguments.

The window flag uses the actual saved `[optimisation_window_start, optimisation_window_end)` interval, including clipping at data boundaries. The nearest matching previous window is referenced if several overlap. Older parameter files without saved windows use `initial_time` or `tc`, and their saved before/after settings (default ±2 days).

## What the residual test establishes

The final bank search reports candidates remaining after all fitted models are removed. The JSON report separately checks finite data, matched grids and the identity `residual = original - round_0_model - round_1_model`. It records residual RMS, fitted counts per round, the strongest remaining coincidences and how many exceed the diagnostic network-SNR threshold of 40. That diagnostic threshold is not a second-round optimisation cut or a calibrated false-alarm probability. A consistent subtraction identity alone does not establish noise-only residuals.

Residual PSD estimation is optional: pass `--output-plot` and `--psd-files` to `hwilas_check_search_residuals` to request that separate diagnostic. The default Bash workflow does not recalculate PSDs.

## Validation

Focused tests cover flags, selection below SNR 40, catalogue provenance, metadata-preserving subtraction and detection of an incorrect residual. A real FastBHB fixture injects two higher-mode signals 13.64 hours apart with nominal network SNRs 100 and 20. The first fit/subtraction is followed by a flagged second-round fit with no SNR-40 cut, a combined two-event catalogue, a final matched-filter/coincidence scan and a residual report/PSD plot. Less than 2% of the injected signal energy remains in the model error relative to the known noise, and no final coincidence exceeds the diagnostic threshold of 40.

The full Mojito two-round job has not been executed as part of these code checks.

Targeted physical fits improved A/E residuals near days 173, 264 and 300, but the day-173 fit still predicts excess T power. These checks do not establish a clean Mojito residual or validate the forward-model T convention against the original injection generator.

## Data plots at each round

The script writes `round_0/data.png` for the original data, `round_1/data.png` for the first residual, and `final/residual_test/data.png` for the final residual. Dashed red `axvline` markers show all truth injection times; dotted black markers show the fitted times of signals already removed. Both are full-height and behind the channel traces. Plotting decimation scales the displayed cadence so times stay aligned.

The example discards the final three days before filtering and from data plots. The two-day template-support guard also excludes triggers at the retained end, so the last five days have no triggers.
