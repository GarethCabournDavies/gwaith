# Stochastic template banks

For Sangria, run `bash make_sangria_bank.sh` from `examples/template_banks`.
It uses `hwilas_make_stochastic_bank --config sangria_bank.toml --nproc 4 --seed 1`
and plots the resulting bank. The config retains the example's uniform total
mass range 1e6–2e7, q range 1–4, spin ranges −0.98–0.98, minimal match 0.9 and
acceptance tolerance 0.05. Placement uses FastBHB (2,2), TDI-1.5 and one-day
support at 5-second cadence, replacing the BBHx/PyCBC 30-day placement path.
Both A/E PSDs must cover the configured band. Validate a generated bank before
using it for a search; the full Sangria placement has not been run during cleanup.

## Local Mojito stochastic bank

Activate `env_hwilas_search`, then run from any directory:

```bash
bash /Users/gareth/lisa/mbhbs_search/examples/template_banks/make_mojito_bank.sh
```

The script uses the new `tools.template_banks.stochastic_bank` builder and
`mojito_bank.toml`. It generates FastBHB IMRPhenomHM (2,2), TDI-2 templates with
per-template cutoffs retaining 99% of network SNR, using both smoothed Mojito A/E PSDs from
`examples/search/output/mojito/round_0/psd/`. These PSDs must exist first.

Each proposal gets its own `f_lower`: integrate the full (2,2) FastBHB A/E
response weighted by the channel PSDs, and discard at most 1−0.99² of SNR²
below the cutoff. `waveform.f_lower` is the analysis floor. Placement evaluates
frequency-domain templates on a common grid, so inspiral duration does not
truncate the placement waveforms. Accepted cutoffs and IMRPhenomD inspiral
durations are saved in `f_lower` and `pre_merger_seconds`. The matched-filter
search derives each template's support from its cutoff, adds a taper buffer
and one day after merger, and uses that template's merger index for alignment.
Existing banks must be rebuilt to adopt the 99% criterion; their saved cutoffs
still determine their search durations. The percentage describes frequency-domain
SNR retention, before finite-window tapering.

Edit the TOML file to change parameter bounds/distributions, placement controls,
PSD paths/units, seed, or output. Paths resolve against the TOML file. Default
coverage: log-uniform total mass 3e5–5e7 solar masses, uniform q=1–10 and aligned
spins -0.99–0.99, with isotropic sky and inclination.

Outputs under `examples/template_banks/output/`:

- `mojito_bank.hdf`: bank and configuration/placement metadata.
- `mojito_bank.png`: parameter-distribution plot.

Progress and the stopping report are printed directly to the terminal.

The script specifies the config, four workers, and output paths directly.
Edit those command arguments if needed. When changing `bank.output_file` in the
config, update the plotting paths in the script too.

The builder overwrites the output bank on each fresh run. The example uses
`seed = 1` for a repeatable proposal sequence with unchanged inputs; set
`bank.resume` explicitly to top up an existing bank. A proposal-limit stop reports `converged=false`; validate coverage
with independent draws before treating the bank as complete:

```bash
python -m tools.template_banks.verify_bank validate \
  --config examples/template_banks/mojito_bank.toml \
  --bank-file examples/template_banks/output/mojito_bank.hdf \
  --output-file examples/template_banks/output/mojito_bank_validation.json \
  --n 400 --nproc 4
```

To use the generated bank in the search, set `search.bank_file` in
`examples/search/mojito_local.toml` to
`../template_banks/output/mojito_bank.hdf`, and use a fresh search output directory.
Stop a foreground bank run with Ctrl-C; batch checkpoints are saved in the bank.

For large independent injection campaigns, use the cached verifier (no PyCBC):

```bash
hwilas_verify_bank prepare \
  --config examples/template_banks/mojito_bank.toml \
  --bank-file examples/template_banks/output/mojito_bank.hdf \
  --cache-dir /shared/verification/cache
hwilas_verify_bank run \
  --cache-dir /shared/verification/cache --n 200000 --seed 12345 \
  --shard-count 200 --shard-index 0 --nproc 4 \
  --output-file /shared/verification/shard-0.json
# Run all shard indices 0..199 before merging:
hwilas_verify_bank merge \
  --output-file /shared/verification/coverage.json /shared/verification/shard-*.json
hwilas_plot_bank_verification \
  --output-dir /shared/verification/plots \
  /shared/verification/coverage.json /shared/verification/shard-*.json
```

`prepare` requires a new empty cache directory. Bank spectra are generated once;
workers memory-map the cache. Cache storage is approximately
`8 * 2 * templates * (round(172800 / delta_t) / 2 + 1)` bytes
(about 346 kB per template at 4-second cadence). Each worker additionally holds
FFT buffers proportional to `--template-batch` (default 64). Reduce workers or
batch size if memory is constrained. PSD paths must remain accessible and unchanged.

Every injection uses an independent seed derived from the campaign seed and its
global index, so changing the number of shards/workers preserves the population.
Injection parameters, best template row (zero-based), fitting factor, failure,
and evaluated/rejected template counts are saved. Each template is considered;
a triangle-inequality upper bound skips FFTs only when it cannot improve the
current best match, with a float32 roundoff margin. `--no-rejector` provides an
exhaustive FFT baseline. There is no match-threshold early exit or mass-window
cut. The bound may save little for overlapping spectra; benchmark before scaling.

The metric intentionally matches stochastic placement: the (2,2) waveform,
two-day buffer, configured lag window, PSDs, and independently maximized A/E
time/phase overlaps, weighted by injection channel SNR squared. It is not a
coherent network match with a shared lag/phase. This tests coverage of that bank
model and prior, without detector noise, higher-mode/model-systematic tests, or
a detection-efficiency calculation. The configured prior can be broadened at
cache preparation to test another population.

The merger rejects missing, duplicate, out-of-range, or incompatible injections.
It reports the fraction below minimal match, minimum, quantiles (1%, 5%, 50%,
95%), and generation failures. A report with failures is marked incomplete and
the command exits unsuccessfully; failed draws are not treated as covered.

The example workflow follows the style of `sangria_basic_search.sh`: editable
settings at the top, a `RUN_MODE="slurm"` / `"bash"` switch, and four numbered
steps calling the named executables directly. Run from `examples/template_banks`
with an activated hwilas environment:

```bash
cd examples/template_banks
bash verify_bank_slurm.sh
```

Edit `CONFIG_FILE`, `BANK_FILE`, `OUTPUT_DIR`, `number_of_injections`,
`injection_batches`, `workers_per_batch`, and `max_parallel_batches` in the script.
Optional arguments override the three paths: `bash verify_bank_slurm.sh CONFIG
BANK OUTPUT_DIR`. Refresh installed command entry points after updating the
checkout (`pip install -e . --no-deps` from the repository root).

1. Prepare the waveform cache with `hwilas_verify_bank prepare`.
2. Submit the injection array using `verify_bank_injections.sbatch`, or run each
   batch locally in bash mode.
3. Collate completed batches with `hwilas_verify_bank merge`.
4. Plot the complete campaign and each batch with `hwilas_plot_bank_verification`.

Slurm mode uses `sbatch --wait`, so keep the workflow running (for example in
`tmux`) until the array completes. Preparation, collation, and plotting run where
you launch the workflow; only injection recovery is submitted to compute nodes.
A failed Slurm task stops the workflow before collation and plotting. Bash mode
runs batches sequentially with the configured workers per batch.
Set `SBATCH_PARTITION`, `SBATCH_TIME`, and `SBATCH_MEM` for the cluster. Repository,
PSDs, and output storage must be shared with compute nodes. Slurm prints the job
ID; monitor with `squeue -j JOB_ID`, cancel with `scancel JOB_ID`, and inspect
`logs/injections-JOB_ID_TASK_ID.log`. Use a fresh output directory per campaign.

Plots are saved to `plots/coverage.png` and `plots/shard-INDEX.png`. Each shows
a match histogram and cumulative distribution, the minimal-match threshold,
the percentage below that threshold, and successful/failed injection counts.
Failed injections are excluded from match distributions and counted explicitly.
Use `hwilas_plot_bank_verification --bins 100 --output-dir ... REPORT...` to
change the histogram resolution or plot individual reports, including failed
batches, manually.
