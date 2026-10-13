# Process and sum the noiseless Mojito sources

Activate the HPC environment and install the updated checkout to register the new executables:

```bash
python -m pip install -e .
bash examples/data_processing/process_mojito_clean_data.sh
```

The script defaults to the supplied HPC MBHB/L1 directory, the Mojito catalogue and `output/mojito`. Override those paths with three positional arguments:

```bash
bash examples/data_processing/process_mojito_clean_data.sh \
    /path/to/MBHB/L1 /path/to/catalogue.hdf5 /path/to/output/mojito
```

The planner identifies each source from its filename's `sourceN` token and matches it to catalogue `ID`. It requires one raw file per catalogue source and rejects duplicate versions. `TimeCoalescencePhenomTPHMSSBFrame − 800 s` defines merger time on the data axis; `catalogue_time_offset` is explicit in the Bash script.

For each source, a 122-day chunk is processed only when:

```text
chunk end > merger − 40 days
chunk start < merger + 20 days
```

The complete selected chunk is processed with `hwilas_process_mojito_data_chunk`, including its existing filtering overlap, at 4-second A/E/T cadence. Other chunks are skipped. Output names preserve `sourceN` and the chunk index. Change `before_merger_days`, `after_merger_days`, `chunk_duration` or `total_data_duration` near the top of the script if needed.

`hwilas_process_data_combine` then creates `output/mojito/mojito_clean_summed.hdf` on an explicit 0–731-day grid. It adds each input chunk at its stored epoch, sums overlaps across sources and leaves uncovered intervals zero. Addition streams bounded blocks; it does not allocate a full-mission array for every source. The output preserves cadence and frequency-unit metadata and records its input files and completion status. Omitting distant chunks is the requested approximation; it does not prove a source has exactly zero waveform outside the retained chunks. The same executable combines the non-overlapping noisy-data chunks in `process_mojito_data.sh`.

The summation command also accepts full-source files from the original proposed loop:

```bash
hwilas_process_data_combine \
    --data-file source0_downsampled.hdf source1_downsampled.hdf \
    --output-file mojito_clean_summed.hdf \
    --start-time 0 --end-time 63158400 --delta-t 4 \
    --channels LISA_A LISA_E LISA_T
```

Without explicit start/end times, the output covers the union of the inputs. Explicit times provide zero padding and can clip inputs to a chosen range. Cadence, sample-grid origin, channels and units must agree. Existing output files can be overwritten; an input cannot also be the output.

Use `--multiplication-factors` (or `--multiplication-factor`) to scale inputs before adding them. An omitted or empty list means 1 for every file; one value applies to all files; otherwise the list must have exactly one value per file in `--data-file` order. Negative factors subtract and zero factors omit a file's contribution. For example, `--data-file original.hdf model.hdf --multiplication-factors 1 -1` combines data minus model on the chosen grid. Factors are recorded in the output metadata.

Copy the summed file locally and set `NOISELESS_FILE` in `examples/search/mojito_residual_diagnostics.ipynb` to its local path. Its metadata marks fractional-frequency units, so the notebook does not divide by the laser frequency again. The combined file contains all sources; its noiseless residual around an event can therefore include neighbouring signals.

Local tests exercise the real planner and summation CLI, including the complete Bash workflow with raw downsampling stubbed. Processing the real HPC raw files has not been run locally.
