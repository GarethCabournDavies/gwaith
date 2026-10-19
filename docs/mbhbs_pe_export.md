# PE starting values

`hwilas_export_pe_seed` writes one named YAML containing starting parameters,
units, waveform/reference conventions and provenance. It does not supply priors,
a sampler, a likelihood, an observation window or waveform-generation settings.
External teams choose these, including whether to construct priors from the seed.
Native search settings remain in provenance to explain the result, not prescribe PE.

```bash
uv run --no-sync python -m tools.search.export_pe_seed \
  --parameters examples/search/output/mojito/final/all_optimised_parameters.hdf \
  --row-index 0 --target-time-origin 12345678.0 \
  --native-phase-convention pi-over-2-minus-phase \
  --out /private/tmp/hwilas-row0-starting-values.yaml
```

The installed command is `hwilas_export_pe_seed`. Use the prepared uv environment
(`UV_PROJECT_ENVIRONMENT` if needed); `--no-sync` avoids resolving private Git
sources. `--row-index` is zero-based, in descending-statistic order matching the
final CSV, with stable tie ordering. IDs are not used for selection. Existing
output files are overwritten.

`examples/search/mojito_basic_search.sh` counts CSV records excluding the header
and invokes the command once per record, writing `final/pe/seed_<row-index>.yaml`.
The CSV and HDF5 must be matching, unmodified consolidated outputs. The number of
rows may vary. Override `PE_TIME_ORIGIN`, `PE_NATIVE_PHASE_CONVENTION` and
`PE_EXPORT_DIR` as needed. Time origin defaults to `DATA_TIME_ORIGIN`.
No PE or sampler template is required. A failed export stops the loop; completed
files remain available.

## Conventions

Masses are detector/SSB-frame solar masses, with `m1 >= m2` and spins attached to
their components; unordered components are rejected. No redshift conversion is
performed. MBHBS sampling uses `q=m2/m1 <= 1`; Hwilas bank `q` uses the reciprocal.
The exported ratio is derived from component masses, never a stored `q` column.
Distance remains Mpc in YAML. Receiving teams must convert it to metres exactly
once if their waveform interface requires SI distance.
Sky coordinates remain SSB ecliptic longitude/latitude, not RA/declination or
LISA-frame angles. Inclination and polarization remain physical radians;
longitude/phase wrap at 2pi and polarization at pi.

`t_ref = tc + native data_time_origin - target_time_origin`. The caller must
provide the absolute orbit seconds represented by PE time zero. This changes
the coordinate origin, not the waveform reference anchor. ESA trajectories must
use that same origin. The saved search window remains native provenance; no PE observation window is chosen.

For **IMRPhenomTHM**, physical intrinsic parameters and orientation transfer as
starting parameters. `coa_phase` is orbital phase at the saved positive `f_ref`;
`tc` is THM merger time. The harmonic azimuth convention gives a candidate BBHx
phase `pi/2-coa_phase`. The exporter uses that candidate and the shifted merger
time at BBHx's automatic `f_ref=0` anchor. **Neither is an exact cross-model
reference conversion**: both require PE refinement. It does not run BBHx's
same-model reference reanchoring on a THM phase/time or claim THM/PhenomHM
waveform equivalence. Native phase, frequency and time are preserved verbatim.

The local final results currently record `waveform_model=bbhx`, `orbit=esa`,
`f_ref=0`, rather than THM. External BBHx providers require an explicit phase
convention because the entry-point name alone cannot establish their mapping.
The inspected local provider passes `pi/2-coa_phase`, `tc` and `f_ref` to BBHx;
the command above declares that mapping. `direct` is available for providers
that pass phase unchanged. BBHx reference frequency is retained. No equivalence
validation is inferred from that declaration.

Every native event column and selected round attribute is retained, with an
input SHA256 hash. Exported values use BBHx parameter names at the top level;
`provenance` preserves native values, units and the approximate-reference warning.
Catalogue truth remains separate (`truth: null`); the seed is never truth.
The target metadata describes the BBHx/PhenomHM + ESA convention only, not an
enabled PE pipeline. No data, PSDs or orbit files are exported.

## Validation and receiving-team responsibilities

```bash
MBHBS_SOURCE="$PWD/../mbhbs/src" uv run --no-sync python -m pytest -q \
  tests/filter/test_pe_export.py tests/examples/test_search_execution.py
```

The integration check exercises MBHBS's real `get_bbhx_parameter_dict` after the
explicit Mpc-to-metre conversion. This seed is intentionally not a complete
injection, PE or sampler configuration and is not passed off as one. Receiving
teams must combine starting values with their own configurations; no standalone
MBHBS PE invocation is claimed for this file. MBHBS retains responsibility for
raw Mojito loading, processing and PSD construction.

The inspected MBHBS Sequince runner simulates injections and its generic model
uses equal-arm orbits. Its raw-Mojito runner uses Nessai. The combined raw-Mojito,
ESA, Sequince and separate truth/seed plotting capability remains a PE-side gap.
The earlier `codex/mojito-hwilas-seeded-pe` branch was inspected for context;
its adapter/processing modules are not dependencies. MBHBS is untouched and no
PE run is claimed.
