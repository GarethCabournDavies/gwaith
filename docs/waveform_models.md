# Detector waveform models

`hwilas.waveforms.get_fd_waveform` returns `gwaith.types.TimeSeries` for
`FastBHB`, `PhenomTHM`, and independently installed entry-point providers. Each provider returns time-domain detector strain. Models load their dependencies
only when selected.

```python
from hwilas.waveforms import get_fd_waveform

wave = get_fd_waveform(
    'PhenomTHM', tdi='2', delta_t=4., start_time=0., length=512,
    mass1=1e6, mass2=5e5, spin1z=.2, spin2z=-.1, distance=10000.,
    inclination=1., polarization=.5, eclipticlongitude=1.2,
    eclipticlatitude=.3, coa_phase=.4, f_lower=1e-4, f_ref=2e-4, tc=800.,
    ifos=['LISA_A', 'LISA_E'],
)
```

Masses are in solar masses, distance in Mpc, times in seconds, and angles in
radians. `length` counts time samples, including odd sample counts. Alternatively
provide `end_time` as an exclusive endpoint. TDI accepts `1.5`, `2`, and `2.0`.
Select XYZ and/or AET channels with `ifos`.

## PhenomTHM + TDISpline

This ports the waveform construction in MBHBS's
`sandbox/td_explore/scripts/injection_pe_td.py`: PhenomXPy **IMRPhenomTHM** modes,
spin-weighted harmonics, amplitude/phase splines, and a TDISpline response.
It is a different intrinsic approximant from frequency-domain IMRPhenomHM.
The adapter targets PhenomXPy 2.0.3.

Supply physical SSB inclination and polarization, rather than raw catalogue
orbital angular momentum angles. `coa_phase` (or `phi_ref`) is orbital phase at
the positive `f_ref`; its default is `f_lower`. Ecliptic sky and polarization
bases are rotated together into the response's equatorial coordinates.
PhenomTHM does not use FastBHB's observer-azimuth convention.

The adapter targets `tdispline.numpy`. It applies polarization to the complex
modes and passes negative unwrapped complex phase to match `exp(-i phase)`.
`source_oversampling=4` controls the uniform source cadence; `response_nodes=4096`
controls response interpolation. Check convergence for your sampling/band.

The default analytic orbit uses LISAOrbits' Keplerian geometry and its default
orbital phase. It does not reproduce the Mojito FastBHB orbit phase automatically.
`orbit='oem'` uses the included ESA trailing ephemeris; set `data_time_origin` so
the padded detector times fall within that ephemeris. An orbit object may be
supplied with equatorial `get_positions` and `get_ltts` methods. The default
1000-second source padding covers retarded response times.

`hwilas.waveforms.unified_generation.generate_waveform` also accepts these
model names and returns arrays on its supplied uniform detector time grid.
The matched-filter search also accepts registered external providers through `--approximant`.

## BBHx licensing

Upstream BBHx's [master LICENSE](https://github.com/lisa-analysis-tools/BBHx/blob/master/LICENSE)
contains GPL version 3, as does the UK LISA `fix-barycentre-delay` branch
(commit `2725d96d56365422a93edee8bd5b8c6a31a55faa`). This is not a license
introduced by that branch. The upstream README and PyPI description say MIT,
which conflicts with the actual LICENSE file; do not rely on those summaries
as evidence of a permissive license. Historical releases must be checked
individually if their licensing matters.

There is no GPL exception for BBHx in hwilas. BBHx is not a bundled model or
project/environment dependency. The BBHx provider is installed as a separate
external distribution. Entry-point separation describes the software boundary;
it does not establish that a combined installation has no GPL obligations.

## External waveform entry points

A separate installed Python distribution can register a callable:

```toml
[project.entry-points."hwilas.waveforms"]
ExampleModel = "example_waveforms.generator:get_td_waveform"
```

The provider implements this interface:

```python
def get_td_waveform(approximant, tdi='2', **params):
    # Generate physical detector strain on the requested uniform sampling grid.
    # Return {channel_name: gwaith.types.TimeSeries(...)}.
    ...
```

`get_fd_waveform('ExampleModel', **params)` dispatches to this function.
Names are case-insensitive; duplicate names and built-in aliases are rejected.
Discovery reads installed package metadata without importing providers. Only
the selected provider is loaded; its import or generation errors propagate.
`hwilas.waveforms.available_approximants()` lists built-in canonical names and
registered external names without loading models.

Providers receive `approximant`, `tdi`, and the supplied waveform parameters.
When `delta_t` is absent, the dispatcher derives it from `delta_f * length`,
just as for the built-in models. Providers must support `delta_t`, `start_time`
(default zero), and either `length` or exclusive `end_time`, return TimeSeries
with the correct epoch and cadence, and honor requested `ifos`. Masses use solar
masses, distance Mpc, angles radians, and times seconds. Each provider must
document its phase/time anchor and orbit conventions. Returned channels use
`LISA_X/Y/Z` or `LISA_A/E/T` names.

The unified `generate_waveform` interface also accepts registered names. It
passes the detector window and model settings, converts XYZ to AET as needed,
and filters the requested channels. Providers used there must expose a complete
XYZ or AET set by default. Search CLI short-template model selection remains
FastBHB; this hook currently covers the public waveform-generation interfaces.


## Optimisation and refinement settings

`hwilas_optimise_snr_peaks` supports FastBHB (IMRPhenomHM) and PhenomTHM with
TDI spline. Each refinement option inherits the corresponding optimisation
value when omitted:

```bash
--approximant fastbhb --orbits analytic --f-ref 3e-5 \
--refinement-approximant IMRPhenomTHM --refinement-orbits esa \
--refinement-f-ref 1e-4 \
--refinement-generator-keywords response_nodes:4096
```

`--generator-keywords key:value ...` configures optimisation; refinement inherits
these keywords and merges any `--refinement-generator-keywords` overrides.
`--waveform-model`, `--refinement-waveform-model` and `--orbit` remain aliases.
FastBHB requires analytic orbits; THM supports analytic and ESA orbits. ESA needs
the absolute `--data-time-origin` or equivalent input metadata. The default
approximant is FastBHB, orbits are analytic, and `f_ref` defaults to `f_lower`.
Changing any stage setting triggers phase reseeding and comparison in the
refinement model before local fitting. Final waveform metadata records the
refinement settings; `population_*` attributes record optimisation settings.


## Local BBHx installation

The `env_hwilas_search` environment has a CPU build of
`uk-lisa-gs/BBHx@fix-barycentre-delay`, commit
`2725d96d56365422a93edee8bd5b8c6a31a55faa`, installed as
`bbhx==1.0.10+fixbarycentredelay.2725d96`, plus
`hwilas-bbhx-provider==0.1.0`. It is discovered through `hwilas.waveforms`;
no temporary virtual environment or `PYTHONPATH` is needed.

Use `--approximant BBHx --orbits esa --f-ref 0`. For the final THM stage,
add:

```sh
--refinement-approximant PhenomTHM --refinement-orbits esa \
--refinement-f-ref 3e-5 --refinement-generator-keywords response_nodes:4096
```
Supply `--data-time-origin` if the data metadata does not contain it.
The provider automatically converts/caches ESA OEM trajectories; an existing
converted HDF file can be passed with `--generator-keywords 'orbit_file:"/absolute/path/orbits.h5"'`.

The installed BBHx kernel retains its original peak phase reference. The
experimental patch honoring nonzero `f_ref` is not included in this installation.

The orbit backend is `lisaanalysistools==1.2.0+hwilas.h5py316`, whose runtime
code is 1.2.0. Its packaging constraint is locally changed from `h5py<3.15`
to `h5py>=3.15.1,<3.17` to coexist with Mojito's `h5py>=3.15.1` requirement.
Orbit HDF reads, position/light-time interpolation and BBHx generation were
checked with h5py 3.16.0. The original 1.2.0 backend and the newer 1.2.8 backend
are not interchangeable for this BBHx build.

Built wheels, SHA256/provenance manifest and build logs are retained in
`examples/search/diagnostics/output/bbhx_native_dist/` (ignored local artifacts).
To reinstall from those local wheels, select the BBHx wheel, provider wheel,
and the **1.2.0+hwilas.h5py316** lisatools wheel; the original 1.2.0 wheel in
that directory retains the incompatible h5py upper bound.
