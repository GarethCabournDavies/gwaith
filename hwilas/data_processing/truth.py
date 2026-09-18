"""Utilities for extracting "truth" (injected-signal) data from processed files.

Sangria and Mojito simulation files store their injected sources under a ``sky``
group, with one sub-group per source type (e.g. ``mbhb``, ``igb``, ``vgb``,
``dgb``). Each source group contains:

* ``<group>/cat`` -- a structured array of per-source parameters (one row per
  injected source), e.g. for MBHBs: ``Mass1``, ``CoalescenceTime``, ...
* ``<group>/tdi`` -- the summed TDI waveform with fields ``(t, X, Y, Z)`` and
  attributes ``dt`` / ``t0`` describing the sample spacing and epoch.

The helpers here read that data and expose it in a standardised form so that
both the extraction executable (see ``tools/data_processing/extract_truth``)
and the plotting tools can share a single, well-defined interface.

Only :mod:`h5py` and :mod:`numpy` are required. Mojito files are read directly
with :mod:`h5py`: the catalogue files carry their per-source parameters under a
``Binaries`` group (one 1-D dataset per parameter) plus a ``Metadata`` group,
so no additional package is needed to load them.
"""

import logging

import h5py
import numpy as np

from hwilas.conversions import spin_z_from_magnitude_and_polar_angle

logger = logging.getLogger(__name__)

# Mean obliquity of the ecliptic (J2000), used for equatorial->ecliptic conversion.
OBLIQUITY_RAD = np.deg2rad(23.439291111)

#: Source groups that may appear under the ``sky`` group of a Sangria file.
SANGRIA_SOURCE_GROUPS = ("mbhb", "igb", "vgb", "dgb")

#: TDI field names expected in the ``<group>/tdi`` dataset.
TDI_FIELDS = ("t", "X", "Y", "Z")


class TruthData:
    """Container for a standardised truth extraction.

    Attributes:
        source_group: Name of the source group the data was read from
            (e.g. ``"mbhb"``).
        catalog: Structured :class:`numpy.ndarray` of per-source parameters,
            with one row per injected source. Field names follow the file's
            ``cat`` dataset (e.g. ``Mass1``, ``CoalescenceTime``).
        timeseries: Mapping of TDI channel name (``"LISA_A"``, ``"LISA_E"``,
            ``"LISA_T"``) to a 1-D :class:`numpy.ndarray` of the summed truth
            waveform in that channel, or ``None`` if no waveform was read.
        epoch: Start time of the timeseries in seconds (0.0 if unknown).
        delta_t: Sample spacing of the timeseries in seconds (``None`` if
            unknown).
    """

    def __init__(self, source_group, catalog, timeseries=None,
                 epoch=0.0, delta_t=None):
        self.source_group = source_group
        self.catalog = catalog
        self.timeseries = timeseries
        self.epoch = float(epoch) if epoch is not None else 0.0
        self.delta_t = float(delta_t) if delta_t is not None else None

    @property
    def n_sources(self):
        """Number of injected sources in the catalog."""
        return int(len(self.catalog))

    def coalescence_times(self, field="CoalescenceTime"):
        """Return the per-source coalescence times, or an empty array.

        Returns an empty float array when the catalog has no ``field`` (e.g.
        non-MBHB source groups), so callers can rely on a uniform return type.
        """
        if self.catalog.dtype.names is None:
            return np.array([], dtype=np.float64)

        if field in self.catalog.dtype.names:
            return np.asarray(self.catalog[field], dtype=np.float64).ravel()

        if field == "CoalescenceTime":
            resolved = _resolve_catalog_field(
                self.catalog,
                [
                    "TimeCoalescencePhenomTPHMSSBFrame",
                    "CoalescenceTime",
                    "tc",
                    "TimeCoalescencePetersSourceFrame",
                    "TimeCoalescencePeters",
                    "TimeCoalescenceSourceFrame",
                ],
            )
            if resolved is not None:
                return np.asarray(self.catalog[resolved], dtype=np.float64).ravel()

        return np.array([], dtype=np.float64)

    def to_hdf5(self, output_path, channels=None):
        """Write the truth data to a standardised HDF5 file.

        The layout is:

        * root attributes ``source_group``, ``n_sources``, ``epoch``,
          ``delta_t`` (and ``has_timeseries``)
        * a ``cat`` structured dataset holding the full catalog
        * one 1-D dataset per catalog field for convenient access
        * one dataset per TDI channel under an ``aet/`` group when a timeseries
          was read

        Args:
            output_path: Destination HDF5 path.
            channels: Channels to write from the timeseries. Defaults to all
                channels present in :attr:`timeseries`.
        """
        if self.timeseries is None:
            has_timeseries = False
        else:
            has_timeseries = True
            if channels is None:
                channels = list(self.timeseries.keys())

        with h5py.File(output_path, "w") as f:
            f.attrs["source_group"] = self.source_group
            f.attrs["n_sources"] = self.n_sources
            f.attrs["epoch"] = self.epoch
            f.attrs["delta_t"] = (self.delta_t if self.delta_t is not None
                                  else -1.0)
            f.attrs["has_timeseries"] = has_timeseries

            f.create_dataset("cat", data=self.catalog)
            for field in self.catalog.dtype.names:
                f.create_dataset(
                    field,
                    data=np.asarray(self.catalog[field]).ravel(),
                )

            if has_timeseries:
                aet = f.create_group("aet")
                for channel in channels:
                    if channel not in self.timeseries:
                        continue
                    aet.create_dataset(
                        channel,
                        data=np.asarray(self.timeseries[channel], dtype=np.float64),
                    )

        logger.info(
            "Wrote truth for %d %s source(s) to %s (timeseries=%s)",
            self.n_sources, self.source_group, output_path, has_timeseries,
        )


def _load_catalog(h5file, group_name):
    """Load and flatten the ``sky/<group>/cat`` structured array from an open file."""
    cat = h5file[f"sky/{group_name}/cat"][:]
    return np.asarray(cat).ravel()


def _convert_tdi_to_aet(X, Y, Z):
    """Convert X/Y/Z TDI arrays to A/E/T arrays (mirrors ``tdi.linear_AET_conversion``)."""
    A = (Z - X) / np.sqrt(2.0)
    E = (X - 2.0 * Y + Z) / np.sqrt(6.0)
    T = (X + Y + Z) / np.sqrt(3.0)
    return {
        "LISA_A": A,
        "LISA_E": E,
        "LISA_T": T,
    }


def _load_tdi_to_aet(h5file, group_name):
    """Load the ``<group>/tdi`` waveform and convert it to AET arrays.

    Returns ``(timeseries, epoch, delta_t)`` where ``timeseries`` maps channel
    name to a 1-D array, or ``(None, 0.0, None)`` if the group has no ``tdi``
    dataset.
    """
    tdi_path = f"sky/{group_name}/tdi"
    if tdi_path not in h5file:
        return None, 0.0, None

    tdi = h5file[tdi_path]
    X = np.asarray(tdi["X"], dtype=np.float64).ravel()
    Y = np.asarray(tdi["Y"], dtype=np.float64).ravel()
    Z = np.asarray(tdi["Z"], dtype=np.float64).ravel()

    attrs = dict(tdi.attrs)
    delta_t = float(attrs.get("dt", 0.0)) or None
    epoch = float(attrs.get("t0", 0.0))

    return _convert_tdi_to_aet(X, Y, Z), epoch, delta_t




def load_truth_sangria(
    input_path,
    source_group="mbhb",
    include_timeseries=True,
    time_min=None,
    time_max=None,
):
    """Load truth data from a Sangria (LDC-style) HDF5 file.

    Args:
        input_path: Path to the Sangria HDF5 file.
        source_group: Source group under ``sky/`` to read (default ``"mbhb"``).
        include_timeseries: Whether to also load and convert the summed TDI
            waveform into AET channels.

    Returns:
        TruthData: The extracted truth data.

    Raises:
        ValueError: If the requested source group is not present in the file.
    """
    with h5py.File(input_path, "r") as f:
        sky = f.get("sky")
        if sky is None or source_group not in sky:
            available = list(sky.keys()) if sky is not None else []
            raise ValueError(
                f"Source group '{source_group}' not found under 'sky/' in "
                f"{input_path}. Available groups: {available}"
            )

        catalog = _load_catalog(f, source_group)
        logger.info(
            "Loaded %d %s source(s) from %s",
            len(catalog), source_group, input_path,
        )

        _require_fields(
            catalog,
            ["CoalescenceTime", "Mass1", "Mass2"],
            "Sangria",
        )

        if include_timeseries:
            timeseries, epoch, delta_t = _load_tdi_to_aet(f, source_group)
        else:
            timeseries, epoch, delta_t = None, 0.0, None

    catalog = _coalesce_catalog_fields(catalog)
    catalog = _filter_catalog_by_time(catalog, time_min=time_min, time_max=time_max)

    return TruthData(
        source_group=source_group,
        catalog=catalog,
        timeseries=timeseries,
        epoch=epoch,
        delta_t=delta_t,
    )


def _normalise_field_name(name):
    """Lower-case and strip separators so field names are compared robustly."""
    return str(name).lower().replace("_", "").replace(" ", "")


def _resolve_catalog_field(catalog, candidates):
    """Return the first matching field name from a catalog using a relaxed match."""
    if catalog.dtype.names is None:
        return None

    lookup = {_normalise_field_name(name): name for name in catalog.dtype.names}
    for candidate in candidates:
        match = lookup.get(_normalise_field_name(candidate))
        if match is not None:
            return match

    # Fall back to substring matching for frame-specific fields such as
    # ``TimeCoalescencePhenomTPHMSSBFrame`` and similar variants.
    norm_candidates = [_normalise_field_name(c) for c in candidates]
    for name in catalog.dtype.names:
        norm_name = _normalise_field_name(name)
        for candidate in norm_candidates:
            if candidate in norm_name or norm_name.endswith(candidate):
                return name
    return None


def _add_standard_field(catalog, field_name, values):
    """Append a new float field to a structured catalog if it is not present."""
    if field_name in catalog.dtype.names:
        return catalog

    new_dtype = list(catalog.dtype.descr) + [(field_name, "<f8")]
    out = np.empty(len(catalog), dtype=new_dtype)
    for name in catalog.dtype.names:
        out[name] = catalog[name]
    out[field_name] = np.asarray(values, dtype=np.float64)
    return out


def _compute_chirp_mass(mass1, mass2):
    """Compute the binary chirp mass from component masses in solar masses."""
    mass1 = np.asarray(mass1, dtype=np.float64)
    mass2 = np.asarray(mass2, dtype=np.float64)
    return ((mass1 * mass2) ** 3 / (mass1 + mass2) ** 2) ** (1.0 / 5.0)


def _distance_unit_factor_to_mpc(unit):
    """Return multiplicative factor to convert a distance unit to Mpc.

    Unknown units return ``None`` so callers can keep values unchanged and
    decide whether to warn.
    """
    if unit is None:
        return None

    norm = str(unit).strip().lower()
    if norm in {"mpc", "megaparsec", "megaparsecs"}:
        return 1.0
    if norm in {"pc", "parsec", "parsecs"}:
        return 1.0e-6
    if norm in {"kpc", "kiloparsec", "kiloparsecs"}:
        return 1.0e-3
    if norm in {"gpc", "gigaparsec", "gigaparsecs"}:
        return 1.0e3
    return None


def _to_radians(values, *, latitude=False):
    """Interpret angular values as radians when plausible, otherwise degrees."""
    arr = np.asarray(values, dtype=np.float64)
    limit = 0.5 * np.pi if latitude else 2.0 * np.pi
    in_radians = np.abs(arr) <= (1.1 * limit)
    return np.where(in_radians, arr, np.deg2rad(arr))


def _equatorial_to_ecliptic(longitude_eq, latitude_eq):
    """Convert equatorial coordinates (RA, Dec) to ecliptic (lambda, beta)."""
    ra = _to_radians(longitude_eq, latitude=False)
    dec = _to_radians(latitude_eq, latitude=True)

    sin_beta = (
        np.sin(dec) * np.cos(OBLIQUITY_RAD)
        - np.cos(dec) * np.sin(OBLIQUITY_RAD) * np.sin(ra)
    )
    beta = np.arcsin(np.clip(sin_beta, -1.0, 1.0))

    lam = np.arctan2(
        np.sin(ra) * np.cos(OBLIQUITY_RAD) + np.tan(dec) * np.sin(OBLIQUITY_RAD),
        np.cos(ra),
    )
    lam = np.mod(lam, 2.0 * np.pi)
    return lam, beta


def _ensure_ecliptic_from_equatorial(catalog):
    """Add canonical ecliptic fields from RA/Dec when needed and possible."""
    have_lon = "eclipticlongitude" in catalog.dtype.names
    have_lat = "eclipticlatitude" in catalog.dtype.names
    if have_lon and have_lat:
        return catalog

    ra_name = _resolve_catalog_field(
        catalog,
        ["RightAscension", "rightascension", "RA", "ra", "SkyRightAscension"],
    )
    dec_name = _resolve_catalog_field(
        catalog,
        ["Declination", "declination", "DEC", "dec", "SkyDeclination"],
    )
    if ra_name is None or dec_name is None:
        return catalog

    ecl_lon, ecl_lat = _equatorial_to_ecliptic(catalog[ra_name], catalog[dec_name])
    out = catalog
    if not have_lon:
        out = _add_standard_field(out, "eclipticlongitude", ecl_lon)
    if "eclipticlatitude" not in out.dtype.names:
        out = _add_standard_field(out, "eclipticlatitude", ecl_lat)
    return out


def _copy_canonical_field(catalog, canonical_name, candidates):
    """Copy a canonical field from the first matching raw field name."""
    if canonical_name in catalog.dtype.names:
        return catalog

    raw_name = _resolve_catalog_field(catalog, candidates)
    if raw_name is None:
        return catalog
    return _add_standard_field(catalog, canonical_name, catalog[raw_name])


def _ensure_spin_z(catalog, canonical_name, magnitude_candidates, polar_angle_candidates):
    """Ensure a z-projection spin field exists for either component spin."""
    if canonical_name in catalog.dtype.names:
        return catalog

    direct = _resolve_catalog_field(catalog, [canonical_name, canonical_name.upper(), canonical_name.lower()])
    if direct is not None:
        return _add_standard_field(catalog, canonical_name, catalog[direct])

    magnitude_name = _resolve_catalog_field(catalog, magnitude_candidates)
    polar_name = _resolve_catalog_field(catalog, polar_angle_candidates)
    if magnitude_name is not None and polar_name is not None:
        return _add_standard_field(
            catalog,
            canonical_name,
            spin_z_from_magnitude_and_polar_angle(catalog[magnitude_name], catalog[polar_name]),
        )
    return catalog


def standardize_truth_catalog(catalog):
    """Normalise truth catalog fields to a compact project-wide canonical schema.

    The goal is deliberately small and stable: we want the same core information
    from both Mojito and Sangria files without carrying around file-specific field
    names. The canonical fields we expose are:

    * ``tc``
    * ``mass1`` / ``mass2``
    * ``chirp_mass``
    * ``eclipticlongitude`` / ``eclipticlatitude``
    * ``spin1z`` / ``spin2z``
    * ``inclination``
    * ``polarization`` / ``polarisation``
    * optional ``distance`` and ``coa_phase`` when present

    The original file-specific names are preserved in the catalog for
    compatibility, but the standardised fields make downstream code simple.
    """
    if catalog.dtype.names is None:
        return catalog

    out = catalog.copy()

    # Coalescence time.
    out = _copy_canonical_field(
        out,
        "tc",
        [
            "TimeCoalescencePhenomTPHMSSBFrame",
            "TimeCoalescencePetersSourceFrame",
            "TimeCoalescencePeters",
            "TimeCoalescenceSourceFrame",
            "CoalescenceTime",
            "tc",
        ],
    )
    if "CoalescenceTime" not in out.dtype.names and "tc" in out.dtype.names:
        out = _add_standard_field(out, "CoalescenceTime", out["tc"])

    # Component masses.
    out = _copy_canonical_field(out, "mass1", [
        "Mass1", "PrimaryMassSSBFrame", "PrimaryMassSourceFrame", "PrimaryMass",
        "Mass1SourceFrame", "mass1", "mass_1", "m1",
    ])
    out = _copy_canonical_field(out, "mass2", [
        "Mass2", "SecondaryMassSSBFrame", "SecondaryMassSourceFrame", "SecondaryMass",
        "Mass2SourceFrame", "mass2", "mass_2", "m2",
    ])

    if "chirp_mass" not in out.dtype.names and "mass1" in out.dtype.names and "mass2" in out.dtype.names:
        out = _add_standard_field(out, "chirp_mass", _compute_chirp_mass(out["mass1"], out["mass2"]))

    # Sky localisation.
    out = _copy_canonical_field(out, "eclipticlongitude", [
        "EclipticLongitude", "eclipticlongitude", "Longitude", "LongitudeSourceFrame",
        "SkyLongitude",
    ])
    out = _copy_canonical_field(out, "eclipticlatitude", [
        "EclipticLatitude", "eclipticlatitude", "Latitude", "LatitudeSourceFrame",
        "SkyLatitude",
    ])
    out = _ensure_ecliptic_from_equatorial(out)

    # Orientation / wave parameters.
    out = _copy_canonical_field(out, "inclination", [
        "Inclination", "inclination", "iota",
    ])
    out = _copy_canonical_field(out, "polarization", [
        "Polarisation", "Polarization", "polarisation", "polarization",
    ])
    if "polarisation" not in out.dtype.names and "polarization" in out.dtype.names:
        out = _add_standard_field(out, "polarisation", out["polarization"])
    if "polarization" not in out.dtype.names and "polarisation" in out.dtype.names:
        out = _add_standard_field(out, "polarization", out["polarisation"])

    # Useful additional parameters when available.
    out = _copy_canonical_field(out, "distance", [
        "Distance", "distance", "LuminosityDistance", "luminositydistance",
    ])
    out = _copy_canonical_field(out, "coa_phase", [
        "CoaPhase", "coa_phase", "CoalescencePhase", "coalescencephase",
        "PhaseAtCoalescence",
    ])

    # Spin projections along the line of sight.
    out = _ensure_spin_z(
        out,
        "spin1z",
        ["Spin1", "spin1", "chi1", "chi_1", "MagnitudeOfSpin1", "SpinMagnitude1"],
        [
            "PolarAngleOfSpin1", "polarangleofspin1", "ThetaSpin1", "Theta1",
            "Spin1PolarAngle", "SpinTilt1",
        ],
    )
    out = _ensure_spin_z(
        out,
        "spin2z",
        ["Spin2", "spin2", "chi2", "chi_2", "MagnitudeOfSpin2", "SpinMagnitude2"],
        [
            "PolarAngleOfSpin2", "polarangleofspin2", "ThetaSpin2", "Theta2",
            "Spin2PolarAngle", "SpinTilt2",
        ],
    )

    return out


def load_truth_mojito(
    input_path,
    source_group="mbhb",
    include_timeseries=True,
    time_min=None,
    time_max=None,
):
    """Load truth data from a Mojito catalogue HDF5 file.

    Mojito *catalogue* files (as opposed to L1 data files) store their injected
    sources under a root-level ``Binaries`` group: one 1-D dataset per parameter
    (e.g. ``PrimaryMassSourceFrame``, ``TimeCoalescencePetersSourceFrame``), each
    carrying a ``unit`` attribute, plus a root-level ``Metadata`` group holding
    file-level attributes such as ``SourceType`` and ``NumberBinaries``. This is
    read directly with :mod:`h5py`, so the optional ``mojito`` package is not
    required.

    The per-parameter datasets are combined into a single structured array (one
    row per source) and, for convenience, a ``CoalescenceTime`` field is added
    when the file only stores frame-specific coalescence times (see
    :func:`_coalesce_catalog_fields`).

    Args:
        input_path: Path to the Mojito catalogue HDF5 file.
        source_group: Name recorded on the resulting :class:`TruthData`
            (default ``"mbhb"``). The catalogue layout is not grouped by source,
            so this is only used as a label.
        include_timeseries: Accepted for API compatibility with
            :func:`load_truth_sangria`. Catalogue files do not store waveforms,
            so the resulting :class:`TruthData` always has ``timeseries=None``.

    Returns:
        TruthData: The extracted truth data (catalog only, no timeseries).

    Raises:
        ValueError: If the file does not contain a ``Binaries`` group.
    """
    with h5py.File(input_path, "r") as f:
        if "Binaries" not in f:
            available = list(f.keys())
            raise ValueError(
                f"'{input_path}' does not look like a Mojito catalogue file: "
                f"no 'Binaries' group found. Available top-level entries: {available}"
            )

        binaries = f["Binaries"]
        field_names = list(binaries.keys())
        if not field_names:
            raise ValueError(
                f"'{input_path}' has an empty 'Binaries' group; no parameters to "
                "extract."
            )

        required = [
            "TimeCoalescencePhenomTPHMSSBFrame",
            "PrimaryMassSourceFrame",
            "SecondaryMassSourceFrame",
        ]
        missing = [name for name in required if name not in field_names]
        if missing:
            raise ValueError(
                f"Mojito truth catalogue '{input_path}' is missing required field(s): "
                f"{', '.join(missing)}"
            )

        arrays = {name: np.asarray(binaries[name][:], dtype=np.float64).ravel()
                  for name in field_names}

        # Normalise known distance columns to Mpc using Mojito-provided unit
        # attributes so waveform generators see consistent physical units.
        for dist_name in ("LuminosityDistance", "Distance", "distance"):
            if dist_name not in binaries:
                continue
            unit = binaries[dist_name].attrs.get("unit", None)
            factor = _distance_unit_factor_to_mpc(unit)
            if factor is None:
                logger.warning(
                    "Unknown distance unit %r for field '%s' in %s; keeping raw values.",
                    unit,
                    dist_name,
                    input_path,
                )
                continue
            if factor != 1.0:
                arrays[dist_name] = arrays[dist_name] * factor
                logger.info(
                    "Converted '%s' from unit %r to Mpc with factor %.9g.",
                    dist_name,
                    unit,
                    factor,
                )
        n_sources = min(len(a) for a in arrays.values())
        if len({len(a) for a in arrays.values()}) != 1:
            logger.warning(
                "Inconsistent parameter lengths in 'Binaries' of %s; using the "
                "minimum length (%d).", input_path, n_sources,
            )

        catalog = np.empty(n_sources, dtype=[(name, "<f8") for name in field_names])
        for name in field_names:
            catalog[name] = arrays[name][:n_sources]

        # Record the file-level metadata alongside the per-source parameters so
        # it survives the standardised-output round trip.
        if "Metadata" in f:
            meta_attrs = dict(f["Metadata"].attrs)
            for key, value in meta_attrs.items():
                if key not in catalog.dtype.names and np.isscalar(value):
                    try:
                        catalog[key] = value
                    except (ValueError, TypeError):
                        # Skip non-scalar or otherwise incompatible attributes.
                        continue

        catalog = _coalesce_catalog_fields(catalog)
        catalog = _filter_catalog_by_time(catalog, time_min=time_min, time_max=time_max)

    logger.info(
        "Loaded %d %s source(s) from Mojito catalogue file %s",
        n_sources, source_group, input_path,
    )

    return TruthData(
        source_group=source_group,
        catalog=catalog,
        timeseries=None,
        epoch=0.0,
        delta_t=None,
    )


def detect_truth_file_type(input_path):
    """Best-effort detection of whether a truth HDF5 file is Mojito or Sangria.

    The two file types are not always distinguishable purely by layout (both use
    the ``sky/<group>/{cat,tdi}`` structure), so this relies on markers that
    differ between them:

    * A **Mojito** L1 file is written by the ``mojito`` package and carries its
      own root attributes / group layout (e.g. a ``meta`` or ``L1``-style
      metadata group) that Sangria files do not have.
    * A **Sangria** (LDC-style) file has no such markers and is assumed to be the
      default when nothing Mojito-specific is found.

    Returns:
        str: ``"mojito"`` or ``"sangria"``.

    Note:
        This is a heuristic. If you are unsure, pass ``--type`` explicitly to the
        extraction executable rather than relying on auto-detection.
    """
    with h5py.File(input_path, "r") as f:
        root_attrs = set(f.attrs.keys())
        top_groups = list(f.keys())

        mojito_markers = {"meta", "L1", "metadata", "mojito"}
        if mojito_markers & (root_attrs | set(top_groups)):
            return "mojito"

    return "sangria"


def add_truth_arguments(parser):
    """Add standardised truth-overlay CLI options to a plotting parser.

    These are shared by the plotting tools so that every one of them exposes a
    consistent way to overlay injected ("truth") signals on top of its plots:

    - ``--truth-file``: path to a standardised truth HDF5 file (as produced by
      ``tools/data_processing/extract_truth``). When given, the tool loads the
      per-source coalescence times and the summed AET waveform for overlay.
    - ``--plot-truth``: flag to actually draw the truth overlay. The file is only
      loaded when this is set (or when a truth file is supplied), so existing
      behaviour is unchanged unless requested.
    - ``--truth-time-min`` / ``--truth-time-max``: optional window that restricts
      the truth catalogue to sources whose coalescence time falls in the current
      analysis interval before plotting or extraction.

    Returns the parser, following the style of :func:`hwilas.common.add_hwilas_common_options`.
    """
    group = parser.add_argument_group("truth", "Options for overlaying injected signals")
    group.add_argument(
        "--truth-file",
        default=None,
        help="Standardised truth HDF5 file (see tools/data_processing/extract_truth) "
             "from which to read injected-signal coalescence times and the summed AET waveform.",
    )
    group.add_argument(
        "--plot-truth",
        action="store_true",
        help="Overlay the truth signals (requires --truth-file).",
    )
    group.add_argument(
        "--truth-time-min",
        type=float,
        default=None,
        help="Only keep truth sources with coalescence times >= this value (seconds).",
    )
    group.add_argument(
        "--truth-time-max",
        type=float,
        default=None,
        help="Only keep truth sources with coalescence times <= this value (seconds).",
    )
    return parser


def load_truth_file(path, time_min=None, time_max=None):
    """Load a standardised truth file into a :class:`TruthData`.

    The file must have been produced by ``tools/data_processing/extract_truth``
    (i.e. it contains the ``source_group`` attribute and a ``cat`` dataset). This
    is the single entry point the plotting tools use to obtain truth data, so
    they do not need to know about HDF5 layout details.

    Args:
        path: Path to the standardised truth HDF5 file.
        time_min: Optional minimum coalescence time (seconds) to keep.
        time_max: Optional maximum coalescence time (seconds) to keep.

    Returns:
        TruthData: The loaded truth data (catalog and, if present, AET timeseries).

    Raises:
        ValueError: If the file does not look like a standardised truth file.
    """
    with h5py.File(path, "r") as f:
        if "source_group" not in f.attrs or "cat" not in f:
            raise ValueError(
                f"'{path}' does not look like a standardised truth file (missing "
                "'source_group' attribute or 'cat' dataset). Run "
                "tools/data_processing/extract_truth first."
            )
        source_group = str(f.attrs["source_group"])
        catalog = np.asarray(f["cat"][:])

        timeseries = None
        epoch = float(f.attrs.get("epoch", 0.0))
        delta_t = f.attrs.get("delta_t", None)
        if f.attrs.get("has_timeseries", False) and "aet" in f:
            aet = f["aet"]
            timeseries = {ch: np.asarray(aet[ch][:], dtype=np.float64) for ch in aet.keys()}

    catalog = _filter_catalog_by_time(catalog, time_min=time_min, time_max=time_max)

    return TruthData(
        source_group=source_group,
        catalog=catalog,
        timeseries=timeseries,
        epoch=epoch,
        delta_t=None if delta_t is None else float(delta_t),
    )


def _coalesce_catalog_fields(catalog):
    """Return a copy of ``catalog`` guaranteed to expose a ``CoalescenceTime`` field.

    This is intentionally strict: Mojito inputs must use the exact
    ``TimeCoalescencePhenomTPHMSSBFrame`` field, while generic Sangria-style
    inputs must already expose a ``CoalescenceTime`` or ``tc`` field. We do not
    silently guess among aliases.
    """
    catalog = standardize_truth_catalog(catalog)
    if "CoalescenceTime" in catalog.dtype.names:
        return catalog

    if "tc" in catalog.dtype.names:
        return _add_standard_field(catalog, "CoalescenceTime", catalog["tc"])

    if "TimeCoalescencePhenomTPHMSSBFrame" in catalog.dtype.names:
        return _add_standard_field(
            catalog,
            "CoalescenceTime",
            catalog["TimeCoalescencePhenomTPHMSSBFrame"],
        )

    raise ValueError(
        "Truth catalogue does not contain a valid coalescence-time field. "
        "Expected either 'CoalescenceTime', 'tc', or for Mojito inputs "
        "'TimeCoalescencePhenomTPHMSSBFrame'."
    )


def _filter_catalog_by_time(catalog, time_min=None, time_max=None, time_field="CoalescenceTime"):
    """Return a subset of the catalog whose coalescence times fall in a window.

    The filter is deliberately applied at the loader level so plotting and
    extraction both work with the same windowed truth data. If no valid time field
    is found, the original catalog is returned unchanged.
    """
    if catalog.dtype.names is None:
        return catalog

    if time_field not in catalog.dtype.names:
        time_field = _resolve_catalog_field(
            catalog,
            [
                "TimeCoalescencePhenomTPHMSSBFrame",
                "TimeCoalescencePetersSourceFrame",
                "TimeCoalescencePeters",
                "TimeCoalescenceSourceFrame",
                "CoalescenceTime",
                "tc",
            ],
        )
    if time_field is None:
        return catalog

    times = np.asarray(catalog[time_field], dtype=np.float64)
    mask = np.ones(len(catalog), dtype=bool)
    if time_min is not None:
        mask &= times >= float(time_min)
    if time_max is not None:
        mask &= times <= float(time_max)

    selected = catalog[mask]
    logger.info(
        "Truth filter: %d/%d signal(s) in time window [%s, %s] for field '%s'",
        len(selected), len(catalog), time_min, time_max, time_field,
    )
    if len(selected):
        values = np.asarray(selected[time_field], dtype=np.float64)
        if len(values) <= 10:
            logger.info("Signal times in window: %s", np.array2string(values, precision=6, separator=', '))
        else:
            logger.info(
                "First 10 signal times in window: %s",
                np.array2string(values[:10], precision=6, separator=', '),
            )
    return selected

def _require_fields(catalog, required_fields, file_kind):
    """Validate that all required raw fields are present and raise on mismatch."""
    missing = [name for name in required_fields if name not in catalog.dtype.names]
    if missing:
        raise ValueError(
            f"{file_kind} truth catalogue is missing required field(s): {', '.join(missing)}"
        )
    return catalog


__all__ = [
    "TruthData",
    "add_truth_arguments",
    "detect_truth_file_type",
    "load_truth_file",
    "load_truth_mojito",
    "load_truth_sangria",
    "require_fields"
]