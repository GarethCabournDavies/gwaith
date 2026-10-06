"""Time-domain FastBHB waveform generator.

Wraps the LDC ``FastBHB`` (lisabeta-backed) generator to provide MBHB
waveforms through the same hwilas-named parameter interface as the BBHx and
PhenomX generators, so it can be dispatched via :func:`hwilas.waveforms.get_fd_waveform`.

The LDC ``FastBHB`` generator expects a template dictionary with LDC-style
field names (e.g. ``Mass1``, ``CoalescenceTime``). The private
:func:`_build_ldc_template` helper maps the hwilis-named parameters onto that
schema, and :func:`generate_waveform_for_data` provides an adapter
for callers that already hold an LDC structured array.
"""
import astropy.units as un
import numpy as np
from gwaith.types import TimeSeries
from ldc.lisa import orbits
from ldc.waveform.lisabeta import FastBHB

from hwilas.data_processing.tdi import to_timeseries

# Waveform modes passed to FastBHB (the standard HM mode set).
DEFAULT_FASTBHB_MODES = [(2, 2), (2, 1), (3, 3), (3, 2), (4, 4), (4, 3)]

# Channels returned by the generator (AET TDI frame).
FASTBHB_CHANNELS = ('LISA_A', 'LISA_E', 'LISA_T')

# LDC-style field names required by FastBHB templates.
_LDC_TEMPLATE_FIELDS = (
    'Mass1', 'Mass2', 'CoalescenceTime', 'Distance', 'EclipticLatitude',
    'EclipticLongitude', 'Inclination', 'Polarization', 'Spin1x', 'Spin1y',
    'Spin1z', 'Spin2x', 'Spin2y', 'Spin2z', 'PhaseAtCoalescence', 'Cadence',
)

# hwilas-named parameter -> LDC template field name.
_HWILAS_TO_LDC = {
    'mass1': 'Mass1',
    'mass2': 'Mass2',
    'tc': 'CoalescenceTime',
    'distance': 'Distance',
    'eclipticlatitude': 'EclipticLatitude',
    'eclipticlongitude': 'EclipticLongitude',
    # NOTE: 'inclination' and 'polarization' are NOT mapped here. They represent
    # raw LDC catalog angles (InitialPolarAngleL / InitialAzimuthalAngleL) and
    # require conversion via aziPolAngleL2PsiIncl, which is handled explicitly in
    # _build_ldc_template below.
    'spin1x': 'Spin1x',
    'spin1y': 'Spin1y',
    'spin1z': 'Spin1z',
    'spin2x': 'Spin2x',
    'spin2y': 'Spin2y',
    'spin2z': 'Spin2z',
    # Accept BBHx/Phenom-style naming for coalescence phase.
    'phi_ref': 'PhaseAtCoalescence',
    'coa_phase': 'PhaseAtCoalescence',
}

# LDC fields not driven by hwilas-named parameters, with their defaults.
_LDC_DEFAULTS = {
    'Mass1': 0.0,
    'Mass2': 0.0,
    'CoalescenceTime': 0.0,
    'Distance': 0.0,
    'EclipticLatitude': 0.0,
    'EclipticLongitude': 0.0,
    'Inclination': 0.0,
    'Polarization': 0.0,
    'Spin1x': 0.0,
    'Spin1y': 0.0,
    'Spin1z': 0.0,
    'Spin2x': 0.0,
    'Spin2y': 0.0,
    'Spin2z': 0.0,
    # Scalar spin magnitudes (lisabeta chi1/chi2) and coalescence phase are
    # derived in _build_ldc_template; these defaults keep the template complete.
    'Spin1': 0.0,
    'Spin2': 0.0,
    'PhaseAtCoalescence': 0.0,
}

# Lazily-created analytic LISA orbits, shared across calls.
_analytic_orbits = None

# Initial LISA orbit position for the Mojito orbit phase (radians). The LDC
# FastBHB wrapper maps this onto lisabeta's ``Orbitt0``; without it the
# templates see LISA in the wrong place (a ~330 s timing offset, on top of the
# 800 s catalogue time-reference offset). See the Mojito orbit convention.
MOJITO_INITIAL_POSITION_RAD = np.deg2rad(-140.0)


def _get_lisa_orbits(initial_position_rad=MOJITO_INITIAL_POSITION_RAD):
    """Return (and cache) the analytic LISA orbits.

    Parameters
    ----------
    initial_position_rad : float, optional
        Initial LISA orbit position in radians. Defaults to the Mojito value
        (:data:`MOJITO_INITIAL_POSITION_RAD`).
    """
    global _analytic_orbits
    if _analytic_orbits is None:
        _analytic_orbits = orbits.Orbits.type(dict({
            "nominal_arm_length": 2.5e9 * un.m,
            "initial_rotation": 0 * un.rad,
            "initial_position": float(initial_position_rad) * un.rad,
            "orbit_type": "analytic",
        }))
    return _analytic_orbits


def _build_ldc_template(params):
    """Build an LDC-style template dictionary from hwilas-named parameters.

    Parameters
    ----------
    params : dict
        Waveform parameters with hwilas names (``mass1``, ``tc``,
        ``distance``, etc.). ``delta_t`` is used for the ``Cadence`` field.

    Returns
    -------
    dict
        Template dictionary with LDC-style field names, ready to pass to
        :class:`ldc.waveform.lisabeta.FastBHB`.
    """
    template = dict(_LDC_DEFAULTS)
    for hwilas_name, ldc_name in _HWILAS_TO_LDC.items():
        if hwilas_name in params:
            template[ldc_name] = float(params[hwilas_name])

    # Determine whether the LDC spin-projection path will be triggered.
    # When ``PolarAngleOfSpin1`` is present in the template, ``rename_as_lisabeta``
    # overwrites ``Spin1``/``Spin2`` with ``Spin * cos(PolarAngleOfSpin)`` — i.e.
    # it expects the *full* spin magnitude in those fields, not the line-of-sight
    # component. We detect this case and adjust accordingly below.
    will_project_spins = (
        'inclination' in params
        and 'polarization' in params
        and ('spin1_polar_angle' in params or 'spin2_polar_angle' in params)
    )

    # The LDC ``rename_as_lisabeta`` step only emits lisabeta's scalar spin
    # parameters (``chi1``/``chi2``) when the template carries the ``Spin1`` /
    # ``Spin2`` fields. Without them, lisabeta's ``complete_spin_params`` raises
    # "Incomplete spin parameters". hwilas expresses spins as components, so we
    # derive the scalar magnitude here.
    for component in ('x', 'y', 'z'):
        template[f'Spin1{component}'] = float(params.get(f'spin1{component}', 0.0))
        template[f'Spin2{component}'] = float(params.get(f'spin2{component}', 0.0))

    if will_project_spins:
        # rename_as_lisabeta will do Spin = Spin * cos(PolarAngleOfSpin), so we
        # must supply the FULL spin magnitude (not the line-of-sight component).
        # Recover it from the z-component: spin_mag = spin_z / cos(polar_angle).
        for body, polar_key in (('1', 'spin1_polar_angle'), ('2', 'spin2_polar_angle')):
            polar_angle = float(params.get(polar_key, 0.0))
            spin_z = float(params.get(f'spin{body}z', 0.0))
            cos_pa = np.cos(polar_angle)
            if abs(cos_pa) > 1e-15:
                full_mag = abs(spin_z / cos_pa)
            else:
                # Polar angle near pi/2: spin is transverse, z-component ~ 0.
                # Fall back to the component magnitude (best effort).
                components = [template[f'Spin{body}{c}'] for c in ('x', 'y', 'z')]
                full_mag = float(np.sqrt(sum(c * c for c in components)))
            template[f'Spin{body}'] = full_mag
    else:
        # No projection: Spin1/Spin2 should be the line-of-sight magnitude.
        for body in ('1', '2'):
            components = [template[f'Spin{body}{c}'] for c in ('x', 'y', 'z')]
            template[f'Spin{body}'] = float(np.sqrt(sum(c * c for c in components)))

    template['Cadence'] = float(params.get('delta_t', 5.0))

    # IMPORTANT: The LDC catalog stores orbital angular momentum orientation as
    # ``InitialPolarAngleL`` / ``InitialAzimuthalAngleL``, NOT as
    # ``Inclination`` / ``Polarization``. The ``rename_as_lisabeta`` function in
    # LDC's FastBHB converts these to lisabeta's ``inc``/``psi`` ONLY IF the
    # template contains ``PolarAngleOfSpin1`` (and the corresponding spin polar
    # angles). If we pass pre-computed ``Inclination``/``Polarization`` without
    # those fields, they are used as-is — which is wrong when the caller has
    # supplied raw LDC catalog values.
    #
    # To ensure correct conversion, we always include the full LDC spin/orientation
    # fields so that ``rename_as_lisabeta`` performs the proper
    # ``aziPolAngleL2PsiIncl`` transformation whenever inclination/polarization
    # are provided.
    if 'inclination' in params and 'polarization' in params:
        template['InitialPolarAngleL'] = float(params['inclination'])
        template['InitialAzimuthalAngleL'] = float(params['polarization'])
        template['PolarAngleOfSpin1'] = float(params.get('spin1_polar_angle', 0.0))
        template['PolarAngleOfSpin2'] = float(params.get('spin2_polar_angle', 0.0))
        # Remove the incorrectly-named fields so rename_as_lisabeta overwrites them.
        template.pop('Inclination', None)
        template.pop('Polarization', None)

    return template


def get_td_waveform_from_fastbhb(
    approximant,
    tdi='1.5',
    **params
):
    """
    Generate a time-domain MBHB waveform using the LDC ``FastBHB`` generator.

    Parameters
    ----------
    approximant : str
        Approximant name (e.g. ``'FastBHB'``). Only used for validation;
        the underlying lisabeta approximation is always ``IMRPhenomHM``.
    tdi : {'1.5', '2'}, optional
        TDI generation to use: ``'1.5'`` (default) or ``'2'`` (second
        generation TDI).
    **params : dict
        Waveform parameters with hwilas names: ``mass1``, ``mass2``, ``tc``,
        ``distance``, ``eclipticlatitude``, ``eclipticlongitude``,
        ``inclination``, ``polarization``, optional spins (``spin1x``/``y``/``z``,
        ``spin2x``/``y``/``z``), ``coa_phase``, plus the sampling grid
        ``delta_t`` and either an explicit time window (``start_time``,
        ``end_time``) or a grid length (``length``).

    Returns
    -------
    dict
        Dictionary mapping each requested channel (default AET:
        ``'LISA_A'``, ``'LISA_E'``, ``'LISA_T'``) to a
        ``gwaith.types.TimeSeries``.
    """
    ifos = params.get('ifos', list(FASTBHB_CHANNELS))
    unknown_ifos = [ifo for ifo in ifos if ifo not in FASTBHB_CHANNELS]
    if unknown_ifos:
        raise ValueError(
            f"Unknown channel(s) {unknown_ifos} for FastBHB. "
            f"Expected one of {list(FASTBHB_CHANNELS)}."
        )

    tdi_str = str(tdi)
    if tdi_str not in ('1.5', '2'):
        raise ValueError(f"tdi must be '1.5' or '2', got {tdi!r}")

    delta_t = float(params['delta_t'])
    start_time = float(params.get('start_time', 0.0))
    if 'end_time' in params:
        end_time = float(params['end_time'])
    else:
        length = int(params['length'])
        end_time = start_time + length * delta_t

    fast_hm = FastBHB(
        "MBHB",
        approx="IMRPhenomHM",
        T=end_time,
        delta_t=delta_t,
        orbits=_get_lisa_orbits(),
        modes=DEFAULT_FASTBHB_MODES,
    )

    template = _build_ldc_template(params)
    A, E, T = fast_hm.get_td_tdiaet(
        template=template,
        tdi2=(tdi_str == '2'),
    )

    # Cut to the requested start/end time window.
    start_idx = int(start_time / delta_t)
    end_idx = int(end_time / delta_t)

    return to_timeseries(
        {
            ifo: array[start_idx:end_idx]
            for ifo, array in zip(FASTBHB_CHANNELS, (A, E, T))
            if ifo in ifos
        },
        delta_t,
        epoch=start_time,
    )


def generate_waveform_for_data(
    mbhb,
    start_time,
    end_time,
    delta_t,
):
    """Generate a waveform from an LDC structured array.

    Parameters
    ----------
    mbhb : numpy.recarray or similar
        One row of an LDC-style MBHB catalog (structured array with fields
        such as ``Mass1``, ``CoalescenceTime``).
    start_time, end_time : float
        Waveform window in seconds.
    delta_t : float
        Sampling interval in seconds.

    Returns
    -------
    dict
        Dictionary of AET ``gwaith.types.TimeSeries`` objects.
    """
    wave = dict(zip(mbhb.dtype.names, mbhb))
    wave['Cadence'] = delta_t
    return _generate_waveform_from_ldc_template(
        wave,
        start_time,
        end_time,
        delta_t,
    )


def _generate_waveform_from_ldc_template(template, start_time, end_time, delta_t):
    """Generate AET TimeSeries from an already-built LDC template dict."""
    from ldc.waveform.lisabeta import FastBHB

    fast_hm = FastBHB(
        "MBHB",
        approx="IMRPhenomHM",
        T=end_time,
        delta_t=delta_t,
        orbits=_get_lisa_orbits(),
        modes=DEFAULT_FASTBHB_MODES,
    )

    A, E, T = fast_hm.get_td_tdiaet(template=template, tdi2=False)

    start_idx = int(start_time / delta_t)
    end_idx = int(end_time / delta_t)

    return to_timeseries(
        {
            'LISA_A': A[start_idx:end_idx],
            'LISA_E': E[start_idx:end_idx],
            'LISA_T': T[start_idx:end_idx],
        },
        delta_t,
        epoch=start_time,
    )
