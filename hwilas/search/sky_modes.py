"""LISA-frame sky proposals with optional full helicity exploration."""
import logging
from functools import lru_cache
import numpy as np


def lisa_sky_branches(longitude, sin_latitude, polarization, cos_inclination):
    """Generate 8 sky mode branches via reflection and rotation."""
    results = []
    for reflect in [False, True]:
        if reflect:
            s_lat = -sin_latitude
            c_inc = -cos_inclination
            pol_base = np.pi - polarization
        else:
            s_lat = sin_latitude
            c_inc = cos_inclination
            pol_base = polarization
        for k in range(4):
            delta = k * np.pi / 2
            lon_k = (longitude + delta) % (2 * np.pi)
            pol_k = (pol_base + delta) % np.pi
            results.append([lon_k, s_lat, pol_k, c_inc])
    return np.array(results, dtype=float)


@lru_cache(maxsize=1)
def _gwreparam_functions():
    """Load optional transformations, reporting reduced exploration once."""
    try:
        from gwreparam.lisa.symmetry import fold_sky, unfold_sky
        from gwreparam.lisa.coordinates import helicity_coordinates, inverse_helicity_coordinates
    except ModuleNotFoundError as exc:
        if exc.name != "gwreparam":
            raise
        logging.getLogger(__name__).warning(
            "gwreparam is unavailable; using eight sky proposals without helicity "
            "exploration. Install gwreparam separately for full exploration.")
        return None
    return fold_sky, unfold_sky, helicity_coordinates, inverse_helicity_coordinates


def sky_mode_vectors(position, reference_tc, initial_position_deg=-140.):
    """Enumerate 64 sky/helicity branches, or eight sky branches without gwreparam.

    With gwreparam, preserve continuous folded helicity coordinates.
    Without it, preserve phase and use sky reflections and rotations only.
    These are proposals, not assumed waveform symmetries.
    The identity is returned first. Merger time and distance are profiled by the
    caller. At singular face-on or edge-on helicity points, explore sky only.
    """
    from lisabeta.lisa import lisatools as lt
    from ldc.waveform.lisabeta.fast_bhb import ASTRONOMICAL_YEAR
    functions = _gwreparam_functions()
    vector = np.asarray(position, float)
    if vector.shape != (9,):
        raise ValueError('Sky exploration requires nine profiled shape coordinates')
    orbit_origin = np.radians(initial_position_deg)*ASTRONOMICAL_YEAR/(2*np.pi)
    t_lisa, longitude, latitude, polarization = lt.ConvertSSBframeParamsToLframe(
        reference_tc-orbit_origin, vector[4], vector[5], vector[7], constellation_ini_phase=0.)
    if functions is None:
        proposals = lisa_sky_branches(longitude, np.sin(latitude), polarization,
                                      np.cos(vector[6]))
        branches = [vector.copy()]
        for lon_lisa, sin_beta, psi_lisa, cos_inc in proposals[1:]:
            _, lon, lat, psi_ssb = lt.ConvertLframeParamsToSSBframe(
                t_lisa, float(lon_lisa), float(np.arcsin(np.clip(sin_beta, -1, 1))),
                float(psi_lisa), constellation_ini_phase=0.)
            candidate = vector.copy()
            candidate[4:8] = (lon % (2*np.pi), lat,
                              np.arccos(np.clip(cos_inc, -1, 1)), psi_ssb % np.pi)
            branches.append(candidate)
        return np.asarray(branches)
    fold_sky, unfold_sky, helicity_coordinates, inverse_helicity_coordinates = functions
    (lonf, sinbf, psif, cosif), _, sky_label = fold_sky(
        longitude, np.sin(latitude), polarization, np.cos(vector[6]), symmetry_group='8-mode')
    singular = abs(float(cosif)) < 1e-8 or abs(float(cosif)) > 1-1e-8
    if not singular:
        helicity, _, labels = helicity_coordinates(vector[8], cosif, psif, lonf)
        identity = int(labels['hand'] > 0)+2*int(labels['psi'])+4*int(labels['phase'])+8*int(sky_label)
    else:
        identity = int(sky_label)
    branches = []
    for label in range(8 if singular else 64):
        if singular:
            phase, inclination_cosine, psi = vector[8], cosif, psif
            sky = label
        else:
            (phase, inclination_cosine, psi), _ = inverse_helicity_coordinates(
                *helicity, lonf, hand=1 if label % 2 else -1,
                psi_branch=(label//2)%2, phase_branch=(label//4)%2)
            sky = label//8
        (lon_lisa, sin_beta, psi_lisa, cos_inc), _ = unfold_sky(
            lonf, sinbf, psi, inclination_cosine, sky, symmetry_group='8-mode')
        _, lon, lat, psi_ssb = lt.ConvertLframeParamsToSSBframe(t_lisa,
            float(lon_lisa), float(np.arcsin(np.clip(sin_beta,-1,1))), float(psi_lisa),
            constellation_ini_phase=0.)
        candidate = vector.copy()
        candidate[4], candidate[5], candidate[6], candidate[7], candidate[8] = (
            lon % (2*np.pi), lat, np.arccos(np.clip(cos_inc,-1,1)), psi_ssb % np.pi,
            float(phase) % (2*np.pi))
        branches.append(candidate)
    branches = np.asarray(branches)
    return np.vstack([vector, np.delete(branches, identity, axis=0)])
