"""LISA-frame sky/helicity branch proposals using gwreparam."""
import numpy as np


def sky_mode_vectors(position, reference_tc, initial_position_deg=-140.):
    """Enumerate 8 sky x 2 handedness x 2 polarisation x 2 phase branches.

    Preserve continuous folded helicity coordinates across branch proposals.
    These are proposals, not assumed waveform symmetries.
    The identity is returned first. Merger time and distance are profiled by the
    caller. At singular face-on or edge-on helicity points, explore sky only.
    """
    from lisabeta.lisa import lisatools as lt
    from ldc.waveform.lisabeta.fast_bhb import ASTRONOMICAL_YEAR
    from gwreparam.lisa.symmetry import fold_sky, unfold_sky
    from gwreparam.lisa.coordinates import helicity_coordinates, inverse_helicity_coordinates
    vector = np.asarray(position, float)
    if vector.shape != (9,):
        raise ValueError('Sky exploration requires nine profiled shape coordinates')
    orbit_origin = np.radians(initial_position_deg)*ASTRONOMICAL_YEAR/(2*np.pi)
    t_lisa, longitude, latitude, polarization = lt.ConvertSSBframeParamsToLframe(
        reference_tc-orbit_origin, vector[4], vector[5], vector[7], constellation_ini_phase=0.)
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
