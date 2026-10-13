"""Per-template start frequency from FastBHB's aligned-spin timing model."""
import logging
import numpy as np
from scipy.optimize import brentq

logger = logging.getLogger(__name__)


def duration_from_frequency(params, frequency):
    """Return the (2,2) inspiral time in seconds at a template's start frequency."""
    from lisabeta.waveforms.bbh.pyIMRPhenomD import IMRPhenomDh22AmpPhase
    from astropy.constants import G, M_sun, c
    if not np.isfinite(frequency) or frequency <= 0:
        raise ValueError("Template start frequency must be positive and finite")
    mass_seconds = (G*M_sun/c**3).to_value('s')*(params['mass1']+params['mass2'])
    model = IMRPhenomDh22AmpPhase(np.array([min(frequency, 1e-3/mass_seconds)]),
        params['mass1'], params['mass2'], params['spin1z'], params['spin2z'], params['distance'])
    duration = -float(model.compute_toff(frequency))
    if not np.isfinite(duration):
        raise ValueError("Nonfinite template time-to-merger")
    return max(0., duration)


def lower_frequency_for_snr(frequencies, density, fraction=.99):
    """Discard at most 1-fraction**2 of the integrated network SNR squared."""
    from scipy.integrate import cumulative_trapezoid
    frequencies, density = np.asarray(frequencies), np.asarray(density)
    if (frequencies.ndim != 1 or density.shape != frequencies.shape or len(frequencies) < 2
            or not np.isfinite(frequencies).all() or np.any(np.diff(frequencies) <= 0)
            or not np.isfinite(density).all() or np.any(density < 0)
            or not 0 < fraction <= 1):
        raise ValueError("Invalid SNR integration grid or retention fraction")
    cumulative = cumulative_trapezoid(density, frequencies, initial=0.)
    if cumulative[-1] <= 0:
        raise ValueError("Nonpositive network template SNR")
    # Pick the lower bin edge so discretisation cannot discard extra power.
    i = np.searchsorted(cumulative, (1-fraction**2)*cumulative[-1], side='right')-1
    return float(frequencies[max(0, i)])


def lower_frequency_for_duration(params, settings):
    """Solve time-to-merger = pre-merger duration, bounded by the analysis band.

    FastBHB.get_tc uses this same IMRPhenomD (2,2) phase timing function.
    The configured f_lower is a floor; it may shorten a massive system's
    available inspiral below the requested duration.
    """
    from lisabeta.waveforms.bbh.pyIMRPhenomD import IMRPhenomDh22AmpPhase
    floor, upper = settings['f_lower'], settings['f_upper']
    duration = settings['pre_merger_days']*86400.
    from astropy.constants import G, M_sun, c
    total_mass_seconds = (G*M_sun/c**3).to_value('s')*(params['mass1']+params['mass2'])
    # Initialise inside the inspiral, not at an analysis limit above model cutoff.
    initial_frequency = min(floor, 1e-3/total_mass_seconds)
    model = IMRPhenomDh22AmpPhase(np.array([initial_frequency]), params['mass1'], params['mass2'],
                                params['spin1z'], params['spin2z'], params['distance'])
    def residual(f):
        return -float(model.compute_toff(f))-duration
    low, high = residual(floor), residual(upper)
    if not np.isfinite([low, high]).all():
        raise ValueError("Nonfinite template time-to-merger")
    if low <= 0:
        cutoff = float(floor)
    elif high >= 0:
        raise ValueError("Template start frequency exceeds the analysis upper cutoff")
    else:
        cutoff = float(brentq(residual, floor, upper, xtol=1e-12))
    logger.debug("Template masses=%g,%g spins=%g,%g duration=%gs f_lower=%gHz",
                 params['mass1'], params['mass2'], params['spin1z'], params['spin2z'], duration, cutoff)
    return cutoff
