"""Generate short search templates using the common detector waveform interface."""
import numpy as np
from . import get_fd_waveform
from .plugins import reference_frequency
from .short_fastbhb import short_template as fast_template, template_geometry, frequency_template as fast_frequency_template


def short_template(params, settings, modes=None):
    model = settings.get('approximant', 'fastbhb')
    if model == 'fastbhb':
        return fast_template(params, settings, modes=modes)
    dt = settings['delta_t']
    n, merger = template_geometry(params, settings)
    tc = max(2*86400., merger*dt)
    start = tc-merger*dt
    keywords = dict(settings.get('generator_keywords', {}))
    selected = keywords.pop('mode_array', modes)
    if model == 'phenomthm' and selected is not None:
        selected = sorted({(int(l), sign*abs(int(m))) for l,m in selected for sign in (-1,1)})
    generation = dict(params, start_time=start, delta_t=dt, length=n, tc=tc,
        f_lower=settings['f_lower'], f_upper=settings['f_upper'],
        f_ref=reference_frequency(model, settings['f_lower'], settings.get('f_ref')),
        orbit=settings.get('orbit', 'analytic'), data_time_origin=settings.get('data_time_origin',0.))
    if selected is not None:
        generation['mode_array'] = selected
    generation.update(keywords)
    raw = get_fd_waveform(model, tdi=settings.get('tdi','2'), **generation)
    ramp = .5-.5*np.cos(np.pi*np.clip(np.arange(n)*dt/settings['taper_seconds'],0.,1.))
    return {c: np.asarray(series.data)*ramp*settings.get('laser_frequency',1.) for c,series in raw.items()}


def frequency_template(params, frequencies, settings, modes=None):
    if settings.get('approximant','fastbhb') == 'fastbhb':
        return fast_frequency_template(params, frequencies, settings, modes=modes)
    from scipy.interpolate import CubicSpline
    waves = short_template(params, settings, modes=modes)
    dt = settings['delta_t']
    size = 8*len(next(iter(waves.values())))
    grid = np.fft.rfftfreq(size,dt)
    return {c:CubicSpline(grid,dt*np.fft.rfft(wave,n=size))(frequencies) for c,wave in waves.items()}
