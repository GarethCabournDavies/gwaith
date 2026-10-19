"""Short FastBHB grids using direct SSB angles and a shifted orbit/time origin."""
import logging
import numpy as np

logger = logging.getLogger(__name__)
CHANNELS = ("LISA_A", "LISA_E")
MODES = [(2, 2), (2, 1), (3, 3), (3, 2), (4, 4), (4, 3)]


def frequency_template(params, frequencies, settings, modes=None):
    """Evaluate the full analytic-orbit response without a time-window cutoff."""
    import astropy.units as u
    from ldc.lisa import orbits
    from ldc.waveform.lisabeta import FastBHB
    orb = orbits.Orbits.type(dict(nominal_arm_length=2.5e9*u.m,
        initial_rotation=0*u.rad, initial_position=np.radians(settings['initial_position_deg'])*u.rad,
        orbit_type='analytic'))
    gen = FastBHB('MBHB', approx='IMRPhenomHM', T=3*86400., delta_t=settings['delta_t'],
                  orbits=orb, modes=[(2, 2)] if modes is None else modes)
    p = dict(Mass1=params['mass1'], Mass2=params['mass2'], CoalescenceTime=2*86400.,
        Distance=params['distance'], EclipticLatitude=params['eclipticlatitude'],
        EclipticLongitude=params['eclipticlongitude'], Inclination=params['inclination'],
        Polarization=params['polarization'], Spin1=params['spin1z'], Spin2=params['spin2z'],
        PhaseAtCoalescence=params.get('coa_phase', 0.), Cadence=settings['delta_t'])
    gen.wvf_pars.update(minf=settings['f_lower'], maxf=settings['f_upper'],
                        timetomerger_max=None, tmin=None, tmax=None)
    arrays = gen.get_fd_tdiaet(template=p, freqs=frequencies,
                             tdi2=str(settings.get('tdi', '2')) == '2')
    return {c: np.asarray(a)*settings['laser_frequency']
            for c, a in zip(('LISA_A', 'LISA_E', 'LISA_T'), arrays)}


def template_geometry(params, settings):
    """Return sample count and merger index, allowing per-template time support."""
    dt = settings['delta_t']
    if 'pre_merger_seconds' in settings:
        merger = int(np.ceil(settings['pre_merger_seconds']/dt))
        post = int(np.ceil(settings.get('post_merger_seconds', 86400.)/dt))
        if merger < 0 or post <= 0:
            raise ValueError('Invalid template time support')
        return merger+post, merger
    return round(2*86400/dt), round(1.5*86400/dt)


def template(params, t_start, n, settings, modes=None):
    import astropy.units as u
    from ldc.lisa import orbits
    from ldc.waveform.lisabeta import FastBHB
    from ldc.waveform.lisabeta.fast_bhb import ASTRONOMICAL_YEAR
    dt = settings["delta_t"]
    reference_tc = settings.get("reference_tc", 2*86400.)
    shift = params["tc"] - reference_tc
    t0 = t_start - shift
    if t0 < 0:
        raise ValueError("Short waveform grid starts before the generation origin")
    pos = settings["initial_position_deg"] - 360*shift/ASTRONOMICAL_YEAR
    orb = orbits.Orbits.type(dict(nominal_arm_length=2.5e9*u.m,
        initial_rotation=0*u.rad, initial_position=np.radians(pos)*u.rad, orbit_type="analytic"))
    gen = FastBHB("MBHB", approx="IMRPhenomHM", T=t0+n*dt, delta_t=dt,
                  orbits=orb, modes=MODES if modes is None else modes)
    p = dict(Mass1=params["mass1"], Mass2=params["mass2"], CoalescenceTime=reference_tc,
             Distance=params["distance"], EclipticLatitude=params["eclipticlatitude"],
             EclipticLongitude=params["eclipticlongitude"], Inclination=params["inclination"],
             Polarization=params["polarization"], Spin1=params["spin1z"], Spin2=params["spin2z"],
             PhaseAtCoalescence=params.get("coa_phase", 0.), Cadence=dt)
    gen.wvf_pars["minf"] = settings.get("f_lower", 3e-5)
    gen.wvf_pars["maxf"] = settings.get("f_upper", .09)
    gen.wvf_pars['timetomerger_max'] = None
    if settings.get('f_ref') is not None:
        gen.wvf_pars.update(fref_for_phiref=settings['f_ref'], phiref=0.)
    gen.wvf_pars.update(settings.get('generator_keywords', {}))
    logger.debug("FastBHB modes=%s samples=%d cadence=%g orbit_phase=%g", modes, n, dt, pos)
    A, E, T = gen.get_td_tdiaet(template=p, tdi2=(str(settings.get("tdi", "2")) == "2"))
    k0 = round(t0/dt)
    result = {}
    for c, arr in zip(("LISA_A", "LISA_E", "LISA_T"), (A, E, T)):
        if c not in settings.get("channels", CHANNELS):
            continue
        x = np.asarray(arr, float)[k0:k0+n]*settings["laser_frequency"]
        result[c] = np.pad(x, (0, n-len(x)))
    return result


def short_template(params, settings, modes=None):
    """Generate a buffered template with optional per-template inspiral support."""
    dt = settings["delta_t"]
    n, merger = template_geometry(params, settings)
    p = {**params, "tc": params.get("tc", 2*86400.)}
    # Move the generation origin far enough back to include the entire inspiral.
    generation_settings = {**settings, 'reference_tc': max(2*86400., merger*dt)}
    out = template(p, p["tc"]-merger*dt, n, generation_settings, modes)
    k0 = 0 if 'pre_merger_seconds' in settings else merger-round(settings["pre_merger_days"]*86400/dt)
    kt = min(round(settings["taper_seconds"]/dt), merger-k0)
    ramp = .5-.5*np.cos(np.pi*np.arange(kt)/kt)
    for x in out.values():
        x[:k0] = 0
        x[k0:k0+kt] *= ramp
    return out
