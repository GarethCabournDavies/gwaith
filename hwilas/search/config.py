"""Validated TOML configuration for local search and stochastic placement."""
from pathlib import Path
import tomllib
import numpy as np

PRIORS = {
    "total_mass": (3e5, 5e7, "log_uniform"), "q": (1., 10., "uniform"),
    "spin1z": (-.99, .99, "uniform"), "spin2z": (-.99, .99, "uniform"),
    "eclipticlatitude": (-np.pi/2, np.pi/2, "sin_uniform"),
    "eclipticlongitude": (0., 2*np.pi, "uniform"),
    "inclination": (0., np.pi, "cos_uniform"),
    "polarization": (0., np.pi, "uniform"), "coa_phase": (0., 2*np.pi, "uniform"),
}

def load_config(path):
    path = Path(path).resolve()
    with path.open("rb") as f:
        cfg = tomllib.load(f)
    cfg.setdefault("waveform", {})
    w = cfg["waveform"]
    for k, v in dict(delta_t=4., initial_position_deg=-140., laser_frequency=2.816e14,
                     f_lower=3e-5, f_upper=.09, pre_merger_days=1., taper_seconds=3600.).items():
        w.setdefault(k, v)
    if w["delta_t"] <= 0 or not 0 < w["f_lower"] < w["f_upper"] < .5/w["delta_t"]:
        raise ValueError("Waveform band must lie strictly inside Nyquist")
    if 'snr_fraction' not in w and not 0 < w["taper_seconds"] < w["pre_merger_days"]*86400 <= 1.5*86400:
        raise ValueError("Template must fit the 1.5-day pre-merger buffer")
    if not np.isfinite(w['taper_seconds']) or w['taper_seconds'] <= 0:
        raise ValueError('Template taper must be positive and finite')
    if 'snr_fraction' in w and (not 0 < w['snr_fraction'] <= 1
            or w.get('snr_frequency_samples', 32768) < 1024):
        raise ValueError('Invalid SNR retention fraction or frequency grid')
    for section, keys in {"search": ("data_file", "bank_file", "output_dir", "psd_A", "psd_E"),
                          "bank": ("output_file", "psd_A", "psd_E", "resume")}.items():
        for key in keys:
            if cfg.get(section, {}).get(key):
                cfg[section][key] = str((path.parent / cfg[section][key]).resolve())
    unknown = set(cfg.get("prior", {})) - set(PRIORS)
    if unknown:
        raise ValueError(f"Unknown prior parameters: {sorted(unknown)}")
    for key, (lo, hi, dist) in PRIORS.items():
        p = cfg.setdefault("prior", {}).setdefault(key, {})
        for k, v in dict(min=lo, max=hi, distribution=dist).items():
            p.setdefault(k, v)
        lo, hi, dist = p["min"], p["max"], p["distribution"]
        if not np.isfinite([lo, hi]).all() or lo >= hi:
            raise ValueError(f"Invalid limits for {key}")
        if dist not in {"uniform", "log_uniform", "sin_uniform", "cos_uniform"}:
            raise ValueError(f"Unknown distribution for {key}: {dist}")
        if dist == "log_uniform" and lo <= 0:
            raise ValueError(f"Log-uniform {key} must be positive")
        if dist == "sin_uniform" and not -np.pi/2 <= lo < hi <= np.pi/2:
            raise ValueError("sin_uniform limits must lie in [-pi/2, pi/2]")
        if dist == "cos_uniform" and not 0 <= lo < hi <= np.pi:
            raise ValueError("cos_uniform limits must lie in [0, pi]")
        if key == "total_mass" and lo <= 0 or key == "q" and lo < 1:
            raise ValueError("Mass must be positive and q >= 1")
        if key.startswith("spin") and not -1 < lo < hi < 1:
            raise ValueError("Aligned spins must lie strictly inside (-1, 1)")
    return cfg


def draw(rng, n, priors):
    cols = {}
    for name, p in priors.items():
        lo, hi, d = p["min"], p["max"], p["distribution"]
        if d == "log_uniform":
            a = np.exp(rng.uniform(np.log(lo), np.log(hi), n))
        elif d == "sin_uniform":
            a = np.arcsin(rng.uniform(np.sin(lo), np.sin(hi), n))
        elif d == "cos_uniform":
            a = np.arccos(rng.uniform(np.cos(hi), np.cos(lo), n))
        else:
            a = rng.uniform(lo, hi, n)
        cols[name] = a
    M, q = cols.pop("total_mass"), cols.pop("q")
    cols.update(mass1=M*q/(1+q), mass2=M/(1+q), distance=np.full(n, 1e4))
    return [{k: float(v[i]) for k, v in cols.items()} for i in range(n)]
