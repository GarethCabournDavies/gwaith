"""HDF5 bank interchange shared by placement and local search."""
import h5py
import numpy as np

KEYS = ("mass1", "mass2", "spin1z", "spin2z", "eclipticlatitude", "eclipticlongitude",
        "inclination", "polarization", "coa_phase", "distance")


def load_bank(path):
    with h5py.File(path) as f:
        cols = {k: f[k][:] for k in KEYS}
        if "f_lower" in f:
            cols["f_lower"] = f["f_lower"][:]
        if 'pre_merger_seconds' in f:
            cols['pre_merger_seconds'] = f['pre_merger_seconds'][:]
        n = len(cols["mass1"])
        if any(len(v) != n for v in cols.values()) or not n:
            raise ValueError("Bank columns must have equal, nonzero lengths")
    return [{k: float(v[i]) for k, v in cols.items()} for i in range(n)]


def write_bank(path, params, settings, **attrs):
    with h5py.File(path, "w") as f:
        for k in KEYS:
            f[k] = np.array([p[k] for p in params], float)
        n = len(params)
        for k, v in dict(mode_array=22., f_final=settings["f_upper"],
                         tc=2*86400., t_obs_start=settings["pre_merger_days"]*86400.).items():
            f[k] = np.full(n, v)
        f["f_lower"] = np.array([p.get("f_lower", settings["f_lower"]) for p in params], float)
        if params and all('pre_merger_seconds' in p for p in params):
            f['pre_merger_seconds'] = np.array([p['pre_merger_seconds'] for p in params], float)
            f['t_obs_start'][:] = f['pre_merger_seconds'][:]
        f["approximant"] = np.array([settings.get('approximant','FastBHB')]*n,
                                    dtype=h5py.string_dtype())
        f["total_mass"] = f["mass1"][:]+f["mass2"][:]
        f["q"] = f["mass1"][:]/f["mass2"][:]
        f.attrs.update(**attrs)
