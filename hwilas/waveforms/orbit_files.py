"""Cache ESA OEM trajectories in the ecliptic HDF5 layout used by file responses."""
import hashlib
import json
import logging
import os
from pathlib import Path
from tempfile import NamedTemporaryFile

import h5py
import numpy as np
from platformdirs import user_cache_dir

from hwilas.data_processing.truth import OBLIQUITY_RAD

logger = logging.getLogger(__name__)
LINKS = [12, 23, 31, 13, 32, 21]
ROTATION = np.array([[1., 0., 0.], [0., np.cos(OBLIQUITY_RAD), np.sin(OBLIQUITY_RAD)],
                     [0., -np.sin(OBLIQUITY_RAD), np.cos(OBLIQUITY_RAD)]])


def esa_oem_paths(files=None, version='2.0.1'):
    """Resolve local files, downloading missing included files into lisaorbits' cache."""
    if files is None:
        from lisaorbits import OEMOrbits
        from lisaorbits.oem import _ESA_REPOSITORY
        names = OEMOrbits._get_included_paths('esa-trailing', version=version)
        files = [_ESA_REPOSITORY.fetch(name, progressbar=False) for name in names]
    if len(files) != 3:
        raise ValueError('Provide three OEM files in spacecraft 1, 2, 3 order')
    paths = tuple(Path(path).expanduser().resolve() for path in files)
    for path in paths:
        if not path.is_file():
            raise FileNotFoundError(path)
    return paths


def cached_esa_orbit_file(data_time_origin, end_time, *, oem_files=None,
                          cache_dir=None, delta_t=3600., version='2.0.1'):
    """Convert once per source content, frame, origin, coverage and sample cadence."""
    origin, stop, dt = float(data_time_origin), float(end_time), float(delta_t)
    if not np.isfinite([origin, stop, dt]).all() or min(stop, dt) <= 0:
        raise ValueError('Finite origin and positive orbit duration/cadence are required')
    paths = esa_oem_paths(oem_files, version)
    sources = []
    for path in paths:
        with path.open('rb') as stream:
            sources.append(dict(path=str(path), sha256=hashlib.file_digest(stream, 'sha256').hexdigest()))
    size = int(np.ceil(stop/dt))+1
    manifest = dict(format_version=1, sources=sources, data_time_origin=origin,
                    delta_t=dt, size=size, frame='ecliptic', links=LINKS,
                    obliquity=float(OBLIQUITY_RAD))
    encoded = json.dumps(manifest, sort_keys=True)
    digest = hashlib.sha256(encoded.encode()).hexdigest()
    cache = Path(cache_dir or os.environ.get('HWILAS_ORBIT_CACHE',
                     str(Path(user_cache_dir('hwilas'))/'orbits'))).expanduser()
    cache.mkdir(parents=True, exist_ok=True)
    output = cache/f'esa-{digest}.h5'
    if output.is_file():
        with h5py.File(output, 'r') as saved:
            if saved.attrs.get('manifest') != encoded:
                raise ValueError(f'Orbit cache metadata mismatch: {output}')
        logger.debug('Reusing converted ESA orbits: %s', output)
        return output
    from lisaorbits import OEMOrbits
    orbit = OEMOrbits(*map(str, paths))
    times = origin+np.arange(size)*dt
    if times[0]-1000 < orbit.t_start or times[-1]+1000 > orbit.t_end:
        raise ValueError('ESA OEM coverage excludes the requested detector timeline')
    arrays = dict(x=orbit.compute_position(times) @ ROTATION.T,
                  v=orbit.compute_velocity(times) @ ROTATION.T,
                  n=orbit.compute_unit_vector(times, LINKS) @ ROTATION.T,
                  ltt=orbit.compute_ltt(times, LINKS))
    if any(not np.isfinite(array).all() for array in arrays.values()):
        raise ValueError('Converted ESA trajectories contain nonfinite samples')
    if not np.allclose(np.linalg.norm(arrays['n'], axis=-1), 1., atol=1e-9, rtol=0.):
        raise ValueError('ESA link vectors are not unit length')
    if np.any(arrays['ltt'] <= 0):
        raise ValueError('ESA light travel times must be positive')
    temporary = None
    try:
        with NamedTemporaryFile(dir=cache, suffix='.h5', delete=False) as temp:
            temporary = Path(temp.name)
        with h5py.File(temporary, 'w') as saved:
            saved.attrs.update(t0=0., dt=dt, size=size, data_time_origin=origin,
                               frame='ecliptic', manifest=encoded, cache_key=digest)
            for key, value in arrays.items():
                saved['tcb/'+key] = value
        os.replace(temporary, output)
    finally:
        if temporary is not None and temporary.exists():
            temporary.unlink()
    logger.info('Cached converted ESA orbits: %s', output)
    return output
