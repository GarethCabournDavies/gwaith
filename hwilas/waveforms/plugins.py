"""Discover independently installed waveform providers without importing them."""
from importlib import metadata
from functools import lru_cache
import logging

logger = logging.getLogger(__name__)
ENTRY_POINT_GROUP = 'hwilas.waveforms'
BUILTIN_NAMES = frozenset({'fastbhb', 'phenomthm', 'imrphenomthm', 'phenomthm+tdispline'})


def waveform_entry_points():
    """Return case-insensitive names, rejecting ambiguous provider registrations."""
    providers = {}
    for entry in metadata.entry_points(group=ENTRY_POINT_GROUP):
        name = entry.name.casefold()
        if name in BUILTIN_NAMES or name in providers:
            raise ValueError(f'Duplicate or reserved waveform provider name: {entry.name!r}')
        providers[name] = entry
    return providers


def get_external_generator(approximant):
    """Import only the selected provider; propagate its dependency errors."""
    entry = waveform_entry_points().get(str(approximant).casefold())
    if entry is None:
        raise ValueError(f'Approximant {approximant!r} not recognised; external models '
                         f'must register an entry point in {ENTRY_POINT_GROUP!r}')
    logger.debug('Loading waveform provider %s from %s', entry.name, entry.value)
    generator = entry.load()
    if not callable(generator):
        raise TypeError(f'Waveform entry point {entry.name!r} must resolve to a callable')
    return generator


def available_approximants():
    """List canonical built-in and installed external names without loading models."""
    return ('FastBHB', 'PhenomTHM', *(entry.name for entry in waveform_entry_points().values()))


def normalise_approximant(value):
    aliases = {'fastbhb':'fastbhb', 'phenomhm':'fastbhb', 'imrphenomhm':'fastbhb',
               'phenomthm':'phenomthm', 'imrphenomthm':'phenomthm', 'phenomthm+tdispline':'phenomthm'}
    name = str(value).casefold()
    if name in aliases:
        return aliases[name]
    if name in waveform_entry_points():
        return name
    raise ValueError(f'Unknown waveform {value!r}; install its hwilas.waveforms entry point; installed models: {available_approximants()}')


@lru_cache(maxsize=16)
def external_generator(approximant):
    """Keep provider discovery/import outside the per-template evaluation loop."""
    return get_external_generator(approximant)


def reference_frequency(approximant, f_lower, requested=None):
    if requested is not None:
        return float(requested)
    if approximant not in ('fastbhb', 'phenomthm'):
        return float(getattr(external_generator(approximant), 'default_f_ref', f_lower))
    return float(f_lower)


def prepare_generator(approximant, *, orbit, data_time_origin, start_time, end_time, keywords=None):
    """Let an external provider prepare immutable resources once in the parent process."""
    keywords = dict(keywords or {})
    if approximant not in ('fastbhb', 'phenomthm'):
        prepare = getattr(external_generator(approximant), 'prepare', None)
        if prepare is not None:
            keywords = prepare(orbit=orbit, data_time_origin=data_time_origin,
                               start_time=start_time, end_time=end_time, **keywords)
    return keywords
