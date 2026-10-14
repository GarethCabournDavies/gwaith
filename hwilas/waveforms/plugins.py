"""Discover independently installed waveform providers without importing them."""
from importlib import metadata
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
