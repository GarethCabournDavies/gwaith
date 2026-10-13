"""Canonical ranking fields with read compatibility for older SNR-named files."""
import numpy as np


def canonical_statistics(rows):
    result = dict(rows)
    for key in list(result):
        if key == 'snr' or key.startswith('snr_'):
            target = 'statistic'+key[3:]
            result.setdefault(target, result[key])
            del result[key]
    return result


def read_statistic(group, channel=None):
    suffix = '' if channel is None else '_'+channel
    for name in ('statistic'+suffix, 'snr'+suffix):
        if name in group:
            return np.asarray(group[name][:])
    if channel is None:
        channels = [key[len('statistic_'):] for key in group if key.startswith('statistic_')]
        if not channels:
            channels = [key[4:] for key in group if key.startswith('snr_')]
        if channels:
            return np.sqrt(sum(read_statistic(group, c)**2 for c in channels))
    raise KeyError('Missing ranking statistic'+suffix)
