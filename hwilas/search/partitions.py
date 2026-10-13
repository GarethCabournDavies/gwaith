"""Shared parsing and selection for manually partitioned searches."""
import numpy as np


def parse_fraction(value):
    """Return a zero-based group index and positive group count."""
    try:
        part, pieces = map(int, value.split('/'))
    except (ValueError, AttributeError):
        raise ValueError('Partition must be N/M') from None
    if pieces < 1 or not 0 <= part < pieces:
        raise ValueError('Partition requires 0 <= N < M')
    return part, pieces


def partition_range(count, fraction='0/1'):
    """Return contiguous template-bank bounds for a zero-based group."""
    part, pieces = parse_fraction(fraction)
    return count * part // pieces, count * (part + 1) // pieces


def event_partition(count, fraction='0/1'):
    """Return interleaved global event indices for a zero-based group."""
    part, pieces = parse_fraction(fraction)
    indices = np.arange(count, dtype=np.int64)
    return indices[indices % pieces == part]
