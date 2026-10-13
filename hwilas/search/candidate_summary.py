"""Inclusive round-labelled seed catalogues and cross-round proximity checks."""
from pathlib import Path
import json
import h5py
import numpy as np
from hwilas.search.statistics import canonical_statistics


def nearby_round_events(times, rounds, window):
    """Identify the nearest event in a different round without rejecting any row."""
    if not np.isfinite(window) or window < 0:
        raise ValueError('Close-event window must be finite and nonnegative')
    times, rounds = np.asarray(times), np.asarray(rounds)
    if not np.all(np.isfinite(times)):
        raise ValueError('Candidate times must be finite')
    nearest = np.full(len(times), -1, dtype=np.int64)
    separation = np.full(len(times), np.nan)
    for i, time in enumerate(times):
        other = np.flatnonzero(rounds != rounds[i])
        if len(other):
            j = other[np.argmin(abs(times[other]-time))]
            nearest[i], separation[i] = j, abs(times[j]-time)
    return dict(close_other_round=separation <= window,
                nearest_other_round_index=nearest,
                nearest_other_round_separation_seconds=separation)


def retain_earlier_round_events(times, rounds, window):
    """Keep all same-round events; reject later events near retained earlier rounds."""
    nearby_round_events(times, rounds, window)
    times, rounds = np.asarray(times), np.asarray(rounds)
    keep = np.ones(len(times), dtype=bool)
    for current in np.unique(rounds):
        earlier = np.flatnonzero(keep & (rounds < current))
        if not len(earlier):
            continue
        for index in np.flatnonzero(rounds == current):
            keep[index] = not np.any(abs(times[earlier]-times[index]) <= window)
    return keep


def combine_candidates(paths, output_file, close_window=3600., exclude_later_close=False):
    """Concatenate candidates preserving source identity and statistic definitions."""
    if Path(output_file).resolve() in {Path(p).resolve() for p in paths}:
        raise ValueError('Combined catalogue must differ from its inputs')
    tables, metadata = [], []
    for path in paths:
        with h5py.File(path) as f:
            table = canonical_statistics({k:v[:] for k,v in f.items()
                if isinstance(v,h5py.Dataset) and v.ndim == 1})
            n = len(table['template_id'])
            if any(len(v) != n for v in table.values()):
                raise ValueError(f'Candidate columns differ in length: {path}')
            if 'end_time' not in table:
                channels = [k[10:] for k in table if k.startswith('statistic_')]
                table['end_time'] = np.mean([table[f'time_{c}'] for c in channels],axis=0)
            if 'statistic' not in table:
                channels = [k[10:] for k in table if k.startswith('statistic_')]
                table['statistic'] = np.sqrt(sum(table[f'statistic_{c}']**2 for c in channels))
            tables.append(table); metadata.append(dict(f.attrs))
    fields = sorted(set().union(*(t.keys() for t in tables)) - {'search_round','round_event_index'})
    combined = {}
    for key in fields:
        dtype = np.result_type(*(t[key].dtype for t in tables if key in t))
        if dtype.kind == 'u': dtype = np.dtype('i8')
        fill = False if dtype.kind == 'b' else -1 if dtype.kind in 'iu' else np.nan
        combined[key] = np.concatenate([t.get(key,np.full(len(t['template_id']),fill,dtype=dtype)) for t in tables])
    combined['search_round'] = np.concatenate([np.full(len(t['template_id']),r,dtype=np.int64) for r,t in enumerate(tables)])
    combined['round_event_index'] = np.concatenate([np.arange(len(t['template_id'])) for t in tables])
    combined['combined_event_index'] = np.arange(len(combined['end_time']), dtype=np.int64)
    excluded_count = 0
    if exclude_later_close:
        keep = retain_earlier_round_events(combined['end_time'], combined['search_round'], close_window)
        excluded_count = int(np.count_nonzero(~keep))
        combined = {k:v[keep] for k,v in combined.items()}
    combined.update(nearby_round_events(combined['end_time'],combined['search_round'],close_window))
    neighbours = combined['nearest_other_round_index']
    valid = neighbours >= 0
    neighbours[valid] = combined['combined_event_index'][neighbours[valid]]
    Path(output_file).parent.mkdir(parents=True,exist_ok=True)
    with h5py.File(output_file,'w') as f:
        f.attrs.update(source_candidate_files=json.dumps([str(p) for p in paths]),close_event_window=close_window, exclude_later_close_events=exclude_later_close, excluded_event_count=excluded_count)
        provenance=f.create_group('round_metadata')
        for r,attrs in enumerate(metadata):provenance.create_group(str(r)).attrs.update(attrs)
        for k,v in combined.items():f[k]=v
    return combined
