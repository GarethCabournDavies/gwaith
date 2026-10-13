"""Window flags, parameter catalogues and aligned multi-round subtraction."""
import json
from contextlib import ExitStack
from pathlib import Path

import h5py
import numpy as np


def previous_window_flags(times, parameter_file):
    """Flag candidates inside an earlier fit window without rejecting them."""
    times = np.asarray(times)
    flags = np.zeros(len(times), dtype=bool)
    indices = np.full(len(times), -1, dtype=np.int64)
    if parameter_file is None:
        return flags, indices
    with h5py.File(parameter_file, 'r') as f:
        if 'optimisation_window_start' in f and 'optimisation_window_end' in f:
            starts, ends = f['optimisation_window_start'][:], f['optimisation_window_end'][:]
        else:
            centres = f['initial_time'][:] if 'initial_time' in f else f['tc'][:]
            starts = centres-f.attrs.get('optimisation_window_before', 172800.)
            ends = centres+f.attrs.get('optimisation_window_after', 172800.)
    for i, time in enumerate(times):
        matches = np.flatnonzero((starts <= time) & (time < ends))
        if len(matches):
            flags[i] = True
            indices[i] = matches[np.argmin(abs((starts[matches]+ends[matches])/2-time))]
    return flags, indices


def combine_parameters(paths, output_file, csv_file=None, close_window=3600., exclude_later_close=False):
    """Keep every fitted event and identify its source round and row."""
    if Path(output_file).resolve() in {Path(path).resolve() for path in paths}:
        raise ValueError('Combined parameter output must differ from inputs')
    tables = []
    metadata = []
    for path in paths:
        with h5py.File(path, 'r') as f:
            n = len(f['tc'])
            table = {key:f[key][:] for key in f if isinstance(f[key], h5py.Dataset)}
            if any(values.ndim != 1 or len(values) != n for values in table.values()):
                raise ValueError(f'Parameter columns must have one value per event: {path}')
            tables.append(table)
            metadata.append(dict(f.attrs))
    fields = sorted(set().union(*(table.keys() for table in tables)) - {'search_round', 'round_event_index'})
    combined = {}
    for field in fields:
        present = [table[field] for table in tables if field in table]
        dtype = np.result_type(*(values.dtype for values in present))
        if dtype.kind == 'u':
            dtype = np.dtype('i8')
        fill = False if dtype.kind == 'b' else -1 if dtype.kind in 'iu' else np.nan
        combined[field] = np.concatenate([table.get(field, np.full(len(table['tc']), fill, dtype=dtype)) for table in tables])
    combined['search_round'] = np.concatenate([np.full(len(table['tc']), i, dtype=np.int64) for i, table in enumerate(tables)])
    combined['round_event_index'] = np.concatenate([np.arange(len(table['tc']), dtype=np.int64) for table in tables])
    from hwilas.search.candidate_summary import nearby_round_events, retain_earlier_round_events
    combined['combined_event_index'] = np.arange(len(combined['tc']), dtype=np.int64)
    excluded_count = 0
    if exclude_later_close:
        keep = retain_earlier_round_events(combined['tc'], combined['search_round'], close_window)
        excluded_count = int(np.count_nonzero(~keep))
        combined = {k:v[keep] for k,v in combined.items()}
    combined.update(nearby_round_events(combined['tc'], combined['search_round'], close_window))
    neighbours = combined['nearest_other_round_index']
    valid = neighbours >= 0
    neighbours[valid] = combined['combined_event_index'][neighbours[valid]]
    fields = sorted(set(combined) - {'search_round', 'round_event_index'})
    Path(output_file).parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(output_file, 'w') as f:
        f.attrs.update(close_event_window=close_window, exclude_later_close_events=exclude_later_close,
                       excluded_event_count=excluded_count)
        f.attrs['source_parameter_files'] = json.dumps([str(path) for path in paths])
        provenance = f.create_group('round_metadata')
        for i, attrs in enumerate(metadata):
            provenance.create_group(str(i)).attrs.update(attrs)
        for field, values in combined.items():
            f.create_dataset(field, data=values)
    if csv_file:
        Path(csv_file).parent.mkdir(parents=True, exist_ok=True)
        from hwilas.search.event_csv import write_optimised_csv
        optimizers = [metadata[r].get('optimizer', 'unknown') for r in combined['search_round']]
        write_optimised_csv(csv_file, combined, optimizers)
    return combined


def validate_grids(data, other, channels):
    """Require identical real-data grids before subtraction or residual checks."""
    for key in ('epoch', 'delta_t'):
        if float(data.attrs[key]) != float(other.attrs[key]):
            raise ValueError(f'Timeseries {key} differs between {data.filename} and {other.filename}')
    for channel in channels:
        if data[channel].shape != other[channel].shape:
            raise ValueError(f'{channel} lengths differ between {data.filename} and {other.filename}')
    key = 'divide_by_laser_frequency'
    if key in data.attrs and key in other.attrs and data.attrs[key] != other.attrs[key]:
        raise ValueError('Timeseries frequency units differ')


def subtract_files(data_file, signal_files, output_file, channels, chunk_size=262144):
    """Subtract each round exactly once, preserving cadence and unit metadata."""
    if Path(output_file).resolve() in {Path(path).resolve() for path in [data_file, *signal_files]}:
        raise ValueError('Subtraction output must differ from its input files')
    with ExitStack() as stack:
        data = stack.enter_context(h5py.File(data_file, 'r'))
        signals = [stack.enter_context(h5py.File(path, 'r')) for path in signal_files]
        for signal in signals:
            validate_grids(data, signal, channels)
        Path(output_file).parent.mkdir(parents=True, exist_ok=True)
        with h5py.File(output_file, 'w') as output:
            output.attrs.update(dict(data.attrs))
            output.attrs['subtracted_signal_files'] = json.dumps([str(path) for path in signal_files])
            output.attrs['subtraction_channels'] = json.dumps(list(channels))
            for channel in channels:
                n = len(data[channel])
                ds = output.create_dataset(channel, shape=(n,), dtype='f8')
                for j in range(0, n, chunk_size):
                    residual = np.asarray(data[channel][j:j+chunk_size], dtype=float)
                    for signal in signals:
                        residual -= signal[channel][j:j+chunk_size]
                    ds[j:j+chunk_size] = residual
            # Channels not fitted by the search retain their original samples.
            for key in data:
                if key not in channels:
                    data.copy(key, output)


def rebuild_signal_model(data_file, parameter_file, output_file, channels):
    """Regenerate the retained physical fits on their saved grids and sum them."""
    from hwilas.search.residual_diagnostics import load_parameters, waveform_window
    if Path(output_file).resolve() in {Path(data_file).resolve(), Path(parameter_file).resolve()}:
        raise ValueError('Model output must differ from inputs')
    fits, metadata = load_parameters(parameter_file)
    with h5py.File(data_file) as data:
        epoch, dt = float(data.attrs['epoch']), float(data.attrs['delta_t'])
        size = len(data[channels[0]])
        if any(len(data[c]) != size for c in channels):
            raise ValueError('Data channels must share a time grid')
        scale = 1. if data.attrs.get('divide_by_laser_frequency', True) else float(data.attrs['laser_frequency'])
        Path(output_file).parent.mkdir(parents=True, exist_ok=True)
        with h5py.File(output_file, 'w') as output:
            output.attrs.update(dict(data.attrs))
            output.attrs['source_parameter_file'] = str(parameter_file)
            output.attrs['fitted_event_count'] = len(fits)
            for c in channels:
                output.create_dataset(c, shape=(size,), dtype='f8', fillvalue=0.)
            for fit, attrs in zip(fits, metadata):
                start, end = fit['optimisation_window_start'], fit['optimisation_window_end']
                j0, j1 = round((start-epoch)/dt), round((end-epoch)/dt)
                if not 0 <= j0 < j1 <= size or not np.isclose(epoch+j0*dt,start,rtol=0.,atol=1e-6) or not np.isclose(epoch+j1*dt,end,rtol=0.,atol=1e-6):
                    raise ValueError('Saved fit window does not lie on the original data grid')
                model = waveform_window(fit, attrs, channels, start, j1-j0, dt, laser_scale=scale)
                for c in channels:
                    output[c][j0:j1] = output[c][j0:j1] + model[c]
