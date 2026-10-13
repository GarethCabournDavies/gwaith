"""Write a complete, round-labelled HDF5/CSV catalogue of fitted signals."""
import argparse
import logging
import json
from pathlib import Path
import h5py
import numpy as np
from hwilas import add_hwilas_common_options, init_logging
from hwilas.search.rounds import combine_parameters, rebuild_signal_model


def merge_partitions(paths, output):
    """Merge complete, non-overlapping partitions of one optimisation run."""
    tables, metadata = [], []
    for path in paths:
        with h5py.File(path) as f:
            tables.append({k:f[k][:] for k in f if isinstance(f[k], h5py.Dataset)})
            metadata.append(dict(f.attrs))
    first = metadata[0]
    if 'optimisation_signature' not in first or 'selected_event_count' not in first:
        raise ValueError('Missing partition provenance')
    for table, attrs in zip(tables, metadata):
        if attrs.get('optimisation_signature') != first['optimisation_signature']:
            raise ValueError('Partitions belong to different optimisation runs/settings')
        if attrs.get('selected_event_count') != first['selected_event_count']:
            raise ValueError('Partitions have different globally selected event counts')
        if table.keys() != tables[0].keys() or 'global_event_index' not in table:
            raise ValueError('Partition schemas differ or global event IDs are missing')
        if any(v.ndim != 1 or len(v) != len(table['global_event_index']) for v in table.values()):
            raise ValueError('Invalid partition columns')
    combined = {k:np.concatenate([table[k] for table in tables]) for k in tables[0]}
    indices = combined['global_event_index']
    order = np.argsort(indices)
    if not np.array_equal(indices[order], np.arange(first['selected_event_count'])):
        raise ValueError('Missing or duplicated fitted events; refusing an incomplete merge')
    combined = {k:v[order] for k,v in combined.items()}
    if Path(output).resolve() in {Path(p).resolve() for p in paths}:
        raise ValueError('Merged output must differ from partition inputs')
    Path(output).parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(output, 'w') as f:
        f.attrs.update(first)
        f.attrs['event_fraction'] = '0/1'
        f.attrs['source_parameter_files'] = json.dumps([str(p) for p in paths])
        for k,v in combined.items():
            f[k] = v
    return combined


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    add_hwilas_common_options(parser)
    parser.add_argument('--input-files', nargs='+', required=True)
    parser.add_argument('--output-file', required=True)
    parser.add_argument('--csv-file')
    parser.add_argument('--data-file', help='Original data defining the rebuilt model grid')
    parser.add_argument('--output-timeseries-file', help='Rebuild summed waveforms from retained fits')
    parser.add_argument('--channels', nargs='+', default=['LISA_A','LISA_E','LISA_T'])
    parser.add_argument('--close-event-window', type=float, default=3600.,
                        help='Flag cross-round fitted times within this many seconds; default 3600; retains all events')
    parser.add_argument('--same-round', action='store_true',
                        help='Merge manual event partitions of one round, requiring complete event coverage')
    parser.add_argument('--exclude-later-close-events', action='store_true',
                        help='Exclude later-round events within close-event-window of retained earlier events')
    args = parser.parse_args()
    if bool(args.data_file) != bool(args.output_timeseries_file):
        parser.error('Provide data-file and output-timeseries-file together')
    init_logging(args.verbose)
    if args.same_round:
        table = merge_partitions(args.input_files, args.output_file)
        if args.csv_file:
            from tools.search.candidate_hdf_to_csv import write_optimised_csv
            Path(args.csv_file).parent.mkdir(parents=True, exist_ok=True)
            with h5py.File(args.output_file) as f:
                optimizer = f.attrs.get('optimizer', 'unknown')
            write_optimised_csv(args.csv_file, table, optimizer)
    else:
        table = combine_parameters(args.input_files, args.output_file, args.csv_file, args.close_event_window, args.exclude_later_close_events)
    if args.output_timeseries_file:
        rebuild_signal_model(args.data_file, args.output_file, args.output_timeseries_file, args.channels)
    logging.getLogger(__name__).info('Combined %d fitted signals from %d rounds', len(table['tc']), len(args.input_files))


if __name__ == '__main__':
    main()
