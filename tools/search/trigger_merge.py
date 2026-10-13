"""Merge trigger partitions, preserving channel statistics and clustering loudest first."""
import argparse
import logging
from hwilas.search.statistics import canonical_statistics, read_statistic
import h5py
import numpy as np
from hwilas import add_hwilas_common_options, init_logging

logger = logging.getLogger(__name__)


def cluster_rows(rows, window, across_templates=False):
    """Keep the loudest candidate in each time neighbourhood."""
    rows = canonical_statistics(rows)
    keep = np.zeros(len(rows['statistic']), dtype=bool)
    groups = [None] if across_templates else np.unique(rows['template_id'])
    for group in groups:
        indices = (np.arange(len(keep)) if group is None
                   else np.flatnonzero(rows['template_id'] == group))
        selected = []
        for idx in indices[np.argsort(-rows['statistic'][indices], kind='stable')]:
            if all(abs(rows['end_time'][idx]-rows['end_time'][previous]) >= window for previous in selected):
                selected.append(idx)
                keep[idx] = True
    return {key: value[keep] for key, value in rows.items()}


def write_rows(path, rows, attrs):
    with h5py.File(path, 'w') as f:
        f.attrs.update(attrs)
        for key, value in rows.items():
            f.create_dataset(key, data=value, compression='gzip')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    add_hwilas_common_options(parser)
    parser.add_argument('--trigger-files', nargs='+', required=True)
    parser.add_argument('--output-file', required=True)
    parser.add_argument('--unclustered-output-file')
    parser.add_argument('--cluster-window', type=float, default=7200.)
    parser.add_argument('--cluster-across-templates', action='store_true',
                        help='Cluster network candidates across the whole bank')
    args = parser.parse_args()
    if not np.isfinite(args.cluster_window) or args.cluster_window < 0:
        parser.error('Cluster window must be finite and nonnegative')
    init_logging(args.verbose)
    parts, intervals, complete, attrs = [], [], True, {}
    for path in args.trigger_files:
        with h5py.File(path) as f:
            part = {key: f[key][:] for key in f if isinstance(f[key], h5py.Dataset)}
            part = canonical_statistics(part)
            if not {'template_id', 'end_time', 'statistic'} <= part.keys():
                raise ValueError('Missing trigger columns')
            if parts and part.keys() != parts[0].keys():
                raise ValueError('Trigger partition schemas differ')
            if any(value.ndim != 1 or len(value) != len(part['statistic']) for value in part.values()):
                raise ValueError('Trigger columns must have equal one-dimensional lengths')
            if not np.isfinite(part['statistic']).all() or not np.isfinite(part['end_time']).all():
                raise ValueError('Trigger times and statistics must be finite')
            parts.append(part)
            if 'analysis_intervals' in f.attrs:
                intervals.extend(f.attrs['analysis_intervals'])
            else:
                complete = False
            for key in ('statistic', 'channels'):
                if key in f.attrs:
                    if key in attrs and not np.array_equal(attrs[key], f.attrs[key]):
                        raise ValueError('Trigger statistics or channels differ between partitions')
                    attrs[key] = f.attrs[key]
    rows = {key: np.concatenate([p[key] for p in parts]) for key in parts[0]}
    order = np.lexsort((rows['end_time'], rows['template_id']))
    rows = {key: values[order] for key, values in rows.items()}
    # Partition-local IDs become unique IDs in the merged candidate list.
    for key in rows:
        if key.startswith('trigger_id_'):
            rows[key] = np.arange(len(order), dtype=np.int64)
    if complete:
        attrs['analysis_intervals'] = np.asarray(intervals).reshape(-1, 2)
    if args.unclustered_output_file:
        write_rows(args.unclustered_output_file, rows, attrs)
    clustered = cluster_rows(rows, args.cluster_window, args.cluster_across_templates)
    attrs['cluster_window'] = args.cluster_window
    attrs['cluster_across_templates'] = args.cluster_across_templates
    write_rows(args.output_file, clustered, attrs)
    logger.info('Merged %d triggers; retained %d after clustering', len(rows['statistic']), len(clustered['statistic']))


if __name__ == '__main__':
    main()
