"""Export merged candidates or optimised events from saved HDF5 to shareable CSV."""
import argparse
import logging
from pathlib import Path
from hwilas.search.statistics import canonical_statistics, read_statistic
import h5py
import numpy as np
from hwilas import add_hwilas_common_options, init_logging

logger = logging.getLogger(__name__)


from hwilas.search.event_csv import write_candidate_csv, write_optimised_csv


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    add_hwilas_common_options(parser)
    parser.add_argument('--input-file', required=True)
    parser.add_argument('--output-file', required=True)
    parser.add_argument('--previous-event-files', nargs='+', help='Earlier event HDFs used to exclude close events from this CSV')
    parser.add_argument('--exclusion-window', type=float, default=3600., help='Earlier-event exclusion in seconds; default 3600')
    parser.add_argument('--bank-file', help='Optional template parameters for candidate exports')
    args = parser.parse_args()
    init_logging(args.verbose)
    if Path(args.input_file).resolve() == Path(args.output_file).resolve():
        parser.error('CSV output must differ from the input HDF5')
    with h5py.File(args.input_file) as f:
        rows = {k: v[:] for k, v in f.items() if isinstance(v, h5py.Dataset) and v.ndim == 1}
        attrs = dict(f.attrs)
    if not rows or len({len(v) for v in rows.values()}) != 1:
        raise ValueError('Event columns must have equal lengths')
    if args.previous_event_files:
        from hwilas.search.candidate_summary import retain_earlier_round_events
        def event_times(table):
            if 'tc' in table: return table['tc']
            if 'end_time' in table: return table['end_time']
            channels = [k[10:] for k in canonical_statistics(table) if k.startswith('statistic_')]
            return np.mean([table[f'time_{c}'] for c in channels], axis=0)
        earlier = []
        for path in args.previous_event_files:
            with h5py.File(path) as f:
                earlier.extend(event_times({k:v[:] for k,v in f.items() if isinstance(v,h5py.Dataset) and v.ndim==1}))
        times = event_times(rows)
        keep = retain_earlier_round_events(np.r_[earlier,times], np.r_[np.zeros(len(earlier)),np.ones(len(times))], args.exclusion_window)[len(earlier):]
        rows.setdefault('global_event_index' if 'optimised_stat' in rows else 'combined_event_index', np.arange(len(times)))
        rows = {k:v[keep] for k,v in rows.items()}
    Path(args.output_file).parent.mkdir(parents=True, exist_ok=True)
    if 'optimised_stat' in rows:
        write_optimised_csv(args.output_file, rows, attrs.get('optimizer', 'unknown'))
    else:
        statistic = attrs.get('statistic_name', attrs.get('statistic', 'legacy peak coincidence' if 'end_time' not in rows else 'legacy SNR'))
        write_candidate_csv(args.output_file, rows, args.bank_file, statistic)


if __name__ == '__main__':
    main()
