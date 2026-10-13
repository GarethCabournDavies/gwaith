"""Collect every round's clustered candidates for global-fit seed selection."""
import argparse
from pathlib import Path
import h5py
from hwilas import add_hwilas_common_options, init_logging
from hwilas.search.candidate_summary import combine_candidates
from tools.search.candidate_hdf_to_csv import write_candidate_csv


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    add_hwilas_common_options(parser)
    parser.add_argument('--input-files',nargs='+',required=True)
    parser.add_argument('--output-file',required=True)
    parser.add_argument('--csv-file',required=True)
    parser.add_argument('--bank-file')
    parser.add_argument('--close-event-window',type=float,default=3600.,help='Cross-round proximity flag in seconds; default 3600; retains all candidates')
    parser.add_argument('--exclude-later-close-events', action='store_true',
                        help='Exclude later-round events within close-event-window of retained earlier events')
    args=parser.parse_args()
    init_logging(args.verbose)
    outputs=[Path(args.output_file).resolve(),Path(args.csv_file).resolve()]
    if len(set(outputs))!=2 or outputs[1] in {Path(p).resolve() for p in args.input_files}:
        parser.error('Outputs must differ from each other and all inputs')
    rows=combine_candidates(args.input_files,args.output_file,args.close_event_window, args.exclude_later_close_events)
    Path(args.csv_file).parent.mkdir(parents=True,exist_ok=True)
    with h5py.File(args.output_file) as f:
        labels=[f['round_metadata'][str(i)].attrs.get('statistic_name',f['round_metadata'][str(i)].attrs.get('statistic','unspecified')) for i in range(len(args.input_files))]
    write_candidate_csv(args.csv_file,rows,args.bank_file,[labels[r] for r in rows['search_round']])


if __name__=='__main__':main()
