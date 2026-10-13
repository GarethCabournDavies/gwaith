"""Combine processed chunks on a common grid, adding overlaps and zero-filling gaps."""
import argparse
from hwilas import add_hwilas_common_options,init_logging
from hwilas.data_processing.sum_timeseries import sum_files


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    add_hwilas_common_options(parser)
    parser.add_argument('--data-file',nargs='*',default=[])
    parser.add_argument('--output-file',required=True)
    parser.add_argument('--channels',nargs='+',default=['LISA_A','LISA_E','LISA_T'])
    parser.add_argument('--start-time',type=float,help='Output epoch; otherwise earliest input epoch')
    parser.add_argument('--end-time',type=float,help='Exclusive output end; otherwise latest input end')
    parser.add_argument('--delta-t',type=float,help='Output cadence; otherwise input cadence')
    parser.add_argument('--block-size',type=int,default=262144,help='Samples per streaming addition')
    parser.add_argument('--multiplication-factors','--multiplication-factor',nargs='*',type=float,default=[],
                        help='0 values means 1; one value applies to all files; otherwise one value per file in input order')
    args=parser.parse_args();init_logging(args.verbose)
    sum_files(args.data_file,args.output_file,args.channels,args.start_time,args.end_time,
              args.delta_t,args.block_size,args.multiplication_factors)


if __name__=='__main__':main()
