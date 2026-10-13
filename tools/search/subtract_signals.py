"""Subtract aligned search models while preserving the original data metadata."""
import argparse
from hwilas import add_hwilas_common_options, init_logging
from hwilas.search.rounds import subtract_files


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    add_hwilas_common_options(parser)
    parser.add_argument('--data', required=True)
    parser.add_argument('--signal-files', nargs='+', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--channels', nargs='+', default=['LISA_A','LISA_E','LISA_T'])
    args = parser.parse_args()
    init_logging(args.verbose)
    subtract_files(args.data, args.signal_files, args.output, args.channels)


if __name__ == '__main__':
    main()
