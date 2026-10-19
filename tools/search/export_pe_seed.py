"""Create one named PE seed YAML for one search-result row."""
import argparse
from pathlib import Path

from hwilas.search.pe_export import export_pe_seed


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--parameters', type=Path, required=True)
    parser.add_argument('--row-index', type=int, required=True, help='Zero-based row in descending-statistic order, matching the final CSV')
    parser.add_argument('--target-time-origin', type=float, required=True,
                        help='Absolute orbit seconds corresponding to PE time zero')
    parser.add_argument('--native-phase-convention',
                        choices=['pi-over-2-minus-phase', 'direct'],
                        help='Required for external BBHx providers; inspect the provider')
    parser.add_argument('--out', type=Path, required=True, help='Exact output YAML filename')
    args = parser.parse_args()
    try:
        export_pe_seed(args.parameters, args.row_index, args.out, args.target_time_origin,
                       args.native_phase_convention)
    except ValueError as error:
        parser.error(str(error))


if __name__ == '__main__':
    main()
