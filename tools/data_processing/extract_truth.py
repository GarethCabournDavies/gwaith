"""
Extract "truth" (injected-signal) data from a Sangria or Mojito HDF5 file and
write it to a standardised HDF5 file.

This is a single entry point that replaces the former ``extract_truth_sangria``
and ``extract_truth_mojito`` executables. The input file type can be given
explicitly with ``--type {sangria,mojito}`` or auto-detected from the file's
structure when the flag is omitted (see :func:`hwilas.data_processing.truth.detect_truth_file_type`).

The truth is read from the ``sky/<source-group>`` group (catalog at
``sky/<group>/cat`` and summed TDI waveform at ``sky/<group>/tdi``) and written
out with the layout defined by :class:`hwilas.data_processing.truth.TruthData`.
"""

import argparse
import logging

from hwilas import add_hwilas_common_options, init_logging
from hwilas.data_processing.truth import (
    SANGRIA_SOURCE_GROUPS,
    detect_truth_file_type,
    load_truth_mojito,
    load_truth_sangria,
)

logger = logging.getLogger(__name__)


def _select_loader(file_type):
    """Return the loader function for a given truth file type."""
    if file_type == "mojito":
        return load_truth_mojito
    return load_truth_sangria


def main():
    parser = argparse.ArgumentParser(
        description="Extract truth (injected-signal) data from a Sangria or Mojito "
                    "HDF5 file and write it to a standardised HDF5 file."
    )
    add_hwilas_common_options(parser)
    parser.add_argument(
        "--data-file",
        required=True,
        help="Path to the input Sangria or Mojito HDF5 file",
    )
    parser.add_argument(
        "--output-file",
        required=True,
        help="Path to the output standardised truth HDF5 file",
    )
    parser.add_argument(
        "--type",
        choices=["sangria", "mojito"],
        default=None,
        help="Type of input file. If omitted, the type is auto-detected from the "
             "file's structure (pass this explicitly if detection is ambiguous).",
    )
    parser.add_argument(
        "--source-group",
        default="mbhb",
        choices=list(SANGRIA_SOURCE_GROUPS),
        help="Source group under 'sky/' to extract (default: mbhb)",
    )
    parser.add_argument(
        "--channels",
        nargs="+",
        default=["LISA_A", "LISA_E", "LISA_T"],
        help="AET channels to include in the output timeseries. "
             "Default: LISA_A LISA_E LISA_T",
    )
    parser.add_argument(
        "--no-timeseries",
        action="store_true",
        help="Do not extract the summed TDI waveform (catalog only)",
    )

    args = parser.parse_args()

    init_logging(args.verbose)

    if args.type is None:
        args.type = detect_truth_file_type(args.data_file)
        logger.info("Auto-detected input file type as '%s'", args.type)
    else:
        logger.info("Using input file type '%s' (as given)", args.type)

    load_truth = _select_loader(args.type)

    logger.info("Loading truth from %s file %s (group=%s)",
                args.type, args.data_file, args.source_group)

    try:
        truth = load_truth(
            args.data_file,
            source_group=args.source_group,
            include_timeseries=not args.no_timeseries,
        )
    except ImportError as exc:
        # Only the Mojito path raises this (missing optional dependency).
        logger.error("%s", exc)
        raise SystemExit(1)

    logger.info("Extracted %d %s source(s); timeseries=%s",
                truth.n_sources, args.source_group,
                truth.timeseries is not None)

    if truth.timeseries is not None:
        # Keep only the requested channels.
        truth.timeseries = {
            ch: truth.timeseries[ch] for ch in args.channels
            if ch in truth.timeseries
        }
        if not truth.timeseries:
            logger.warning("None of the requested channels %s were present; "
                           "writing catalog only", args.channels)
            truth.timeseries = None

    truth.to_hdf5(args.output_file, channels=args.channels)
    logger.info("Done. Wrote truth to %s", args.output_file)


if __name__ == "__main__":
    main()