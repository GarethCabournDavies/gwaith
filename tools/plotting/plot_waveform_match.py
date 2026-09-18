
import argparse
from hwilas import add_hwilas_common_options, add_truth_arguments, init_logging

logger = logging.getLogger(__name__)

def load_arguments():
    parser = argparse.ArgumentParser(
        description="Plot clustered matched-filter SNR peaks."
    )
    add_hwilas_common_options(parser)
    add_truth_arguments(parser)

    return parser.parse_args()

def main():
    args = load_arguments()
    init_logging(args.verbose)

    print(args)

    logger.info("Done!")