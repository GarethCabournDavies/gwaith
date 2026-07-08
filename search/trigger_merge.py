"Take trigger files and merge them into a single file."
"Apply per-template clustering"

import argparse
import h5py
import logging
import numpy as np

from matplotlib import pyplot as plt
from matplotlib import use
use('Agg')

import pycbc

logger = logging.getLogger(__name__)


parser = argparse.ArgumentParser()
pycbc.add_common_pycbc_options(parser)
parser.add_argument(
    "--trigger-files",
    nargs='+',
    required=True,
    help="Files with trigger information to merge",
)
parser.add_argument(
    "--output-file",
    required=True,
    help="File to final list of output triggers"
)
parser.add_argument(
    "--testing-plot",
    help="Generate plots used for testing in this file"
)

args = parser.parse_args()

pycbc.init_logging(args.verbose)

unique_template_ids = set()
template_ids_per_file = {}

n_triggers = 0

# First - find out how many triggers there are as a total in all files
for trigger_filename in args.trigger_files:
    logger.info("Processing file %s", trigger_filename)
    with h5py.File(trigger_filename, 'r') as trigger_file:
        template_ids = np.unique(trigger_file['template_id'])
        n_triggers += trigger_file['template_id'].size
        template_ids_per_file[trigger_filename] = list(template_ids)
        unique_template_ids.update(template_ids)


logger.info("Total template ids: %d", len(unique_template_ids))
logger.info("Total triggers: %d", n_triggers)

curr_size = 0
if args.testing_plot is not None:
    fig, ax = plt.subplots()

with h5py.File(args.output_file, 'w') as output_file:
    tid_ds = output_file.create_dataset(
        "template_id",
        shape=(n_triggers,),
        dtype=np.int32,
        compression='gzip',
    )
    time_ds = output_file.create_dataset(
        "end_time",
        shape=(n_triggers,),
        dtype=float,
        compression='gzip',
    )
    snr_ds = output_file.create_dataset(
        "snr",
        shape=(n_triggers,),
        dtype=float,
        compression='gzip',
    )

    for trigger_filename in args.trigger_files:
        with h5py.File(trigger_filename, 'r') as trigger_file:
            trigger_tids = trigger_file['template_id'][:]
            trigger_times = trigger_file['end_time'][:]
            trigger_snrs = trigger_file['snr'][:]
            this_filesize = trigger_snrs.size

            tid_ds[curr_size:curr_size + this_filesize] = trigger_tids
            time_ds[curr_size:curr_size + this_filesize] = trigger_times
            snr_ds[curr_size:curr_size + this_filesize] = trigger_snrs

        if args.testing_plot is not None:
            ax.scatter(
                trigger_times,
                trigger_snrs,
                c=trigger_tids,
                alpha=0.1,
            )

        curr_size += this_filesize

if args.testing_plot is not None:
    fig.savefig(args.testing_plot)
    plt.close(fig)

    
logger.info("Done!")
