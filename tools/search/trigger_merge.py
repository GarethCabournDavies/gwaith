"Take trigger files and merge them into a single file."
"Apply per-template clustering"

import argparse
import logging

import h5py
import numpy as np
from matplotlib import pyplot as plt
from matplotlib import use

use('Agg')


from hwilas import add_hwilas_common_options, init_logging

logger = logging.getLogger(__name__)

def main():
    parser = argparse.ArgumentParser()
    add_hwilas_common_options(parser)
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
    parser.add_argument(
        "--cluster-window",
        type=float,
        default=7200.0,
        help="Cluster triggers per template within this time window in seconds"
    )

    args = parser.parse_args()

    init_logging(args.verbose)

    all_tids = []
    all_times = []
    all_snrs = []

    for trigger_filename in args.trigger_files:
        logger.info("Processing file %s", trigger_filename)
        with h5py.File(trigger_filename, 'r') as trigger_file:
            all_tids.append(trigger_file['template_id'][:])
            all_times.append(trigger_file['end_time'][:])
            all_snrs.append(trigger_file['snr'][:])

    if len(all_tids) == 0:
        raise ValueError("No trigger files provided")

    merged_tids = np.concatenate(all_tids).astype(np.int32, copy=False)
    merged_times = np.concatenate(all_times).astype(np.float64, copy=False)
    merged_snrs = np.concatenate(all_snrs).astype(np.float64, copy=False)

    logger.info("Total template ids before clustering: %d", np.unique(merged_tids).size)
    logger.info("Total triggers before clustering: %d", merged_tids.size)

    # Sort by template id then time so downstream coincidence searchsorted logic
    # sees monotonic per-template times.
    order = np.lexsort((merged_times, merged_tids))
    merged_tids = merged_tids[order]
    merged_times = merged_times[order]
    merged_snrs = merged_snrs[order]

    keep = np.zeros(merged_tids.size, dtype=bool)
    for template_id in np.unique(merged_tids):
        indices = np.where(merged_tids == template_id)[0]
        if indices.size == 0:
            continue
        times = merged_times[indices]
        snrs = merged_snrs[indices]
        local_keep = np.ones(indices.size, dtype=bool)
        loudest_first = np.argsort(snrs)[::-1]
        for local_idx in loudest_first:
            if not local_keep[local_idx]:
                continue
            within = np.abs(times - times[local_idx]) < args.cluster_window
            local_keep[within] = False
            local_keep[local_idx] = True
        keep[indices[local_keep]] = True

    merged_tids = merged_tids[keep]
    merged_times = merged_times[keep]
    merged_snrs = merged_snrs[keep]

    logger.info("Total triggers after clustering: %d", merged_tids.size)

    if args.testing_plot is not None:
        fig, ax = plt.subplots()

    with h5py.File(args.output_file, 'w') as output_file:
        n_triggers = merged_tids.size
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
        tid_ds[:] = merged_tids
        time_ds[:] = merged_times
        snr_ds[:] = merged_snrs

        if args.testing_plot is not None and n_triggers > 0:
            ax.scatter(
                merged_times,
                merged_snrs,
                c=merged_tids,
                alpha=0.1,
            )

    if args.testing_plot is not None:
        fig.savefig(args.testing_plot)
        plt.close(fig)

        
    logger.info("Done!")

if __name__ == "__main__":
    main()