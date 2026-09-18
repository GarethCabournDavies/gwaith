import argparse
import logging

import h5py
import numpy as np
from gwaith.io import MultiOptionAction

from hwilas import add_hwilas_common_options, init_logging


logger = logging.getLogger(__name__)

def main():
    parser = argparse.ArgumentParser()

    add_hwilas_common_options(parser)
    parser.add_argument(
        "--trigger-merge-files",
        required=True,
        nargs=2,
        action=MultiOptionAction,
        help="Merged trigger files ",
    )

    parser.add_argument(
        "--coincidence-window",
        type=float,
        default=7200,
        help="Window to use when assessing coincidence, in seconds. Default 100"
    )
    parser.add_argument(
        "--output-file",
        required=True,
        help="File to store output peaks in SNR"
    )

    parser.add_argument(
        "--expected-snr-file",
        help="File containing the optimal SNRs / fitting factors for the MBHBs in Sangria dataset"
    )

    args = parser.parse_args()

    init_logging(args.verbose)

    template_ids = {}
    channels = set()

    for channel, trigger_filename in args.trigger_merge_files.items():
        with h5py.File(trigger_filename, 'r') as trigger_file:
            tids = np.unique(trigger_file['template_id'])
            template_ids[channel] = tids
            channels.update([channel])

    channels = sorted(channels)

    total_template_ids = set()
    for channel in channels:
        total_template_ids.update(template_ids[channel])

    shared_template_ids = set(template_ids[channels[0]])
    for channel in channels[1:]:
        shared_template_ids = shared_template_ids.intersection(template_ids[channel])



    logger.info("Total template ids: %d", len(total_template_ids))
    logger.info("Shared template ids: %d", len(shared_template_ids))

    logger.info("Channels: %s", channels)

    INITIAL_COINC_SIZE = 10000
    coincident_triggers = {
        "template_id": np.empty(INITIAL_COINC_SIZE, dtype=np.int32),
    }
    for channel in channels:
        coincident_triggers.update({
            f"time_{channel}": np.empty(INITIAL_COINC_SIZE, dtype=np.float32),
            f"snr_{channel}": np.empty(INITIAL_COINC_SIZE, dtype=np.float32),
            f"trigger_id_{channel}": np.empty(INITIAL_COINC_SIZE, dtype=np.int32),
        })
    n_coincs = 0
    allocated_coincs = INITIAL_COINC_SIZE

    # Load all trigger data into memory
    triggers = {
        channel: {}
        for channel in channels
    }

    for channel in channels:
        trigger_filename =  args.trigger_merge_files[channel]
        with h5py.File(trigger_filename, 'r') as trigger_file:
            triggers[channel]['template_id'] = trigger_file['template_id'][:]
            triggers[channel]['end_time'] = trigger_file['end_time'][:]
            triggers[channel]['snr'] = trigger_file['snr'][:]

    # Loop through the template IDs and make coincidence
    for template_idx in shared_template_ids:
        logger.info("Processing template %d", template_idx)

        indices_0 = np.where(triggers[channels[0]]['template_id'] == template_idx)[0]
        indices_1 = np.where(triggers[channels[1]]['template_id'] == template_idx)[0]

        if not indices_0.size or not indices_1.size:
            continue

        trigger_times_0 = triggers[channels[0]]['end_time'][indices_0]
        trigger_times_1 = triggers[channels[1]]['end_time'][indices_1]
        trigger_snrs_0 = triggers[channels[0]]['snr'][indices_0]
        trigger_snrs_1 = triggers[channels[1]]['snr'][indices_1]

        for local_idx_0, time_0 in enumerate(trigger_times_0):
            left = np.searchsorted(trigger_times_1, time_0 - args.coincidence_window)
            right = np.searchsorted(trigger_times_1, time_0 + args.coincidence_window)

            if (right - left) == 1:
                # Coincident triggers!
                logger.debug(
                    "Trigger at time %.1f in channel %s is within "
                    "coincidence window of trigger in %s at time %.1f",
                    time_0,
                    channels[0],
                    channels[1],
                    trigger_times_1[left],
                )

                if n_coincs >= allocated_coincs:
                    new_size = allocated_coincs + INITIAL_COINC_SIZE
                    logger.info("Resizing coincident trigger arrays to %d", new_size)
                    for key in coincident_triggers:
                        coincident_triggers[key].resize(new_size, refcheck=False)
                    allocated_coincs = new_size

                global_idx_0 = indices_0[local_idx_0]
                global_idx_1 = indices_1[left]

                coincident_triggers['template_id'][n_coincs] = template_idx
                coincident_triggers[f'trigger_id_{channels[0]}'][n_coincs] = global_idx_0
                coincident_triggers[f'trigger_id_{channels[1]}'][n_coincs] = global_idx_1
                coincident_triggers[f'time_{channels[0]}'][n_coincs] = time_0
                coincident_triggers[f'snr_{channels[0]}'][n_coincs] = trigger_snrs_0[local_idx_0]
                coincident_triggers[f'time_{channels[1]}'][n_coincs] = trigger_times_1[left]
                coincident_triggers[f'snr_{channels[1]}'][n_coincs] = trigger_snrs_1[left]
                n_coincs += 1

            elif (right - left) > 1:
                # This case is not handled yet.
                # It means multiple triggers from channel 1 fall within the
                # window of a single trigger from channel 0.
                raise RuntimeError(
                    "Multiple triggers found within the coincidence window"
                )

    # Trim arrays to the actual number of coincidences
    for key in coincident_triggers:
        coincident_triggers[key] = coincident_triggers[key][:n_coincs]

    logger.info(
        "Found %d coincident triggers",
        n_coincs
    )

    # Now lets do some clustering - cluster by qstatistic
    # For now the statistic is the quadratic sum of SNRs
    # and the time is the midpoint of the trigger times
    # We may want something more complicated in the future
    logger.info("Clustering")

    coinc_time = (
        coincident_triggers[f'time_{channels[0]}'] + coincident_triggers[f'time_{channels[1]}']
    ) / 2
    coinc_stat = (
        coincident_triggers[f'snr_{channels[0]}'] ** 2 + coincident_triggers[f'snr_{channels[1]}'] ** 2
    ) ** 0.5
    # coinc_tids = coincident_triggers['template_id'][:]

    n_coincs_before_clustering = len(coinc_time)

    # Now, perform a standard greedy clustering on the coincs
    # First, sort by statistic - loudest first
    indices = np.argsort(coinc_stat)[::-1]

    keep = np.ones(n_coincs_before_clustering, dtype=bool)
    for sorted_coinc_idx in range(n_coincs_before_clustering):
        coinc_idx = indices[sorted_coinc_idx]
        if not keep[coinc_idx]:
            continue
        within_window = np.abs(coinc_time - coinc_time[coinc_idx]) < args.coincidence_window
        keep[within_window] = False
        keep[coinc_idx] = True

    clustered_coincident_triggers = {}
    for key in coincident_triggers:
        clustered_coincident_triggers[key] = coincident_triggers[key][keep]

    logger.info(
        "%d out of %d coincidences remain after clustering",
        clustered_coincident_triggers['template_id'].size,
        n_coincs_before_clustering
    )

    logger.info("Outputting to file")
    with h5py.File(args.output_file, 'w') as output_trigger_file:
        for key, value in clustered_coincident_triggers.items():
            output_trigger_file.create_dataset(
                key,
                data=value,
                compression='gzip'
            )


    logger.info("Done!")

if __name__ == "__main__":
    main()