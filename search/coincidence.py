import h5py

import argparse
import logging
import numpy as np
from matplotlib import use
use('Agg')
from matplotlib import pyplot as plt

import pycbc
from pycbc.types import MultiDetOptionAction  # Custom action for argparse
from pycbc.events.threshold_cpu import parallel_thresh_cluster

parser = argparse.ArgumentParser()

pycbc.add_common_pycbc_options(parser)
parser.add_argument(
    "--trigger-merge-files",
    required=True,
    nargs=2,
    action=MultiDetOptionAction,
    help="Merged trigger files ",
)

parser.add_argument(
    "--coincidence-window",
    type=float,
    default=3600,
    help="Window to use when assessing coincidence, in seconds. Default 100"
)
parser.add_argument(
    "--output-file",
    required=True,
    help="File to store output peaks in SNR"
)
parser.add_argument(
    "--num-timeslides",
    type=int,
    default=0,
    help="Number of timeslides to perform on each side of zero lag. "
         "Each slide is one coincidence window."
)
parser.add_argument(
    "--testing-plots",
    action='store_true',
    help="Generate plots used for testing"
)

parser.add_argument(
    "--expected-snr-file",
    help="File containing the optimal SNRs / fitting factors for the MBHBs in Sangria dataset"
)

args = parser.parse_args()

logger = logging.getLogger(__name__)

# Initialize logging for the PyCBC library
if args.verbose is None:
    # Lets have info logging on by default for now
    pycbc.init_logging(1)
else:
    # Adding --verbose adds to the debug logging
    pycbc.init_logging(args.verbose + 1)

template_ids = {}
channels = set()

for channel, trigger_filename in args.trigger_merge_files.items():
    with h5py.File(trigger_filename, 'r') as trigger_file:
        tids = np.unique(trigger_file['template_id'])
        template_ids[channel] = tids
        channels.update([channel])

channels = list(channels)

total_template_ids = set()
for channel in channels:
    total_template_ids.update(template_ids[channel])

shared_template_ids = set(template_ids[channels[0]])
for channel in channels[1:]:
    shared_template_ids = shared_template_ids.intersection(template_ids[channel])



logger.info("Total template ids: %d", len(total_template_ids))
logger.info("Shared template ids: %d", len(shared_template_ids))

logger.info("Channels: %s", channels)

coincident_triggers = {
    "template_id": [],
    "timeslide_id": [],
}
for channel in channels:
    coincident_triggers.update({
        f"time_{channel}": [],
        f"snr_{channel}": [],
    })

# Open the files
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

    this_template = {
        channel: triggers[channels[i]]['template_id'] == template_idx
        for i, channel in enumerate(channels)
    }

    trigger_times_0 = triggers[channels[0]]['end_time'][this_template[channels[0]]]
    trigger_times_1 = triggers[channels[1]]['end_time'][this_template[channels[1]]]
    trigger_snrs_0 = triggers[channels[0]]['snr'][this_template[channels[0]]]
    trigger_snrs_1 = triggers[channels[1]]['snr'][this_template[channels[1]]]

    # slide_id = 0 is zero-lag (on-source)
    # slide_id != 0 are the timeslides
    slide_ids = [0]
    if args.num_timeslides > 0:
        slide_ids.extend(
            list(range(-args.num_timeslides, 0)) + list(range(1, args.num_timeslides + 1))
        )

    for slide_id in slide_ids:
        time_shift = slide_id * args.coincidence_window
        shifted_trigger_times_0 = trigger_times_0 + time_shift

        for idx_one, time_0 in enumerate(shifted_trigger_times_0):
            left = np.searchsorted(trigger_times_1, time_0 - args.coincidence_window)
            right = np.searchsorted(trigger_times_1, time_0 + args.coincidence_window)

            if (right - left) == 1:
                # Coincident triggers!
                logger.debug(
                    "Slide %d: Trigger at time %.1f in channel %s is within coincidence "
                    "window of trigger in %s at time %.1f",
                    slide_id,
                    trigger_times_0[idx_one],
                    channels[0],
                    channels[1],
                    trigger_times_1[left],
                )
                # Add to the list of triggers
                coincident_triggers['template_id'].append(template_idx)
                coincident_triggers['timeslide_id'].append(slide_id)
                coincident_triggers[f'time_{channels[0]}'].append(trigger_times_0[idx_one])
                coincident_triggers[f'snr_{channels[0]}'].append(trigger_snrs_0[idx_one])
                coincident_triggers[f'time_{channels[1]}'].append(trigger_times_1[left])
                coincident_triggers[f'snr_{channels[1]}'].append(trigger_snrs_1[left])
            elif (right - left) > 1:
                # This case is not handled yet.
                # It means multiple triggers from channel 1 fall within the window of a single trigger from channel 0.
                pass

# Convert to numpy arrays
for key in coincident_triggers.keys():
    if 'template_id' in key or 'timeslide_id' in key:
        dtype = np.int32
    else:
        dtype = np.float32
    coincident_triggers[key] = np.array(coincident_triggers[key], dtype=dtype)
logger.info("Found %d coincident triggers", len(coincident_triggers['template_id']))

# Now lets do some clustering - cluster by quadratic sum of SNRs
# (aka network SNR), using the midpoint of the trigger times
logger.info("Clustering")

network_times = (
    coincident_triggers[f'time_{channels[0]}'] + coincident_triggers[f'time_{channels[1]}']
) / 2
network_snrs = (
    coincident_triggers[f'snr_{channels[0]}'] ** 2 + coincident_triggers[f'snr_{channels[1]}'] ** 2
) ** 0.5
network_tids = coincident_triggers['template_id'][:]


# Do clustering - nothing particularly fancy here,
# just check if there is anything louder within the window,
# if there is then just get rid of this coincidence


n_triggers = network_times.size
keep = np.ones(n_triggers, dtype=bool)
# Sort by SNR (descending), and then by time (ascending) as a tie-breaker
# to ensure deterministic clustering.
sorted_indices = np.lexsort((network_times, -network_snrs))

for i in sorted_indices:
    # if we've already removed this one, skip
    if not keep[i]:
        continue

    # find other triggers within the window
    # that have not been removed yet
    time = network_times[i]
    in_window_and_kept = (
        (np.abs(time - network_times) < args.coincidence_window) & keep
    )

    # The current trigger is the loudest in this window
    # because we are iterating from loudest to quietest.
    # So, remove all other triggers in the window.
    # First, set current trigger to False in the mask to not remove it
    in_window_and_kept[i] = False

    # Now remove all others that are True in the mask
    keep[in_window_and_kept] = False

clustered_coincident_triggers = {}
clustered_coincident_triggers['template_id'] = coincident_triggers['template_id'][keep]
clustered_coincident_triggers['timeslide_id'] = coincident_triggers['timeslide_id'][keep]
for channel in channels:
    clustered_coincident_triggers[f'time_{channel}'] = coincident_triggers[f'time_{channel}'][keep]
    clustered_coincident_triggers[f'snr_{channel}'] = coincident_triggers[f'snr_{channel}'][keep]

logger.info("%d out of %d coincidences remain after clustering", clustered_coincident_triggers['template_id'].size, n_triggers)

logger.info("Outputting to file")
with h5py.File(args.output_file, 'w') as output_trigger_file:
    for key in clustered_coincident_triggers.keys():
        output_trigger_file.create_dataset(
            key,
            data=clustered_coincident_triggers[key],
            compression='gzip'
        )

if args.testing_plots:

    # We only plot things with timeslide ID of zero (foreground)
    clustered_coincident_triggers = {
        key: clustered_coincident_triggers[key][clustered_coincident_triggers['timeslide_id'] == 0]
        for key in clustered_coincident_triggers.keys()
    }
    coincident_triggers = {
        key: coincident_triggers[key][coincident_triggers['timeslide_id'] == 0]
        for key in coincident_triggers.keys()
    }


    with h5py.File(args.expected_snr_file, 'r') as expected_snr_file:
        mbhb_times = expected_snr_file["mbhb_time"][:]
        expected_snrs = expected_snr_file["optimal_snr"][:] * expected_snr_file["fitting_factor"][:]

    cnorm = plt.Normalize(
        coincident_triggers['template_id'].min(),
        coincident_triggers['template_id'].max()
    )

    fig, ax = plt.subplots(figsize=(10,4))

    ax.scatter(
        mbhb_times / 86400,
        expected_snrs,
        marker='x',
        c='red',
        label="MBHB expected SNR for this bank",
        zorder=100
    )
    for mbhb_time in mbhb_times:
        ax.axvline(mbhb_time / 86400, c='tab:pink', linestyle=':')

    ax.scatter(
        coincident_triggers[f'time_{channels[0]}'] / 86400,
        coincident_triggers[f'snr_{channels[0]}'],
        marker='^',
        c=coincident_triggers['template_id'],
        norm=cnorm,
        cmap='jet',
        alpha=0.1,
    )
    ax.scatter(
        coincident_triggers[f'time_{channels[1]}'] / 86400,
        coincident_triggers[f'snr_{channels[1]}'],
        marker='v',
        c=coincident_triggers['template_id'],
        norm=cnorm,
        cmap='jet',
        alpha=0.1,
    )
    network_times = (
        coincident_triggers[f'time_{channels[0]}'] + coincident_triggers[f'time_{channels[1]}']
    ) / 2
    network_snrs = (
        coincident_triggers[f'snr_{channels[0]}'] ** 2 + coincident_triggers[f'snr_{channels[1]}'] ** 2
    ) ** 0.5
    
    im = ax.scatter(
        network_times / 86400,
        network_snrs,
        marker='o',
        c=coincident_triggers['template_id'],
        norm=cnorm,
        cmap='jet',
        alpha=0.1,
    )
    clustered_network_times = (
        clustered_coincident_triggers[f'time_{channels[0]}'] + clustered_coincident_triggers[f'time_{channels[1]}']
    ) / 2
    clustered_network_snrs = (
        clustered_coincident_triggers[f'snr_{channels[0]}'] ** 2 + clustered_coincident_triggers[f'snr_{channels[1]}'] ** 2
    ) ** 0.5
    
    ax.scatter(
        clustered_network_times / 86400,
        clustered_network_snrs,
        marker='*',
        edgecolors='k',
        c='y',
        zorder=1000,
        s=100
    )
    ax.scatter([],[], label='Network', c='k', marker='o')
    ax.scatter([],[], label=channels[0], c='k', marker='^')
    ax.scatter([],[], label=channels[1], c='k', marker='v')
    ax.scatter([],[], label="Clustered", c='y', marker='*', edgecolors='k')
    ax.legend(loc='upper right')

    cb = plt.colorbar(im, ax=ax)
    cb.set_label('Template ID')
    
    ax.set_xlabel('Time (days)')
    ax.set_ylabel('SNR')
    # ax.set_xlim(120, 145)
    fig.savefig('output/coincident_triggers.png')
    plt.close(fig)

logger.info("Done!")