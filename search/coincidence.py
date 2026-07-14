import h5py

import argparse
import logging
import numpy as np
from matplotlib import use
use('Agg')
from matplotlib import pyplot as plt

import pycbc
from scipy.signal import find_peaks
from pycbc.types import MultiDetOptionAction  # Custom action for argparse

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
    default=7200,
    help="Window to use when assessing coincidence, in seconds. Default 100"
)
parser.add_argument(
    "--output-file",
    required=True,
    help="File to store output peaks in SNR"
)
parser.add_argument(
    "--testing-plots",
    help="Directory into which to store plots used for testing"
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

channels = sorted(list(channels))

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
}
for channel in channels:
    coincident_triggers.update({
        f"time_{channel}": [],
        f"snr_{channel}": [],
        f"trigger_id_{channel}": [],
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

    for idx_one, time_0 in enumerate(trigger_times_0):
        left = np.searchsorted(trigger_times_1, time_0 - args.coincidence_window)
        right = np.searchsorted(trigger_times_1, time_0 + args.coincidence_window)

        if (right - left) == 1:
            # Coincident triggers!
            logger.debug(
                "Trigger at time %.1f in channel %s is within coincidence "
                "window of trigger in %s at time %.1f",
                trigger_times_0[idx_one],
                channels[0],
                channels[1],
                trigger_times_1[left],
            )
            # Add to the list of triggers
            coincident_triggers['template_id'].append(template_idx)
            coincident_triggers[f'trigger_id_{channels[0]}'].append(idx_one)
            coincident_triggers[f'trigger_id_{channels[1]}'].append(left)
            # SNR and time could be got from indexing on the trigger merge files, but add these in so that
            # everything is in one place - this may need updated later
            coincident_triggers[f'time_{channels[0]}'].append(trigger_times_0[idx_one])
            coincident_triggers[f'snr_{channels[0]}'].append(trigger_snrs_0[idx_one])
            coincident_triggers[f'time_{channels[1]}'].append(trigger_times_1[left])
            coincident_triggers[f'snr_{channels[1]}'].append(trigger_snrs_1[left])
        elif (right - left) > 1:
            # This case is not handled yet.
            # It means multiple triggers from channel 1 fall within the window of a single trigger from channel 0.
            raise RuntimeError(
                "Multiple triggers found within the coincidence window"
            )

# Convert to numpy arrays
for key in coincident_triggers.keys():
    if any(key.startswith(s) for s in ['template_id', 'timeslide_id', 'trigger_id']):
        dtype = np.int32
    else:
        dtype = np.float32
    coincident_triggers[key] = np.array(coincident_triggers[key], dtype=dtype)
logger.info("Found %d coincident triggers", len(coincident_triggers['template_id']))

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
coinc_tids = coincident_triggers['template_id'][:]

n_coincs = len(coinc_time)

# Now, perform a standard greedy clustering on the coincs
# First, sort by statistic - loudest first
indices = np.argsort(coinc_stat)[::-1]

keep = np.ones(n_coincs, dtype=bool)
for sorted_coinc_idx in range(n_coincs):
    coinc_idx = indices[sorted_coinc_idx]
    if not keep[coinc_idx]:
        continue
    within_window = np.abs(coinc_time - coinc_time[coinc_idx]) < args.coincidence_window
    keep[within_window] = False
    keep[coinc_idx] = True

clustered_coincident_triggers = {}
for key in coincident_triggers.keys():
    clustered_coincident_triggers[key] = coincident_triggers[key][keep]

logger.info("%d out of %d coincidences remain after clustering", clustered_coincident_triggers['template_id'].size, n_coincs)

logger.info("Outputting to file")
with h5py.File(args.output_file, 'w') as output_trigger_file:
    for key in clustered_coincident_triggers.keys():
        output_trigger_file.create_dataset(
            key,
            data=clustered_coincident_triggers[key],
            compression='gzip'
        )

# Print the 25 highest-statistic events

if args.testing_plots is not None:
    logger.info("Plotting")

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
    coinc_times_all = (
        coincident_triggers[f'time_{channels[0]}'] + coincident_triggers[f'time_{channels[1]}']
    ) / 2
    coinc_stat_all = (
        coincident_triggers[f'snr_{channels[0]}'] ** 2 + coincident_triggers[f'snr_{channels[1]}'] ** 2
    ) ** 0.5
    
    im = ax.scatter(
        coinc_times_all / 86400,
        coinc_stat_all,
        marker='o',
        c=coincident_triggers['template_id'],
        norm=cnorm,
        cmap='jet',
        alpha=0.1,
    )
    coinc_time_clustered = (
        clustered_coincident_triggers[f'time_{channels[0]}'] + clustered_coincident_triggers[f'time_{channels[1]}']
    ) / 2
    coinc_stat_clustered = (
        clustered_coincident_triggers[f'snr_{channels[0]}'] ** 2 + clustered_coincident_triggers[f'snr_{channels[1]}'] ** 2
    ) ** 0.5
    
    ax.scatter(
        coinc_time_clustered / 86400,
        coinc_stat_clustered,
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
    # ax.set_xlim(230, 240)
    # ax.set_ylim(0, 200)
    fig.savefig(f'{args.testing_plots}/coincident_triggers.png')
    plt.close(fig)

    fig, ax = plt.subplots()
    bins = np.logspace(
        np.log10(coinc_stat_clustered.min()),
        np.log10(coinc_stat_clustered.max()),
        100
    )
    hist_snr, bins = np.histogram(coinc_stat_clustered, bins=bins)
    cumhist = np.cumsum(hist_snr[::-1])[::-1]
    ax.step(
        bins[:-1],
        cumhist,
        label="Clustered Coincs Statistic"
    )
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel('SNR')
    ax.set_ylabel('Count of louder events (per year)')

    sort_expected = np.argsort(expected_snrs)
    ax.step(
        expected_snrs[sort_expected],
        np.arange(len(expected_snrs))[::-1] + 1,
        label="Expected statistic for Signals"
    )
    ax.legend(loc='upper right')

    fig.savefig(f'{args.testing_plots}/snr_histogram.png')
    plt.close(fig)



logger.info("Done!")