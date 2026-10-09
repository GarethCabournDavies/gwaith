"A basic script to plot mchirp, eta, chieff as a corner plot for a bank file"

from matplotlib import pyplot as plt
from matplotlib import use

use('Agg')
import argparse
import logging

import h5py
import ldc.io.hdf5 as hdfio
import numpy as np

from hwilas import add_hwilas_common_options, init_logging
from hwilas.conversions import (
    chi_eff_from_component_spins,
    mchirp_eta_from_mass1_mass2,
)

logger = logging.getLogger(__name__)

def main():
    parser = argparse.ArgumentParser()
    add_hwilas_common_options(parser)
    parser.add_argument(
        "--bank-file",
        required=True,
        help="The template bank file to plot"
    )
    parser.add_argument(
        "--output-plot",
        required=True,
        help="The plot image file"
    )
    parser.add_argument(
        "--signal-definition-file",
        help="Signal Definition File, assumed to be a sangria dataset"
    )
    args = parser.parse_args()

    init_logging(args.verbose)

    logger.info('Loading bank file')
    with h5py.File(args.bank_file, "r") as f:
        m1 = f['mass1'][:]
        m2 = f['mass2'][:]
        s1z = f['spin1z'][:]
        s2z = f['spin2z'][:]

    logger.info('Converting parameters')
    mchirp, eta = mchirp_eta_from_mass1_mass2(m1, m2)
    chieff = chi_eff_from_component_spins(m1, m2, s1z, s2z)

    spin_range = (-1, 1)
    mchirp_range = (np.min(mchirp), np.max(mchirp))
    eta_range_size = 0.25 - np.min(eta)
    mass_ratio_range = (np.min(eta) - 0.1 * eta_range_size, 0.25)
    ranges = [mchirp_range, mass_ratio_range, spin_range]

    # Log-spaced histogram edges need at least two entries for small banks.
    n_bins = max(2, int(len(m1) / 10))

    logger.info('Plotting')
    fig, axes = plt.subplots(3,3, figsize=(10,10))

    #  Histograms along the diagonal, mchirp is in log space
    mchirp_bins = np.logspace(
        np.log10(
            mchirp_range[0]
        ),
        np.log10(
            mchirp_range[1]
        ),
        n_bins
    )
    axes[0,0].hist(
        mchirp,
        bins=mchirp_bins,
        density=True,
        color='green' 
    )
    axes[1,1].hist(
        eta,
        bins=n_bins,
        density=True,
        color='green' 
    )
    axes[2,2].hist(
        chieff,
        bins=n_bins,
        density=True,
        color='green' 
    )

    # scatter plots for the bottom-left triangle
    axes[1,0].scatter(
        mchirp,
        eta,
    )
    axes[2,0].scatter(
        mchirp,
        chieff,
    )
    axes[2,1].scatter(
        eta,
        chieff,
    )

    if args.signal_definition_file is not None:
        mbhbs, _ = hdfio.load_array(
            args.signal_definition_file,
            name="sky/mbhb/cat"
        )
        mbhb_mchirp, mbhb_eta = mchirp_eta_from_mass1_mass2(
            mbhbs['Mass1'],
            mbhbs['Mass2']
        )
        mbhb_chieff = chi_eff_from_component_spins(
            mbhbs['Mass1'],
            mbhbs['Mass2'],
            mbhbs['Spin1'],
            mbhbs['Spin2']
        )
        axes[1,0].scatter(
            mbhb_mchirp,
            mbhb_eta,
            marker='*',
            c='y'
        )
        axes[2,0].scatter(
            mbhb_mchirp,
            mbhb_chieff,
            marker='*',
            c='y'
        )
        axes[2,1].scatter(
            mbhb_eta,
            mbhb_chieff,
            marker='*',
            c='y'
        )

    # Log scales for mchirp
    axes[0,0].set_xscale('log')
    axes[1,0].set_xscale('log')
    axes[2,0].set_xscale('log')

    # Log scales for histograms
    axes[0,0].set_yscale('log')
    axes[1,1].set_yscale('log')
    axes[2,2].set_yscale('log')

    # Move eta and chieff hist y labels to right hand side
    axes[1,1].yaxis.tick_right()
    axes[2,2].yaxis.tick_right()


    # Hide axis tick labels for eta vs chieff subplot
    axes[2,1].set_yticklabels([])

    # Labels along the bottom x axis
    axes[2,0].set_xlabel('Chirp Mass $\\mathcal{M}$, $M_\\odot$')
    axes[2,1].set_xlabel('Symmetric Mass ratio $\\eta$')
    axes[2,2].set_xlabel('Dimensionless Spin $\\chi_\\text{eff}$')

    # Labels along the left y axis
    axes[0,0].set_ylabel('Chirp Mass $\\mathcal{M}$, $M_\\odot$')
    axes[1,0].set_ylabel('Symmetric Mass ratio $\\eta$')
    axes[2,0].set_ylabel('Dimensionless Spin $\\chi_\\text{eff}$')

    # set axis limits where appropriate
    for row in (0, 1, 2):
        for col in range(row):
            axes[row, col].set_ylim(*ranges[row])
        for col in range(row+1):
            # also apply x limits to histograms
            axes[row, col].set_xlim(*ranges[col])

    # Ticks point inwards and live on both sides for all plots:
    for ax in axes.flat:
        ax.tick_params(direction='in', which='both')
        ax.xaxis.set_ticks_position('both')
        ax.yaxis.set_ticks_position('both')

    # Hide unused axes
    axes[0,1].axis('off')
    axes[0,2].axis('off')
    axes[1,2].axis('off')

    fig.subplots_adjust(wspace=0, hspace=0)

    fig.suptitle(f"Template bank {args.bank_file}, size: {len(m1)}")

    fig.savefig(args.output_plot)

    logger.info("Done!")

if __name__ == "__main__":
    main()