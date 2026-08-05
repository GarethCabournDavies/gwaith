"""Calculate Optimal SNR and fitting factors from the bank for Sangria inputs"""

import argparse
import logging
import h5py
import numpy as np
import copy
from astropy import units as un

import ldc.io.hdf5 as hdfio
from ldc.lisa import orbits
from ldc.waveform.lisabeta import FastBHB

from gwaith.io import MultiDetOptionAction

from hwilas import add_hwilas_common_options, init_logging
from hwilas.data_processing.utils import AET, to_timeseries
from hwilas.filter import matched_filter_frequencyseries
from hwilas.psd.read import from_txt as psd_from_txt
from hwilas.waveforms import get_fd_waveform


logger = logging.getLogger(__name__)


lisa_orbits = orbits.Orbits.type({
    "nominal_arm_length":2.5e9*un.m,
    "initial_rotation":0*un.rad,
    "initial_position":0*un.rad,
    "orbit_type":"analytic"}
)


def load_ldc_timeseries(
    filename,
    data_group="obs/tdi",
    remove_noiseless_groups=None,
    delta_t=5.
):

    tdi_ts, _ = hdfio.load_array(filename, name=data_group)
    X = tdi_ts['X']
    Y = tdi_ts['Y']
    Z = tdi_ts['Z']

    if remove_noiseless_groups is not None:
        for ng in remove_noiseless_groups:
            tdi_to_rm, _ = hdfio.load_array(filename, name=ng)
            X -= tdi_to_rm['X']
            Y -= tdi_to_rm['Y']
            Z -= tdi_to_rm['Z']

    return AET(X,Y,Z, delta_t)

def fast_tdi(lisa_orbits, mbhb, start_time, end_time, dt):

    fast_hm = FastBHB(
        "MBHB",
        approx="IMRPhenomHM",
        T=end_time,
        delta_t=dt,
        orbits=lisa_orbits,
        modes=[(2,2), (2,1), (3,3), (3,2), (4,4), (4,3)]
    )

    A, E, T = fast_hm.get_td_tdiaet(
        template=mbhb,
        tdi2=False
    )

    # Cut to the start/end time as appropriate
    start_idx = int(start_time / dt)
    end_idx = int(end_time / dt)

    return to_timeseries(
        {
            'LISA_A': A[start_idx:end_idx],
            'LISA_E': E[start_idx:end_idx],
            'LISA_T': T[start_idx:end_idx],
        },
        dt,
        epoch=start_time
    )

def generate_waveform_for_data(
    mbhb,
    start_time,
    end_time,
    delta_t,
):
    wave = dict(zip(mbhb.dtype.names, mbhb)) 
    wave['Cadence'] = delta_t
    return fast_tdi(
        lisa_orbits,
        wave,
        start_time,
        end_time,
        delta_t
    )


def main():
    parser = argparse.ArgumentParser()
    add_hwilas_common_options(parser)
    parser.add_argument(
        "--bank-file",
        required=True,
        help="The template bank for calculating fitting factors"
    )
    parser.add_argument(
        "--data-file",
        required=True,
        help="Data file containing truth for the signals in Sangria dataset"
    )
    parser.add_argument(
        "--waveform-start",
        type=float,
        help="Time-before-merger to start the waveform, in seconds. Default=30 days",
        default=30 * 86400
    )
    parser.add_argument(
        "--waveform-end",
        type=float,
        help="Time-after-merger to finish the waveform, in seconds. Default=3 days",
        default=3 * 86400
    )
    parser.add_argument(
        "--psd-files",
        required=True,
        action=MultiDetOptionAction,
        help="Files containing PSD for optimal SNR and fitting factor calculations, given as channel:filename pairs"
    )
    parser.add_argument(
        "--output-file",
        required=True,
        help="File to store output optimal SNR and Fitting Factors"
    )
    parser.add_argument(
        "--tdi-generation",
        choices=["1.5", "2"],
        default="1.5",
        help="Which generation of TDI is used to calculate the waveforms"
    )

    args = parser.parse_args()

    init_logging(args.verbose)

    logger.info("Loading Bank %s", args.bank_file)
    with h5py.File(args.bank_file, 'r') as bank_file:
        num_templates = len(bank_file['mass1'])
        mass1 = bank_file['mass1'][:]
        mass2 = bank_file['mass2'][:]
        s1 = bank_file['spin1z'][:]
        s2 = bank_file['spin2z'][:]
        eclipticlongitude = bank_file['eclipticlongitude'][:]
        eclipticlatitude = bank_file['eclipticlatitude'][:]
        inclination = bank_file['inclination'][:]
        polarization = bank_file['polarization'][:]


    logger.info("Loading MBHB data from %s", args.data_file)
    mbhbs, _ = hdfio.load_array(
        args.data_file,
        name="sky/mbhb/cat"
    )

    tlen = args.waveform_end + args.waveform_start
    delta_t = 5
    flen = tlen // 2 + 1
    delta_f = 1. / tlen

    logger.info("Loading PSD")
    psds = {
        channel: psd_from_txt(
            args.psd_files[channel],
            flen,
            delta_f,
            delta_f,
            is_asd_file=False,
        )
        for channel in ['LISA_A', 'LISA_E']
    }

    logger.info("Calculating Optimal SNRs")
    mbhb_data = {}
    optimal_snrs = [None] * len(mbhbs)
    for i, mbhb in enumerate(mbhbs):
        logger.debug("Processing MBHB %d", i)
        data_start_time = mbhb['CoalescenceTime'] - args.waveform_start
        data_end_time = mbhb['CoalescenceTime'] + args.waveform_end

        logger.debug("Generating Waveforms")
        waveform_for_removal = generate_waveform_for_data(
            mbhb,
            data_start_time,
            data_end_time,
            delta_t,
        )
        mbhb_data[i] = waveform_for_removal

        logger.debug("Calculating Optimal SNR")
        A_snr = matched_filter_frequencyseries(
            waveform_for_removal['LISA_A'].to_frequencyseries(),
            waveform_for_removal['LISA_A'].to_frequencyseries(),
            psd=psds['LISA_A'],
            n_samples=len(waveform_for_removal['LISA_A']),
            epoch=float(waveform_for_removal['LISA_A'].epoch),
        )
        E_snr = matched_filter_frequencyseries(
            waveform_for_removal['LISA_E'].to_frequencyseries(),
            waveform_for_removal['LISA_E'].to_frequencyseries(),
            psd=psds['LISA_E'],
            n_samples=len(waveform_for_removal['LISA_E']),
            epoch=float(waveform_for_removal['LISA_E'].epoch),
        )

        snr = abs(A_snr.data ** 2 + E_snr.data ** 2) ** 0.5

        logger.info("Optimal SNR for signal %d: %.3f", i, np.max(snr))
        optimal_snrs[i] = np.max(snr)

    waveform_params_shared = {
        'f_lower': 1e-6,
        'low-frequency-cutoff': 1e-6, 
        'tdi': args.tdi_generation,
        't_offset': 0,
        'cutoff_deltat': 0,
        't_obs_start': int(args.waveform_start + args.waveform_end),
        'delta_f': delta_f,
        'f_final': 0.2 / 2 + delta_f / 2, # This is a hack to ensure we get the point at f_final returned
        'approximant':'BBHX_PhenomD',
        'mode_array':  [(2,2)],
        'length': flen,
        'tc': tlen
    }

    logger.info("Calculating Fitting Factors")
    fitting_factors = [0] * len(mbhbs)
    best_wf = [None] * len(mbhbs)
    for idx in range(num_templates):
        if idx % 10:
            logger.debug("Processing template %d", idx)
        else:
            logger.info("Processing template %d", idx)

        wf_params = copy.deepcopy(waveform_params_shared)
        wf_params.update(
            {
                "mass1": mass1[idx],
                "mass2": mass2[idx],
                "spin1z": s1[idx],
                "spin2z": s2[idx],
                "eclipticlongitude": eclipticlongitude[idx],
                "eclipticlatitude": eclipticlatitude[idx],
                "inclination": inclination[idx],
                "polarization": polarization[idx],
            }
        )
        
        wf = get_fd_waveform(
            approximant=wf_params['approximant'],
            **wf_params,
            ifos=['LISA_A', 'LISA_E'],
        )

        for i, mbhb in enumerate(mbhbs):
            A_snr = matched_filter_frequencyseries(
                wf['LISA_A'],
                mbhb_data[i]['LISA_A'].to_frequencyseries(),
                psd=psds['LISA_A'],
                n_samples=len(mbhb_data[i]['LISA_A']),
                epoch=float(mbhb_data[i]['LISA_A'].epoch),
            )
            E_snr = matched_filter_frequencyseries(
                wf['LISA_E'],
                mbhb_data[i]['LISA_E'].to_frequencyseries(),
                psd=psds['LISA_E'],
                n_samples=len(mbhb_data[i]['LISA_E']),
                epoch=float(mbhb_data[i]['LISA_E'].epoch),
            )

            snr = abs(A_snr.data ** 2 + E_snr.data ** 2) ** 0.5
            ff = np.max(snr) / optimal_snrs[i]
            if ff > 1:
                # This shouldnt happen - warn and carry on
                logger.warning("Match greater than one for waveform %d and MBHB %d", idx, i)
                ff = 1
            if ff > fitting_factors[i]:
                fitting_factors[i] = ff
                best_wf[i] = idx

    logger.info("Fitting factors:")
    for i in range(len(fitting_factors)):
        logger.info("MBHB %d: %.3f (best waveform %d)", i, fitting_factors[i], best_wf[i])

    for i in range(len(fitting_factors)):
        logger.info("MBHB %d expected SNR using this bank: %.3f", i, optimal_snrs[i] * fitting_factors[i])


    logger.info("Writing output")
    with h5py.File(args.output_file, "w") as output_file:
        output_file.create_dataset("mbhb_time", data=mbhbs['CoalescenceTime'])
        output_file.create_dataset("optimal_snr", data=optimal_snrs)
        output_file.create_dataset("fitting_factor", data=fitting_factors)
        output_file.create_dataset("best_waveform", data=best_wf)

    logger.info("Done!")

if __name__ == '__main__':
    main()