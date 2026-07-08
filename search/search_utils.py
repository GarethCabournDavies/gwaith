import numpy as np
import copy
import h5py
import logging
import os
from functools import cache

from matplotlib import pyplot as plt

import ldc.io.hdf5 as hdfio

from scipy import signal
from scipy.signal import windows

import pycbc.psd
import pycbc.types
import pycbc.waveform
import pycbc.filter
import pycbc.pnutils
from pycbc.strain.gate import gate_and_paint
from pycbc.types import TimeSeries


def to_timeseries(waveform_dict, delta_t, epoch=0):
    return_dict = {}
    for channel, array in waveform_dict.items():
        array_ts = TimeSeries(array, delta_t=delta_t, epoch=epoch)
        return_dict[channel] = array_ts
    return return_dict

def AET(X,Y,Z, delta_t, epoch=0):
    A = (Z - X)/np.sqrt(2.0)
    E = (X - 2.0*Y + Z)/np.sqrt(6.0)
    T = (X + Y + Z)/np.sqrt(3.0)

    AET_ts = to_timeseries(
        {
            'LISA_A': A,
            'LISA_E': E,
            'LISA_T': T,
        },
        delta_t,
        epoch=epoch
    )

    return AET_ts

def load_sangria_timeseries(
    filename,
    data_group="obs/tdi",
):
    tdi_ts, _ = hdfio.load_array(filename, name=data_group)
    X = tdi_ts['X']
    Y = tdi_ts['Y']
    Z = tdi_ts['Z']

    attrs = hdfio.load_attributes(filename, name=data_group)
    delta_t = attrs['dt']
    epoch = attrs['t0']

    return AET(X,Y,Z, delta_t, epoch=epoch)

####################################################
# Function to get SNR given data and wform params
####################################################
def get_snr_series(
        params,
        data_ow_f,
        psds_for_whitening,
        delta_t=5,
        hh=None,
        original_length=None,
        waveforms=None,
    ):

    if waveforms is None:
        waveforms = generate_waveform_lisa_pre_merger_inpaint(
            params,
            psds_for_whitening,
            sample_rate=1./delta_t,
        )

    snr = {}
    for channel in waveforms.keys():
        snr[channel] = pycbc.filter.matched_filter(
            waveforms[channel],
            data_ow_f[channel],
            sigmasq=1
        )

    if hh is not None:
        for channel, hh_array in hh.items():
            hh_array._epoch = data_ow_f[channel]._epoch
            snr[channel] /= (hh_array ** 0.5 * (2 ** 0.5))
    else:
        raise RuntimeError('Situation not yet supported')

    if original_length is not None:
        for channel in snr.keys():
            # Trim to the original length
            snr[channel] = snr[channel][:original_length]

    return snr


def pick_best_snr_pair(
    snr_A, 
    snr_E,
    start_idx=0,
):
    """
    Pick the best pair of SNR samples from snr_A and snr_E

    Returns:
      snr: tuple (snr_A_value, snr_E_value)
      indices: tuple (index_A_abs, index_E_abs) absolute indices in the SNR series (provided start_idx is given)
      times: tuple (A_time, E_time) times of the peaks
    """

    # Get max of A series 
    argmax_A = np.argmax(
        abs(snr_A.data)
    )
    max_idx = len(snr_A.data)

    # Find the maximum value of E series over a small interval around this
    mineval = max(argmax_A - 20, 0)
    maxeval = min(argmax_A + 20, max_idx)

    argmax_E = np.argmax(
        abs(snr_E.data[mineval:maxeval])
    )
    argmax_E = argmax_E + mineval
    snr = (
        abs(snr_A.data[argmax_A]),
        abs(snr_E.data[argmax_E])
    )
    snr_sq = abs(snr[0]**2 + snr[1]**2)

    # Get max of E series
    amax2_E = np.argmax(
        abs(snr_E.data)
    )
    # Search a small window around the maximum of E to see if that is better
    minaval = max(amax2_E - 20, 0)
    maxaval = min(amax2_E + 20, max_idx)
    amax2_A = np.argmax(
        abs(snr_A.data[minaval:maxaval])
    )
    amax2_A = amax2_A + minaval
    snr2 = (
        abs(snr_A.data[amax2_A]),
        abs(snr_E.data[amax2_E])
    )
    snr2_sq = abs(snr2[0]**2 + snr2[1]**2)
    if snr2_sq > snr_sq:
        snr = snr2
        argmax_A = amax2_A
        argmax_E = amax2_E

    A_time = argmax_A*snr_A._delta_t + float(snr_A._epoch)
    E_time = argmax_E*snr_E._delta_t + float(snr_E._epoch)

    return snr, (start_idx + argmax_A, start_idx + argmax_E), (A_time, E_time)


def get_snr_from_series(
        params,
        data_f,
        psds_for_whitening,
        search_time,
        delta_t=5,
        cutoff_time=0,
        time_samples=518400,
        zeroed_length=2**20,
        gaps=None,
    ):

    cutoff_idx = int(time_samples - cutoff_time / delta_t)

    hh = compute_hh_inner_product(
        params,
        psds_for_whitening,
        sample_rate=1./delta_t,
        inpaint_start=cutoff_idx,
        zeroed_length=zeroed_length,
        gaps=gaps,
        epoch=data_f['LISA_A'].epoch
    )

    snrs = get_snr_series(
        params,
        data_f,
        psds_for_whitening,
        delta_t=delta_t,
        hh=hh,
        original_length=time_samples,
    )

    snr_A = abs(snrs['LISA_A'])
    snr_E = abs(snrs['LISA_E'])

    if search_time is None:
        search_indices = len(snr_A)
    else:
        search_indices = int(search_time//delta_t)
    start_idx = len(snr_A) - search_indices
    search_slice = slice(start_idx, len(snr_A))

    # Use helper to pick the best SNR pair within the search slice
    snr, indices, times = pick_best_snr_pair(
        snr_A[search_slice],
        snr_E[search_slice],
        start_idx=start_idx,
    )
    return snr, indices, times, {'LISA_A':snr_A, 'LISA_E':snr_E}


def get_snr_future_series(
        params,
        data_f,
        psds,
        delta_t=5,
        original_length=518400,
        forward_days=1.0,
        time_points_days=None,
        window_seconds=0.0,
        zeroed_length=2**20,
        gaps=None,
        plot=False,
        plot_dir='.'
    ):
    """
    Compute SNR series starting at `start_index` and extending forward by `forward_days`.
    """

    # Calculate waveforms once here
    waveforms = generate_waveform_lisa_pre_merger_inpaint(
        params,
        psds,
        sample_rate=1. / delta_t,
    )

    logging.debug('Calculating hh inner product')
    hh = compute_hh_inner_product(
        params,
        psds,
        sample_rate=1. / delta_t,
        inpaint_start=original_length,
        zeroed_length=zeroed_length,
        gaps=gaps,
        epoch=data_f['LISA_A'].epoch,
        waveforms=waveforms
    )
    if plot:
        logging.debug('Plotting hh inner product')
        fig, ax = plt.subplots()
        ax.semilogy(hh['LISA_A'].sample_times, hh['LISA_A'], linestyle='-', label='A')
        ax.semilogy(hh['LISA_E'].sample_times, hh['LISA_E'], linestyle='-', label='E')
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('$(h|h)$ inner product')
        ax.legend(loc='upper right')
        for tp in time_points_days:
            ax.axvline(
                hh['LISA_A'].sample_times[original_length] + 86400 * tp,
                c='r', linestyle=':'
            )
        fig.savefig(os.path.join(plot_dir, 'hh.png'))

    logging.debug('Calculating SNR series')
    snrs = get_snr_series(
        params,
        data_f,
        psds,
        delta_t=delta_t,
        hh=hh,
        original_length=None, # Dont do the trimming in the get_snr_series function, basically for sanity plots
        waveforms=waveforms,
    )

    snr_A = snrs['LISA_A']
    snr_E = snrs['LISA_E']
    if plot:
        logging.info(f'unsliced abs(SNR A) **2 series, mean:{abs(snr_A ** 2).data.mean() } std:{abs(snr_A ** 2).data.std()}')
        logging.info(f'unsliced abs(SNR E) **2 series, mean:{abs(snr_E ** 2).data.mean() } std:{abs(snr_E ** 2).data.std()}')

    # Start at the original data length (time_samples) and go forward
    secs_per_day = 86400.0
    forward_samples = int(forward_days * secs_per_day / delta_t)

    # if weve requested more time than there is, complain:
    end_idx = original_length + forward_samples
    if end_idx > len(snr_A):
        raise RuntimeError(
            f'Requesting more datapoints than available. {end_idx} vs {len(snr_A)}'
        )

    if plot:
        logging.debug('Plotting SNR series')
        
        # snrs_norm = get_snr_series(
        #     params,
        #     data_f,
        #     psds,
        #     delta_t=delta_t,
        #     hh=None,
        #     original_length=None, # Dont do the trimming in the get_snr_series function, basically for sanity plots
        # )
        slc = slice(int(86400 / delta_t), len(snr_A) - int(86400 / delta_t))
        fig, ax = plt.subplots()
        ax.plot(
            snr_A.sample_times[slc],
            abs(snr_A)[slc],
            c='tab:blue',
            label='A'
        )
        ax.plot(
            snr_E.sample_times[slc],
            abs(snr_E)[slc],
            c='tab:orange',
            label='E',
            linestyle='--'
        )
        ax.set_ylabel('SNR')
        ax.legend(loc='upper right')
        ax.set_xlabel('Time (s)')
        # ax.plot(snrs_norm['LISA_A'].sample_times, abs(snrs_norm['LISA_A']), c='k')
        ax.axvline(
            snr_A.sample_times[original_length],
            color='k',
            linestyle='--'
        )
        ax.axvline(
            snr_A.sample_times[end_idx],
            color='k',
            linestyle='--'
        )
        # ax.plot(hh['LISA_A'].sample_times, hh['LISA_A'])
        # ax.set_xlim(
        #     float(snr_A._epoch) - 3600,
        #     snr_A.sample_times[end_idx] + 3600
        # )
        ax.set_yscale('log')
        ax.set_ylim(bottom=0.1)
        for tp in time_points_days:
            ax.axvline(
                snr_A.sample_times[original_length] + secs_per_day * tp,
                c='r', linestyle=':'
            )
        fig.savefig(os.path.join(plot_dir, 'snr_series_inpainting_full.png'))


    logging.debug('Slicing the series for each channel')
    snr_slice_A = snr_A[original_length:end_idx]
    snr_slice_E = snr_E[original_length:end_idx]

    if plot:
        logging.info(f'abs(SNR A) **2 series, mean:{abs(snr_slice_A ** 2).data.mean() } std:{abs(snr_slice_A ** 2).data.std()}')
        logging.info(f'abs(SNR E) **2 series, mean:{abs(snr_slice_E ** 2).data.mean() } std:{abs(snr_slice_E ** 2).data.std()}')

    result = {
        'snr_slice': {'LISA_A': snr_slice_A, 'LISA_E': snr_slice_E},
        'windows': []
    }

    # If specific time points are requested, extract best SNR from windows around them
    window_samples = int(window_seconds / delta_t)

    for tp in time_points_days:
        logging.debug(f'Processing time point {tp}')
        # tp is in days from original data end
        end_window_index = int(tp * secs_per_day / delta_t)

        # Clamp to available range
        end_window = min(end_window_index, len(snr_slice_A))
        start_window = max(end_window_index - window_samples, 0)
        window_slice = slice(start_window, end_window)

        snrs_window, window_indices, window_times = pick_best_snr_pair(
            abs(snr_slice_A[window_slice]),
            abs(snr_slice_E[window_slice]),
            start_idx=start_window,
        )

        result['windows'].append({
            'time_point_days': tp,
            'centre_index': end_window_index,
            'snr_A': snrs_window[0],
            'snr_E': snrs_window[1],
            'times': window_times,
            'indices': window_indices,
            'data': (abs(snr_slice_A[window_slice]), abs(snr_slice_E[window_slice]))
        })

    return result

def get_snr_point(
        params,
        data,
        psds_for_whitening,
        delta_t=5,
    ):

    snr_A, snr_E = get_snr_series(
        params,
        data,
        psds_for_whitening,
        delta_t=delta_t
    )

    return abs(snr_A[0]), abs(snr_E[0])

################################
# Function to perform filtering
#################################
def filter_some_waveforms(
        waveform_params,
        psds_for_datagen,
        psds_for_whitening,
        time_before,
        tmpltbank,
        nosignal=False,
        random_seed=137,
        delta_t=5,
        search_time=3600,
        reduce_bank_factor=None,
        label='Label',
        plot_best_wf=False,
    ):
    generation_waveform = copy.deepcopy(waveform_params)
    generation_waveform.update({
        'approximant': 'BBHX_PhenomHM'
    })
    filter_waveform = copy.deepcopy(waveform_params)
    filter_waveform.update({
        'approximant': 'BBHX_PhenomD',
        'mode_array':[(2,2)],
    })

    print(f"Time before {time_before}")

    data = generate_data_lisa_pre_merger_inpaint(
        generation_waveform,
        psds_for_datagen,
        sample_rate=1. / delta_t,
        seed=random_seed,
        no_signal=nosignal,
    )
    cutoff_idx = len(data['LISA_A']) - int(time_before / delta_t)
 
    data = pre_process_data_lisa_pre_merger_inpaint(
        data,
        sample_rate=1. / delta_t,
        psds_for_whitening=psds_for_whitening,
        inpaint_start=cutoff_idx,
        inpaint_end=len(data['LISA_A'])
    )

    data_A_f = data['LISA_A'].to_frequencyseries()
    data_E_f = data['LISA_E'].to_frequencyseries()

    if not nosignal:
        data_nn = generate_data_lisa_pre_merger_inpaint(
            generation_waveform,
            psds_for_datagen,
            sample_rate=1./delta_t,
            no_signal=nosignal,
            zero_noise=True
        )
        cutoff_idx = len(data['LISA_A']) - int(time_before / delta_t)

        data_nn = pre_process_data_lisa_pre_merger_inpaint(
            data_nn,
            sample_rate=1./delta_t,
            psds_for_whitening=psds_for_whitening,
            inpaint_start=cutoff_idx,
            inpaint_end=len(data['LISA_A'])
        )
        data_A_f_nn = data_nn['LISA_A'].to_frequencyseries()
        data_E_f_nn = data_nn['LISA_E'].to_frequencyseries()

        snr, _, _, _ = get_snr_from_series(
            filter_waveform,
            {'LISA_A': data_A_f_nn, 'LISA_E': data_E_f_nn},
            psds_for_whitening,
            search_time,
            delta_t=delta_t,
        )
        print(
            f"With no-higher-modes template, optimal (noiseless) SNR is {snr[0]}, {snr[1]}, "
            f"{(snr[0]**2 + snr[1]**2)**0.5}"
        )

        snr, _, _, _ = get_snr_from_series(
            filter_waveform,
            {'LISA_A': data_A_f, 'LISA_E': data_E_f},
            psds_for_whitening,
            search_time,
            delta_t=delta_t,
        )
        print(
            f"With no-higher-modes template, MF SNR is {snr[0]}, {snr[1]}, "
            f"{(snr[0]**2 + snr[1]**2)**0.5}"
        )

    # Reverse engineer the time-length of the data
    data_length = (len(data_A_f) - 1) * 2 * delta_t
    print(f"Beginning filtering with bank {tmpltbank}")
    max_snr = 0
    snr_vals = "Problem - no SNRs found > 0"
    with h5py.File(tmpltbank, 'r') as bank_file:
        for idx in range(len(bank_file['mass1'])):
            if reduce_bank_factor is not None and idx % reduce_bank_factor:
                # For testing: reduce the bank size by this factor to make the search quicker
                continue
            params = copy.deepcopy(filter_waveform)
            params['tc'] = data_length
            params['mass1'] = bank_file['mass1'][idx]
            params['mass2'] = bank_file['mass2'][idx]
            params['inclination'] = bank_file['inclination'][idx]
            params['polarization'] = bank_file['polarization'][idx]
            params['spin1z'] = bank_file['spin1z'][idx]
            params['spin2z'] = bank_file['spin2z'][idx]
            #params['coa_phase'] = bank_file['coa_phase'][idx]
            params['eclipticlatitude'] = bank_file['eclipticlatitude'][idx]
            params['eclipticlongitude'] = bank_file['eclipticlongitude'][idx]

            snr, iidx, times, _ = get_snr_from_series(
                params,
                {'LISA_A': data_A_f, 'LISA_E': data_E_f},
                psds_for_whitening,
                search_time,
                delta_t=delta_t,
            )
            snr_qs = snr[0]**2 + snr[1]**2
            if snr_qs > max_snr:
                max_snr = snr_qs
                snr_vals = [idx, snr, max_snr**0.5, iidx, times, copy.deepcopy(params)]


    if plot_best_wf:
        plot_best_waveform(
            snr_vals,
            {'LISA_A': data_A_f, 'LISA_E': data_E_f},
            psds_for_whitening,
            time_before,
            window_length,
            search_time,
            kernel_length,
            delta_t=delta_t,
            label=f'{label}_{random_seed}'
        )
    return snr_vals

def get_optimal_snr(
        waveform_params,
        psds_for_whitening,
        cutoff_time,
        window_length=17280,
        delta_t=5,
        kernel_length=17280,
    ):

    waveforms = generate_waveform_lisa_pre_merger_inpaint(
        waveform_params,
        psds_for_whitening,
        sample_rate=1. / delta_t,
    )

    snr = get_snr_point(
        waveform_params,
        waveforms,
        psds_for_whitening,
    )

    return snr

def get_optimal_snr_freq_cut(
        waveform_params,
        psds_for_datagen,
        cutoff_time,
        delta_t=5.,
        f_lower=1e-7,
        delta_f=1e-7,
    ):
    """
    A function to get the optimal SNR using a frequency
    cut rather than the FIR filter and cutting method.
    This is approximate and a useful sanity check
    """
    if cutoff_time == 0:
        end_freq = 1 / (2 * delta_t)
    else:
        # Approximate a time cut by working out
        # the time-frequency track and interpolating
        # the frequency cut based off that
        track_t, track_f = pycbc.pnutils.get_inspiral_tf(
            0,
            waveform_params['mass1'],
            waveform_params['mass2'],
            0,
            0,
            f_lower,
            approximant='SPAtmplt'
        )
        end_freq = np.interp(
            -cutoff_time,
            track_t,
            track_f,
        )
        if end_freq < (f_lower + 2 * delta_f):
            # This frequency / time before merger is too low
            # frequency, and won't give sensible results
            return np.nan

    wf = pycbc.waveform.get_fd_det_waveform(
        **waveform_params,
        ifos=['LISA_A','LISA_E'],
        f_final=end_freq,
    )

    sig = {}
    for channel in ['A','E']:
        chan = f'LISA_{channel}'
        cut_psd = psds_for_datagen[chan][:len(wf[chan])]
        sig[channel] = pycbc.filter.sigma(
            wf[chan],
            cut_psd,
            low_frequency_cutoff=f_lower,
            high_frequency_cutoff=end_freq
        )

    return sig['A'], sig['E']

def plot_best_waveform(
    snr_vals,
    data_f,
    psds_for_whitening,
    time_before,
    search_time,
    delta_t=5,
    label='Label'
):

    snr = get_optimal_snr(
        snr_vals[5],
        psds_for_whitening,
        cutoff_time=time_before,
        delta_t=delta_t,
    )

    data_length = (len(data_f['LISA_A']) - 1) * 2 * delta_t

    print(
        f"With found template, optimal SNR is {snr[0]}, {snr[1]}, "
        f"{(snr[0]**2 + snr[1]**2)**0.5}"
    )

    waveforms = generate_waveform_lisa_pre_merger_inpaint(
        snr_vals[5],
        psds_for_whitening,
        sample_rate=1. / delta_t,
    )

    fig, ax = plt.subplots(1)
    for channel in ['LISA_A', 'LISA_E']:
        ax.plot(
            data_f[channel].sample_frequencies,
            data_f[channel],
            alpha=0.5,
            label=f'{channel} data'
        )
        ax.plot(
            waveforms[channel].sample_frequencies,
            abs(waveforms[channel]),
            alpha=0.5,
            label=f'{channel} waveform'
        )
    if label == 'cutoff':
        ax.axvspan(
            0, 1e-4,
            color='red',
            alpha=0.25,
            zorder=-100
        )
    ax.loglog()
    ax.grid()
    ax.set_xlim(1e-6, 1e-1)

    ax.legend()
    ax.set_xlabel('Frequency, Hz')
    ax.set_title(f'Template {snr_vals[0]}, {time_before / 86400} days before merger, {label} psd')
    fig.savefig(f'bestwf_freq_{label}.png')

    wv_time = {k: v.to_timeseries() for k, v in waveforms.items()}

    fig, (ax0, ax1, ax2, ax3) = plt.subplots(
        1, 4, figsize=(16,4)
    )

    for ax in [ax0, ax1, ax2, ax3]:
        for channel in ['LISA_A', 'LISA_E']:
            ax.plot(
                wv_time[channel].sample_times,
                wv_time[channel].real(),
                alpha=0.5,
                label=channel
            )
        ax.grid()
        ax.set_yscale('symlog', linthresh=1e-5)


    for ax in [ax0, ax1, ax2, ax3]:
        ax.axvspan(
            data_length - time_before,
            data_length,
            color='r',
            zorder=-100,
            alpha=0.25
        )

    ax0.set_xlabel('Zeroed start')
    ax1.set_xlabel('Tapered Waveform start')
    ax2.set_xlabel('Full Waveform')
    ax3.set_xlabel('Zeroed End')
    fig.suptitle(f'Template {snr_vals[0]}, {time_before / 86400} days before merger, {label} psd')

    fig.savefig(f'bestwf_time_{label}.png')


    fig, ax = plt.subplots(1)

    series_out = get_snr_series(
        snr_vals[5],
        data_f,
        psds_for_whitening,
        delta_t=delta_t
    )

    ax.plot(
        series_out[1].sample_times - series_out[1].sample_times[-1],
        (series_out[0] ** 2 + series_out[1] ** 2) ** 0.5,
        c='k',
        label='Sum squared'
    )
    ax.plot(
        series_out[1].sample_times - series_out[1].sample_times[-1],
        series_out[0],
        c='r',
        label='LISA_A'
    )
    ax.plot(
        series_out[1].sample_times - series_out[1].sample_times[-1],
        series_out[1],
        c='b',
        label='LISA_E'
    )
    ax.plot(
        series_out[1].sample_times - series_out[1].sample_times[0],
        (series_out[0] ** 2 + series_out[1] ** 2) ** 0.5,
        c='k',
        linestyle=':',
    )
    ax.plot(
        series_out[1].sample_times - series_out[1].sample_times[0],
        series_out[0],
        c='r',
        linestyle=':',
    )
    ax.plot(
        series_out[1].sample_times - series_out[1].sample_times[0],
        series_out[1],
        c='b',
        linestyle=':',
    )

    snr_best, _, times, _ = get_snr_from_series(
        snr_vals[5],
        data_f,
        psds_for_whitening,
        delta_t=delta_t,
        search_time=search_time
    )

    ax.scatter(
        times[0] - series_out[1].sample_times[-1],
        snr_best[0],
        marker='x',
        color='r',
        zorder=100
    )
    ax.scatter(
        times[0] - series_out[1].sample_times[-1],
        snr_best[1],
        marker='x',
        color='b',
        zorder=100
    )
    ax.scatter(
        sum(times) / 2 - series_out[1].sample_times[-1],
        (snr_best[0] ** 2 + snr_best[1] ** 2) ** 0.5,
        marker='x',
        color='k',
        zorder=100
    )

    ax.axvspan(
        -search_time,
        0,
        color='g',
        zorder=-100,
        alpha=0.25
    )

    ax.set_xlim(
        left=-search_time*2.5,
        right=search_time*0.25
    )
    ax.grid(zorder=-50)
    ax.legend()
    ax.set_xlabel('Time from merger')
    ax.set_ylabel('SNR')
    ax.set_title(f'Template {snr_vals[0]}, {time_before / 86400} days before merger, {label} psd')
    fig.savefig(f'bestwf_snr_series_{label}.png')


def waveform_from_bank(bank_file, idx, waveform_params_shared, data_length):
        bank_wf = copy.deepcopy(waveform_params_shared)
        # Update waveform params to use the ones from the bank file
        bank_wf['tc'] = data_length
        bank_wf['mass1'] = bank_file['mass1'][idx]
        bank_wf['mass2'] = bank_file['mass2'][idx]
        bank_wf['inclination'] = bank_file['inclination'][idx]
        bank_wf['polarization'] = bank_file['polarization'][idx]
        bank_wf['spin1z'] = bank_file['spin1z'][idx]
        bank_wf['spin2z'] = bank_file['spin2z'][idx]
        #bank_wf['coa_phase'] = hfile['coa_phase'][idx]
        bank_wf['eclipticlatitude'] = bank_file['eclipticlatitude'][idx]
        bank_wf['eclipticlongitude'] = bank_file['eclipticlongitude'][idx]

        return bank_wf

"""
This is a module to convert the current FIR filter functions to use inpainting,
but with the same interface and outputs.
"""


def generate_lisa_pre_merger_psds_inpaint(
    psd_file,
    duration,
    sample_rate,
):
    """
    Load PSDs from a text file for use with inpainting methods.
    
    Parameters
    ----------
    psd_file : str
        Path to PSD file
    duration : float
        Duration of data in seconds
    sample_rate : float
        Sample rate in Hz
        
    Returns
    -------
    dict
        Dictionary with 'FD' key containing the frequency-domain PSD
    """
    flen = int(duration * sample_rate) // 2 + 1
    delta_f = 1 / duration
    delta_t = 1 / sample_rate
    td_psd_length = int(duration * sample_rate)

    psd = pycbc.psd.from_txt(
        psd_file, flen, delta_f, delta_f, is_asd_file=False
    )

    return {
        "FD": psd,
    }

def generate_data_lisa_pre_merger_inpaint(
    waveform_params,
    psds_for_datagen,
    zeroed_length,
    sample_rate,
    seed=137,
    zero_noise=False,
    no_signal=False,
):
    """
    Generate simulated LISA data with signal and noise using inpainting approach.
    
    The data is extended with zeros at the end to support variable search times.
    This differs from the FIR filter method which fixes the time-before-merger.
    
    Parameters
    ----------
    waveform_params : dict
        Waveform parameters including 'additional_end_data' for time shift
    psds_for_datagen : dict
        Dictionary of PSDs for each channel to generate noise
    zeroed_length : int
        Total length of output data (original + padding with zeros)
    sample_rate : float
        Sample rate in Hz
    seed : int, optional
        Random seed for noise generation (default: 137)
    zero_noise : bool, optional
        If True, set noise to zero (default: False)
    no_signal : bool, optional
        If True, don't add signal to data (default: False)
        
    Returns
    -------
    dict
        Dictionary of time series data for LISA_A and LISA_E channels
    """

    # Generate injection
    outs = pycbc.waveform.get_fd_det_waveform(
        ifos=['LISA_A','LISA_E'],
        **waveform_params
    )

    touts = {}
    strain_w = {}
    # Shift waveform so the merger is not at the end of the data
    for i, (k, v) in enumerate(outs.items()):
        outs[k] = v.cyclic_time_shift(-waveform_params['additional_end_data'])
        touts[k] = outs[k].to_timeseries()


        # Generate TD noise from the original PSDs
        strain_w[k] = pycbc.noise.noise_from_psd(
            len(touts[k]),
            touts[k].delta_t,
            psds_for_datagen[k],
            seed=seed + i,
        )

        # We need to make sure the noise times match the signal
        strain_w[k]._epoch = touts[k]._epoch

        # If zero noise, set noise to zero
        if zero_noise:
            strain_w[k] *= 0.0

        # Only add signal if no_signal=False
        if not no_signal:
            strain_w[k][:] += touts[k][:]

        # Extend the data to zeroed_length by adding zeroes at the end
        strain_w[k].resize(zeroed_length)
    
    return strain_w


@cache
def get_window(window_length):
    if window_length:
        return signal.windows.hann(window_length * 2 + 1)[:window_length]
    else:
        return None


def generate_waveform_lisa_pre_merger_inpaint(
        waveform_params,
        psds_for_whitening,
        sample_rate=0.2,
):
    """
    Generate LISA waveforms in frequency domain for matched filtering.
    
    Parameters
    ----------
    waveform_params : dict
        Waveform parameters for generation
    psds_for_whitening : dict
        Dictionary of PSDs (not used in this function but kept for API consistency)
    sample_rate : float, optional
        Sample rate in Hz (default: 0.2)
        
    Returns
    -------
    dict
        Dictionary of frequency-domain waveforms for LISA_A and LISA_E channels
    """
    logging.debug('Generating waveform')
    outs = pycbc.waveform.get_fd_det_waveform(
        ifos=['LISA_A','LISA_E'], **waveform_params
    )
    for ifo in ['LISA_A','LISA_E']:
        # to_timesries and to_frequencyseries are not the fastest way to
        # convert. However, if waveform generation dominates, then MEH.
        wf = outs[ifo].to_timeseries()
        # Cut last day from waveform, merger is at the end
        # Also need to cut from the start because of wraparound
        window_len = int(86400 * 0.1) // 5
        taper = windows.hann(2 * window_len)
        taper_start = taper[:window_len]
        taper_end = taper[window_len:]
        wf[0:window_len] = 0
        wf[window_len: 2*window_len] *= taper_start
        wf[len(wf) - 2*window_len:len(wf)-window_len] *= taper_end
        # Zero out everything up to window length (i.e. 1 day)
        wf[0:window_len] = 0
        # For the next day, ramp up the waveform according to a hann window
        wf[window_len: 2*window_len] *= taper_start
        # 2 days - 1 day before merger, ramp the window down
        wf[len(wf) - 2*window_len:len(wf)-window_len] *= taper_end
        # 2 day before merger to the end, just zero everything out
        wf[len(wf)-window_len:] = 0
        outs[ifo] = wf.to_frequencyseries()

    return outs



def apply_inpainting(data, psd, start_idx, end_idx):
    """
    Apply inpainting to data between start_idx and end_idx using the given PSD.
    
    Parameters
    ----------
    data : pycbc.types.TimeSeries
        Time series data to inpaint
    psd : pycbc.types.FrequencySeries
        Power spectral density
    start_idx : int
        Start index for inpainting region
    end_idx : int
        End index for inpainting region
        
    Returns
    -------
    pycbc.types.TimeSeries
        Inpainted time series data
    """
    invpsd = 1. / psd
    # Set the DC component (frequency = 0) of the inverse PSD to zero.
    # This prevents division by zero (or very small values) when computing 1/psd[0],
    # which would otherwise result in infinite or extremely large values at DC.
    invpsd.data[0] = 0.0
    # Where the PSD is zero, this indicates that the PSD is not valid - however this
    # means that the inverse PSD becomes infinite - set this back to zero
    invpsd.data[np.isinf(invpsd.data)] = 0.0
    # Use copy=True to avoid in-place modification of the input data.
    # This ensures that the original data is preserved and prevents unintended side effects.
    return gate_and_paint(data, start_idx, end_idx, invpsd, copy=True)


def pre_process_data_lisa_pre_merger_inpaint(
    data_timeseries,
    sample_rate,
    psds_for_whitening,
    inpaint_start,
    inpaint_end,
    gaps=None,
):
    """
    Truncate, inpaint and then over-whiten the data.
    
    Parameters
    ----------
    data_timeseries : dict
        Dictionary of time series data for each channel
    sample_rate : float
        Sample rate of the data
    psds_for_whitening : dict
        Dictionary of PSDs for each channel
    inpaint_start : int
        Start index for main inpainting region
    inpaint_end : int
        End index for main inpainting region
    gaps : list of tuples, optional
        List of (start_time, end_time) tuples specifying additional gaps to inpaint in seconds.
        For example: [(gap1_start_time, gap1_end_time), (gap2_start_time, gap2_end_time), ...]
        Times are converted to indices using the sample rate.
        
    Returns
    -------
    dict
        Dictionary of whitened frequency series data for each channel
    """
    
    # Initialize painted data with the main inpainting region
    data_painted = {
        channel: apply_inpainting(
            data_timeseries[channel],
            psds_for_whitening[channel],
            inpaint_start,
            inpaint_end
        ) 
        for channel in data_timeseries.keys()
    }
    
    # If additional gaps are specified, apply inpainting to each gap
    if gaps is not None:
        for gap_start_time, gap_end_time in gaps:
            # Convert times to indices
            gap_start_idx = int((gap_start_time - data_timeseries['LISA_A']._epoch) * sample_rate)
            gap_end_idx = int((gap_end_time - data_timeseries['LISA_A']._epoch) * sample_rate)
            data_painted = {
                channel: apply_inpainting(
                    data_painted[channel],
                    psds_for_whitening[channel],
                    gap_start_idx,
                    gap_end_idx
                ) 
                for channel in  data_timeseries.keys()
            }

    inv_psd = {
        channel: 1. / psd
        for channel, psd in psds_for_whitening.items()
    }
    for channel in inv_psd:
        inv_psd[channel][0] = 0.0

    # Overwhiten the data
    data_ow = {
        channel: data_paint.to_frequencyseries() * inv_psd[channel]
        for channel, data_paint in data_painted.items()
    }

    return data_ow


def compute_hh_inner_product(
    waveform_params,
    psds_for_whitening,
    sample_rate,
    zeroed_length,
    inpaint_start,
    epoch=0,
    gaps=None,
    waveforms=None,
):
    """
    Compute the (h|h)(t) inner product for a waveform.
    
    This function generates a waveform, creates a mask to handle the inpainting
    region and gaps, whitens the waveform, and computes the time-dependent
    self-inner product (h|h)(t).
    
    Parameters
    ----------
    waveform_params : dict
        Parameters for waveform generation
    psds_for_whitening : dict
        Dictionary of PSDs for each channel
    sample_rate : float
        Sample rate of the data in Hz
    zeroed_length : int
        Total length of the data in samples (including any zero padding)
    inpaint_start : int
        Index where inpainting starts (points after this are zeroed in mask)
    gaps : list of tuples, optional
        List of (start_time, end_time) tuples specifying gaps to zero in the mask in seconds.
        For example: [(gap1_start_time, gap1_end_time), (gap2_start_time, gap2_end_time), ...]
        Times are converted to indices using the sample rate.
        
    Returns
    -------
    dict
        Dictionary containing the (h|h)(t) time series for each channel
    """
    logging.debug('Generating waveform')
    # Generate waveforms
    if waveforms is None:
        waveforms = generate_waveform_lisa_pre_merger_inpaint(
            waveform_params,
            psds_for_whitening,
            sample_rate=sample_rate,
        )
    
    delta_t = 1.0 / sample_rate
    inner_products = {}
    
    # Compute (h|h)(t) for each channel
    for channel in waveforms:
        logging.debug(f'Computing inner product for channel {channel}')
        # Get the waveform time series
        waveform_ts = waveforms[channel].to_timeseries()
        
        logging.debug('Create a mask')
        mask = pycbc.types.TimeSeries(
            np.ones(zeroed_length, dtype=waveform_ts.dtype),
            delta_t=delta_t,
            epoch=epoch,
        )

        # Zero out points after inpaint_start
        mask[inpaint_start:] = 0
        
        # Add gaps to the mask by zeroing them out
        if gaps is not None:
            for gap_start_time, gap_end_time in gaps:
                # Convert times to indices
                gap_start_idx = int(gap_start_time * sample_rate)
                gap_end_idx = int(gap_end_time * sample_rate)
                mask[gap_start_idx:gap_end_idx] = 0
        
        # Compute inverse PSD and handle DC component
        invpsd = 1.0 / psds_for_whitening[channel]
        invpsd[0] = 0.0
        
        # Convert mask to frequency series
        maskfs = mask.to_frequencyseries()
        
        # Whiten the waveform (using sqrt of inverse PSD)
        waveform = (waveforms[channel].to_frequencyseries() * invpsd**0.5).to_timeseries()
        
        # Square the whitened waveform in time domain
        waveform_sq = waveform * waveform
        
        # Multiply mask by conjugate of squared waveform in frequency domain
        masked_waveform_sq_fs = maskfs * waveform_sq.to_frequencyseries().conj()
        
        # Convert back to time series - this holds (h|h)(t)
        inner_products[channel] = masked_waveform_sq_fs.to_timeseries()
    
    return inner_products