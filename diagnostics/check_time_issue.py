#!/usr/bin/env python3
"""Diagnostic: check if FastBHB waveform mismatch with Sangria data is a time issue.

Generates a FastBHB waveform using the local implementation and compares it
against the noiseless injection stream from the Sangria dataset, testing
various time offsets to see if a shift resolves the mismatch.

Usage:
    conda run -n env_hwilas_search python diagnostics/check_time_issue.py
"""
import sys
import os
import numpy as np
import h5py
from astropy import units as un

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

DATA_FILE = 'examples/datasets/LDC2_sangria_hm_training.hdf'


def main():
    print("=" * 70)
    print("FastBHB vs Sangria noiseless data: time alignment diagnostic")
    print("=" * 70)

    with h5py.File(DATA_FILE, 'r') as f:
        attrs = dict(f['obs/tdi'].attrs)
        delta_t = float(attrs['dt'])
        epoch = float(attrs.get('t0', 0.0))
        print(f"\nData grid: dt={delta_t:.3f} s, t0={epoch:.1f} s")

        ds = f['sky/mbhb/tdi']
        X = np.asarray(ds['X'], dtype=float).ravel()
        Y = np.asarray(ds['Y'], dtype=float).ravel()
        Z = np.asarray(ds['Z'], dtype=float).ravel()
        n_samples = len(X)
        print(f"Noiseless TDI: {n_samples} samples, span={n_samples*delta_t:.0f} s")

        cat = f['sky/mbhb/cat']
        print(f"\nCatalog: {cat.shape[0]} signals")
        
        # Read first signal's parameters
        rec = cat[0]
        tc = float(rec['CoalescenceTime'])
        mass1 = float(rec['Mass1'])
        mass2 = float(rec['Mass2'])
        distance = float(rec['Distance'])
        eclat = float(rec['EclipticLatitude'])
        eclong = float(rec['EclipticLongitude'])
        incl = float(rec['InitialPolarAngleL'])
        pol = float(rec['InitialAzimuthalAngleL'])
        phi_ref = float(rec['PhaseAtCoalescence'])
        spin1 = float(rec['Spin1']) * np.cos(float(rec['PolarAngleOfSpin1']))
        spin2 = float(rec['Spin2']) * np.cos(float(rec['PolarAngleOfSpin2']))

    A_data = (Z - X) / np.sqrt(2.0)
    E_data = (X - 2.0*Y + Z) / np.sqrt(6.0)
    T_data = (X + Y + Z) / np.sqrt(3.0)
    data_aet = {'LISA_A': A_data, 'LISA_E': E_data, 'LISA_T': T_data}
    detector_times = epoch + delta_t * np.arange(n_samples)

    print(f"\nSignal #0: M1={mass1:.0f} M2={mass2:.0f} D={distance:.1f}")
    print(f"  tc={tc:.1f} s ({tc/86400:.1f} d), incl={incl:.3f}, psi={pol:.3f}")
    print(f"  phi_ref={phi_ref:.3f}, spin1z={spin1:.4f}, spin2z={spin2:.4f}")
    # --- Generate FastBHB waveform over a window around tc ---
    from hwilas.waveforms.fastbhb import get_td_waveform_from_fastbhb

    window_before = 1.0 * 86400
    window_after = 0.1 * 86400
    t_min = float(detector_times[0])
    t_max = float(detector_times[-1])
    lo = max(t_min, tc - window_before)
    hi = min(t_max, tc + window_after)
    n_gen = int(round((hi - lo) / delta_t)) + 1
    gen_times = np.linspace(lo, hi, n_gen)

    print(f"\nWindow: [{lo:.0f}, {hi:.0f}] s ({n_gen} samples)")
    print(f"  tc - lo = {tc - lo:.1f} s")

    params = {
        'mass1': mass1, 'mass2': mass2,
        'spin1z': spin1, 'spin2z': spin2,
        'distance': distance, 'inclination': incl, 'polarization': pol,
        'coa_phase': phi_ref, 'eclipticlongitude': eclong,
        'eclipticlatitude': eclat, 'tc': tc, 'delta_t': delta_t,
        'start_time': float(gen_times[0]), 'end_time': float(gen_times[-1]),
    }

    print("\nGenerating FastBHB waveform...")
    aet = get_td_waveform_from_fastbhb('FastBHB', tdi='1.5', **params)
    model_aet = {ch: np.asarray(ts.data, dtype=float) for ch, ts in aet.items()}
    print(f"  Generated {len(model_aet['LISA_A'])} samples")

    lo_idx = int(np.searchsorted(detector_times, gen_times[0], side='left'))
    hi_idx = int(np.searchsorted(detector_times, gen_times[-1], side='right'))
    n_copy = min(hi_idx - lo_idx, len(model_aet['LISA_A']))

    from hwilas.filter import matched_filter_timeseries
    from pycbc.types import TimeSeries

    print("\n--- Direct comparison (no shift) ---")
    for channel in ['LISA_A', 'LISA_E', 'LISA_T']:
        data_win = data_aet[channel][lo_idx:hi_idx]
        model_win = np.zeros(len(data_win))
        model_win[:n_copy] = model_aet[channel][:n_copy]
        win_epoch = float(epoch + delta_t * lo_idx)
        data_ts = TimeSeries(data_win, epoch=win_epoch, delta_t=delta_t)
        model_ts = TimeSeries(model_win, epoch=win_epoch, delta_t=delta_t)
        snr_ts = matched_filter_timeseries(model_ts, data_ts, n_samples=len(data_win))
        snr_peak = float(np.max(np.abs(snr_ts.data)))
        peak_idx = int(np.argmax(np.abs(snr_ts.data)))
        best_lag = (peak_idx if peak_idx <= len(data_win)//2 else peak_idx - len(data_win)) * delta_t
        de = float(np.sum(data_win**2))
        me = float(np.sum(model_win**2))
        print(f"  {channel}: SNR={snr_peak:.4f}, lag={best_lag:.1f}s, "
              f"E_data={de:.2e}, E_model={me:.2e}")



    # --- Time-shift search over +/- 20 samples ---
    print("\n--- Time-shift search (+/- 20 samples) ---")
    best_shifts = {}
    for channel in ['LISA_A', 'LISA_E', 'LISA_T']:
        data_win = data_aet[channel][lo_idx:hi_idx]
        n_data = len(data_win)
        best_snr = 0.0
        best_shift = 0
        for shift in range(-20, 21):
            model_win = np.zeros(n_data)
            s = max(0, shift)
            e = min(len(model_aet[channel]), n_data + shift)
            if s < e:
                model_win[max(0, -shift):max(0, -shift) + (e - s)] = model_aet[channel][s:e]
            win_epoch = float(epoch + delta_t * lo_idx)
            data_ts = TimeSeries(data_win, epoch=win_epoch, delta_t=delta_t)
            model_ts = TimeSeries(model_win, epoch=win_epoch, delta_t=delta_t)
            snr_ts = matched_filter_timeseries(model_ts, data_ts, n_samples=n_data)
            snr_peak = float(np.max(np.abs(snr_ts.data)))
            if snr_peak > best_snr:
                best_snr = snr_peak
                best_shift = shift
        best_shifts[channel] = (best_shift, best_snr)
        print(f"  {channel}: best shift={best_shift:+d} samples ({best_shift*delta_t:+.1f}s), SNR={best_snr:.4f}")

    # --- Test with epoch=0 (Sangria reference style) ---
    print("\n--- Test with epoch=0 (ignore t0, match Sangria reference) ---")
    detector_times_0 = delta_t * np.arange(n_samples)
    lo_idx_0 = int(np.searchsorted(detector_times_0, gen_times[0], side='left'))
    hi_idx_0 = int(np.searchsorted(detector_times_0, gen_times[-1], side='right'))
    n_copy_0 = min(hi_idx_0 - lo_idx_0, len(model_aet['LISA_A']))
    for channel in ['LISA_A', 'LISA_E', 'LISA_T']:
        data_win = data_aet[channel][lo_idx_0:hi_idx_0]
        model_win = np.zeros(len(data_win))
        model_win[:n_copy_0] = model_aet[channel][:n_copy_0]
        win_epoch = float(delta_t * lo_idx_0)
        data_ts = TimeSeries(data_win, epoch=win_epoch, delta_t=delta_t)
        model_ts = TimeSeries(model_win, epoch=win_epoch, delta_t=delta_t)
        snr_ts = matched_filter_timeseries(model_ts, data_ts, n_samples=len(data_win))
        snr_peak = float(np.max(np.abs(snr_ts.data)))
        peak_idx = int(np.argmax(np.abs(snr_ts.data)))
        best_lag = (peak_idx if peak_idx <= len(data_win)//2 else peak_idx - len(data_win)) * delta_t
        print(f"  {channel}: SNR={snr_peak:.4f}, lag={best_lag:.1f}s")

    # --- Summary ---
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"t0 (epoch) = {epoch:.4f} s")
    if epoch != 0.0:
        print(f"WARNING: t0 is non-zero! This means the data grid starts at t={epoch:.1f}s,")
        print("but FastBHB's internal grid starts at t=0. The LISA orbit (detector")
        print("response) is evaluated at times relative to 0, not relative to t0.")
        print("This causes a phase/time misalignment.")
        print(f"\nFIX: Use epoch=0 for the data grid (match Sangria reference).")
    else:
        print("t0 is zero, so time origin mismatch is NOT the issue.")
        print("Look for other causes (parameter mapping, units, etc.).")


if __name__ == "__main__":
    main()
