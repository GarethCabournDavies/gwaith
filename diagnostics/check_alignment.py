#!/usr/bin/env python3
"""Diagnostic: check FastBHB waveform alignment with Sangria data."""
import sys, os
import numpy as np
import h5py
import astropy.units as un

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA_FILE = 'examples/datasets/LDC2_sangria_hm_training.hdf'


def main():
    print("=" * 70)
    print("FastBHB vs Sangria: alignment diagnostic")
    print("=" * 70)

    with h5py.File(DATA_FILE, 'r') as f:
        attrs = dict(f['obs/tdi'].attrs)
        delta_t = float(attrs['dt'])
        epoch = float(attrs.get('t0', 0.0))
        print(f"\nData: dt={delta_t:.3f}s, t0={epoch:.1f}s")

        ds = f['sky/mbhb/tdi']
        X = np.asarray(ds['X'], dtype=float).ravel()
        Y = np.asarray(ds['Y'], dtype=float).ravel()
        Z = np.asarray(ds['Z'], dtype=float).ravel()
        n_samples = len(X)

        cat = f['sky/mbhb/cat']
        rec = cat[0]
        tc = float(rec['CoalescenceTime'][()]) if hasattr(rec['CoalescenceTime'], 'shape') else float(rec['CoalescenceTime'])
        mass1 = float(rec['Mass1'][()]) if hasattr(rec['Mass1'], 'shape') else float(rec['Mass1'])
        mass2 = float(rec['Mass2'][()]) if hasattr(rec['Mass2'], 'shape') else float(rec['Mass2'])
        distance = float(rec['Distance'][()]) if hasattr(rec['Distance'], 'shape') else float(rec['Distance'])
        eclat = float(rec['EclipticLatitude'][()]) if hasattr(rec['EclipticLatitude'], 'shape') else float(rec['EclipticLatitude'])
        eclong = float(rec['EclipticLongitude'][()]) if hasattr(rec['EclipticLongitude'], 'shape') else float(rec['EclipticLongitude'])
        incl = float(rec['InitialPolarAngleL'][()]) if hasattr(rec['InitialPolarAngleL'], 'shape') else float(rec['InitialPolarAngleL'])
        pol = float(rec['InitialAzimuthalAngleL'][()]) if hasattr(rec['InitialAzimuthalAngleL'], 'shape') else float(rec['InitialAzimuthalAngleL'])
        phi_ref = float(rec['PhaseAtCoalescence'][()]) if hasattr(rec['PhaseAtCoalescence'], 'shape') else float(rec['PhaseAtCoalescence'])
        spin1_mag = float(rec['Spin1'][()]) if hasattr(rec['Spin1'], 'shape') else float(rec['Spin1'])
        spin2_mag = float(rec['Spin2'][()]) if hasattr(rec['Spin2'], 'shape') else float(rec['Spin2'])
        spin1_polar = float(rec['PolarAngleOfSpin1'][()]) if hasattr(rec['PolarAngleOfSpin1'], 'shape') else float(rec['PolarAngleOfSpin1'])
        spin2_polar = float(rec['PolarAngleOfSpin2'][()]) if hasattr(rec['PolarAngleOfSpin2'], 'shape') else float(rec['PolarAngleOfSpin2'])

    A_data = (Z - X) / np.sqrt(2.0)
    E_data = (X - 2.0*Y + Z) / np.sqrt(6.0)
    T_data = (X + Y + Z) / np.sqrt(3.0)
    data_aet = {'LISA_A': A_data, 'LISA_E': E_data, 'LISA_T': T_data}
    detector_times = epoch + delta_t * np.arange(n_samples)

    spin1z = spin1_mag * np.cos(spin1_polar)
    spin2z = spin2_mag * np.cos(spin2_polar)

    print(f"\nSignal: M1={mass1:.0f} M2={mass2:.0f} D={distance:.1f}")
    print(f"  tc={tc:.1f}s, incl={incl:.3f}, psi={pol:.3f}, phi_ref={phi_ref:.3f}")
    print(f"  spin1z={spin1z:.4f}, spin2z={spin2z:.4f}")

    # Generate FastBHB waveform
    print("\nGenerating FastBHB waveform...")
    from ldc.lisa import orbits
    from ldc.waveform.lisabeta import FastBHB

    window_before = 1.0 * 86400
    window_after = 0.1 * 86400
    t_min = float(detector_times[0])
    t_max = float(detector_times[-1])
    lo = max(t_min, tc - window_before)
    hi = min(t_max, tc + window_after)

    template = {
        'Mass1': mass1, 'Mass2': mass2,
        'CoalescenceTime': tc, 'Distance': distance,
        'EclipticLatitude': eclat, 'EclipticLongitude': eclong,
        'Inclination': incl, 'Polarization': pol,
        'Spin1x': 0.0, 'Spin1y': 0.0, 'Spin1z': spin1z,
        'Spin2x': 0.0, 'Spin2y': 0.0, 'Spin2z': spin2z,
        'Spin1': spin1_mag, 'Spin2': spin2_mag,
        'PhaseAtCoalescence': phi_ref, 'Cadence': delta_t,
    }

    lisa_orbits = orbits.Orbits.type(dict({
        "nominal_arm_length": 2.5e9 * un.m,
        "initial_rotation": 0 * un.rad,
        "initial_position": 0 * un.rad,
        "orbit_type": "analytic",
    }))

    fast_hm = FastBHB("MBHB", approx="IMRPhenomHM", T=hi, delta_t=delta_t,
                      orbits=lisa_orbits, modes=[(2,2),(2,1),(3,3),(3,2),(4,4),(4,3)])
    A_model, E_model, T_model = fast_hm.get_td_tdiaet(template=template, tdi2=False)

    start_idx = int(lo / delta_t)
    end_idx = int(hi / delta_t)
    model_aet = {
        'LISA_A': A_model[start_idx:end_idx],
        'LISA_E': E_model[start_idx:end_idx],
        'LISA_T': T_model[start_idx:end_idx],
    }
    print(f"  Generated {len(model_aet['LISA_A'])} samples")

    # Direct comparison
    print("\n--- Direct comparison (no shift) ---")
    from hwilas.filter import matched_filter_timeseries
    from pycbc.types import TimeSeries

    gen_times = np.linspace(lo, hi, len(model_aet['LISA_A']))
    lo_idx = int(np.searchsorted(detector_times, gen_times[0], side='left'))
    hi_idx = int(np.searchsorted(detector_times, gen_times[-1], side='right'))
    n_copy = min(hi_idx - lo_idx, len(model_aet['LISA_A']))

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
        print(f"  {channel}: SNR={snr_peak:.4f}, lag={best_lag:.1f}s")

    # Time-shift search
    print("\n--- Time-shift search (+/- 20 samples) ---")
    for channel in ['LISA_A', 'LISA_E', 'LISA_T']:
        data_win = data_aet[channel][lo_idx:hi_idx]
        n_data = len(data_win)
        best_snr, best_shift = 0.0, 0
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
                best_snr, best_shift = snr_peak, shift
        print(f"  {channel}: best shift={best_shift:+d} ({best_shift*delta_t:+.1f}s), SNR={best_snr:.4f}")

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"t0 = {epoch:.4f} s")
    if epoch == 0.0:
        print("Time origin is NOT the issue (t0=0).")
        print("Check: parameter mapping, units, orbit model.")


if __name__ == "__main__":
    main()
