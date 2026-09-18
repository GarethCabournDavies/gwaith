"""Sky location scan to diagnose zero-SNR mismatch between local FastBHB waveforms
and Sangria reference data.

Hypothesis: if the analytic orbit used by hwilas differs from the OEM orbit used
by Sangria, it's equivalent to evaluating the detector response at a different
time -- which would manifest as an effective sky-location (or time) offset that
maximizes the matched-filter SNR.
"""
import sys
import os
import numpy as np
import h5py

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

DATA_FILE = 'examples/datasets/LDC2_sangria_hm_training.hdf'


def load_signal_params(filename, signal_idx=0):
    with h5py.File(filename, 'r') as f:
        cat = f['sky/mbhb/cat']
        rec = cat[signal_idx]

        def get_scalar(field):
            val = rec[field]
            if hasattr(val, 'shape'):
                return float(np.asarray(val).ravel()[0])
            return float(val)

        params = {
            'mass1': get_scalar('Mass1'),
            'mass2': get_scalar('Mass2'),
            'distance': get_scalar('Distance'),
            'eclipticlatitude': get_scalar('EclipticLatitude'),
            'eclipticlongitude': get_scalar('EclipticLongitude'),
            'inclination': get_scalar('InitialPolarAngleL'),
            'polarization': get_scalar('InitialAzimuthalAngleL'),
            'coa_phase': get_scalar('PhaseAtCoalescence'),
            'tc': get_scalar('CoalescenceTime'),
        }

        spin1_mag = get_scalar('Spin1')
        spin2_mag = get_scalar('Spin2')
        spin1_polar = get_scalar('PolarAngleOfSpin1')
        spin2_polar = get_scalar('PolarAngleOfSpin2')

        params['spin1z'] = spin1_mag * np.cos(spin1_polar)
        params['spin2z'] = spin2_mag * np.cos(spin2_polar)
        # Pass spin polar angles so _build_ldc_template triggers the correct
        # aziPolAngleL2PsiIncl conversion of inclination/polarization.
        params['spin1_polar_angle'] = spin1_polar
        params['spin2_polar_angle'] = spin2_polar

    return params


def load_noiseless_data(filename):
    with h5py.File(filename, 'r') as f:
        tdi = f['sky/mbhb/tdi']
        attrs = dict(tdi.attrs)
        delta_t = float(attrs['dt'])
        epoch = float(attrs.get('t0', 0.0))

        X = np.asarray(tdi['X'], dtype=float).ravel()
        Y = np.asarray(tdi['Y'], dtype=float).ravel()
        Z = np.asarray(tdi['Z'], dtype=float).ravel()

    from hwilas.data_processing.tdi import linear_AET_conversion
    aet = linear_AET_conversion(X, Y, Z)

    return aet, delta_t, epoch


def generate_waveform(params, delta_t, start_time, end_time):
    from hwilas.waveforms.fastbhb import get_td_waveform_from_fastbhb

    wf_params = {
        'mass1': params['mass1'],
        'mass2': params['mass2'],
        'distance': params['distance'],
        'eclipticlatitude': params['eclipticlatitude'],
        'eclipticlongitude': params['eclipticlongitude'],
        'inclination': params['inclination'],
        'polarization': params['polarization'],
        'coa_phase': params['coa_phase'],
        'tc': params['tc'],
        'spin1z': params['spin1z'],
        'spin2z': params['spin2z'],
        'spin1_polar_angle': params.get('spin1_polar_angle', 0.0),
        'spin2_polar_angle': params.get('spin2_polar_angle', 0.0),
        'delta_t': delta_t,
        'start_time': start_time,
        'end_time': end_time,
    }

    aet = get_td_waveform_from_fastbhb('FastBHB', tdi='1.5', **wf_params)
    return {ch: np.asarray(ts.data, dtype=float) for ch, ts in aet.items()}


def normalized_cross_correlation(data, model):
    """Peak of the normalized cross-correlation between data and model.

    Returns (peak_value, shift_in_samples). Peak is in [-1, 1] when both signals
    are zero-mean / unit-variance over the overlap region.
    """
    from scipy.signal import correlate

    nz = np.nonzero(model)[0]
    if len(nz) == 0:
        return 0.0, 0
    m_lo, m_hi = nz[0], nz[-1] + 1
    d = data[m_lo:m_hi].astype(float)
    m = model[m_lo:m_hi].astype(float)

    d = (d - np.mean(d)) / (np.std(d) + 1e-30)
    m = (m - np.mean(m)) / (np.std(m) + 1e-30)

    cc = correlate(d, m, mode='full')
    cc = cc / len(d)
    peak_idx = int(np.argmax(cc))
    center = len(cc) // 2
    shift = peak_idx - center
    return float(cc[peak_idx]), shift


def scan_parameter(params, key, values, delta_t, start_time, end_time,
                   aet_data, lo_idx, hi_idx, label):
    """Scan one parameter over `values`, reporting corr peak and residual shift."""
    print(f"\n{'-' * 70}")
    print(f"SCAN: {label} (LISA_A channel)")
    print(f"{'-' * 70}")
    print(f"  {'offset':>12s}  {'corr_peak':>10s}  {'shift [smp]':>12s}")

    best = (None, -np.inf, 0)
    for val in values:
        p = params.copy()
        p[key] = val
        wf = generate_waveform(p, delta_t, start_time, end_time)

        data_win = aet_data['LISA_A'][lo_idx:hi_idx]
        model_win = np.zeros(len(data_win))
        n_copy = min(len(wf['LISA_A']), len(model_win))
        model_win[:n_copy] = wf['LISA_A'][:n_copy]

        peak, shift = normalized_cross_correlation(data_win, model_win)
        marker = "  <-- best" if peak > best[1] else ""
        print(f"  {val:+.4f}     {peak:+.4f}    {shift:+d}{marker}")
        if peak > best[1]:
            best = (val, peak, shift)

    return best


def main():
    print("=" * 70)
    print("Sky Location Scan -- FastBHB vs Sangria reference")
    print("=" * 70)

    params = load_signal_params(DATA_FILE, signal_idx=0)
    aet_data, delta_t, epoch = load_noiseless_data(DATA_FILE)

    tc = params['tc']
    n_samples = len(aet_data['LISA_A'])

    print(f"\nSignal #0:")
    print(f"  M1={params['mass1']:.0f}  M2={params['mass2']:.0f}  "
          f"D={params['distance']:.1f} Mpc")
    print(f"  tc={tc:.1f}s  incl={params['inclination']:.3f}  "
          f"psi={params['polarization']:.3f}")
    print(f"  lat={params['eclipticlatitude']:.4f}  "
          f"lon={params['eclipticlongitude']:.4f}")
    print(f"  Data: dt={delta_t:.2f}s, t0={epoch:.1f}s, n_samples={n_samples}")

    start_time = max(0.0, tc - 86400.0)
    end_time = min(n_samples * delta_t, tc + 0.25 * 86400.0)

    detector_times = epoch + delta_t * np.arange(n_samples)
    lo_idx = int(np.searchsorted(detector_times, start_time, side='left'))
    hi_idx = int(np.searchsorted(detector_times, end_time, side='right'))

    print(f"\nAnalysis window: {start_time:.1f}s to {end_time:.1f}s "
          f"({(hi_idx - lo_idx) * delta_t / 3600:.2f} h)")
    print(f"Data slice: [{lo_idx}, {hi_idx}) = {hi_idx - lo_idx} samples")

    # --- Baseline: reference parameters ---
    print("\n" + "-" * 70)
    print("BASELINE: reference sky location")
    print("-" * 70)

    wf_ref = generate_waveform(params, delta_t, start_time, end_time)

    for channel in ['LISA_A', 'LISA_E', 'LISA_T']:
        data_win = aet_data[channel][lo_idx:hi_idx]
        model_win = np.zeros(len(data_win))
        n_copy = min(len(wf_ref[channel]), len(model_win))
        model_win[:n_copy] = wf_ref[channel][:n_copy]

        peak, shift = normalized_cross_correlation(data_win, model_win)
        print(f"  {channel}: corr_peak={peak:+.4f}  shift={shift:+d} "
              f"samples ({shift * delta_t:+.0f}s)")

    # --- Scans ---
    base_lon = params['eclipticlongitude']
    base_lat = params['eclipticlatitude']
    base_tc = params['tc']

    best_lon = scan_parameter(
        params, 'eclipticlongitude',
        [base_lon + d for d in np.linspace(-0.2, 0.2, 9)],
        delta_t, start_time, end_time, aet_data, lo_idx, hi_idx,
        'ecliptic longitude')

    best_lat = scan_parameter(
        params, 'eclipticlatitude',
        [base_lat + d for d in np.linspace(-0.2, 0.2, 9)],
        delta_t, start_time, end_time, aet_data, lo_idx, hi_idx,
        'ecliptic latitude')

    best_tc = scan_parameter(
        params, 'tc',
        [base_tc + d * delta_t for d in np.array([-50, -25, -10, 0, 10, 25, 50])],
        delta_t, start_time, end_time, aet_data, lo_idx, hi_idx,
        'coalescence time')

    # --- Summary ---
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"  Best dlon: {best_lon[0] - base_lon:+.4f} rad -> "
          f"corr={best_lon[1]:+.4f}, residual shift={best_lon[2]} smp")
    print(f"  Best dlat: {best_lat[0] - base_lat:+.4f} rad -> "
          f"corr={best_lat[1]:+.4f}, residual shift={best_lat[2]} smp")
    print(f"  Best dtc:  {(best_tc[0] - base_tc) / delta_t:+.0f} samples -> "
          f"corr={best_tc[1]:+.4f}, residual shift={best_tc[2]} smp")
    print()
    print("Interpretation:")
    print("  * Baseline corr_peak close to +1  -> waveforms agree, look elsewhere")
    print("  * Best at non-zero dlon/dlat      -> orbit-model (analytic vs OEM) mismatch")
    print("  * Best at non-zero dtc            -> systematic time shift in pipeline")
    print("=" * 70)


if __name__ == '__main__':
    main()
