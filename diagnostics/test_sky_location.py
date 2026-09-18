"""Test sky location variations to diagnose SNR mismatch."""
import sys
import os
import numpy as np
import h5py

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

DATA_FILE = 'examples/datasets/LDC2_sangria_hm_training.hdf'


def load_signal_params(filename, signal_idx=0):
    """Load parameters for a specific signal from the catalog."""
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
    
    return params


def load_noiseless_data(filename):
    """Load noiseless AET data and metadata."""
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
    """Generate FastBHB waveform with given parameters."""
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
        'delta_t': delta_t,
        'start_time': start_time,
        'end_time': end_time,
    }
    
    aet = get_td_waveform_from_fastbhb('FastBHB', tdi='1.5', **wf_params)
    return {ch: np.asarray(ts.data, dtype=float) for ch, ts in aet.items()}


def compute_snr(data, model, delta_t):
    """Compute matched-filter SNR between data and model."""
    from gwaith.types import TimeSeries
    from hwilas.filter import matched_filter_timeseries
    
    n = len(data)
    data_ts = TimeSeries(np.asarray(data, dtype=float), epoch=0.0, delta_t=delta_t)
    model_ts = TimeSeries(np.asarray(model, dtype=float), epoch=0.0, delta_t=delta_t)
    


def main():
    print("=" * 70)
    print("Sky Location Variation Test")
    print("=" * 70)
    
    params = load_signal_params(DATA_FILE, signal_idx=0)
    aet_data, delta_t, epoch = load_noiseless_data(DATA_FILE)
    
    tc = params['tc']
    n_samples = len(aet_data['LISA_A'])
    
    print(f"\nSignal #0 parameters:")
    print(f"  M1={params['mass1']:.0f} M2={params['mass2']:.0f} D={params['distance']:.1f}")
    print(f"  tc={tc:.1f}s, incl={params['inclination']:.3f}, psi={params['polarization']:.3f}")
    print(f"  Ecliptic: lat={params['eclipticlatitude']:.3f}, lon={params['eclipticlongitude']:.3f}")
    print(f"  Data: dt={delta_t:.3f}s, t0={epoch:.1f}s, n_samples={n_samples}")
    
    window_before = 2.0 * 86400
    window_after = 0.5 * 86400
    start_time = max(0.0, tc - window_before)
    end_time = min(n_samples * delta_t, tc + window_after)
    
    print(f"\nWaveform window: {start_time:.1f}s to {end_time:.1f}s")
    
    lo_idx = int(np.searchsorted(epoch + delta_t * np.arange(n_samples), start_time, side='left'))
    hi_idx = int(np.searchsorted(epoch + delta_t * np.arange(n_samples), end_time, side='right'))
    
    print(f"Data slice: [{lo_idx}, {hi_idx}) = {hi_idx - lo_idx} samples")
    
    # Test 1: Reference parameters (baseline)
    print("\n" + "-" * 70)
    print("TEST 1: Reference sky location (baseline)")
    print("-" * 70)
    
    wf_ref = generate_waveform(params, delta_t, start_time, end_time)
    
    for channel in ['LISA_A', 'LISA_E', 'LISA_T']:
        data_win = aet_data[channel][lo_idx:hi_idx]
        model_win = np.zeros(len(data_win))
        n_copy = min(len(wf_ref[channel]), len(model_win))
        model_win[:n_copy] = wf_ref[channel][:n_copy]
        
        snr = compute_snr(data_win, model_win, delta_t)
        print(f"  {channel}: SNR={snr:.4f}")
    
    # Test 2: Vary ecliptic longitude by ±0.1 rad
    print("\n" + "-" * 70)
    print("TEST 2: Vary ecliptic longitude (±0.1 rad)")
    print("-" * 70)
    
    for dlon in [-0.1, -0.05, 0.05, 0.1]:
        params_test = params.copy()
        params_test['eclipticlongitude'] = params['eclipticlongitude'] + dlon
        
        wf_test = generate_waveform(params_test, delta_t, start_time, end_time)
        
        snrs = []
        for channel in ['LISA_A', 'LISA_E']:
            data_win = aet_data[channel][lo_idx:hi_idx]
            model_win = np.zeros(len(data_win))
            n_copy = min(len(wf_test[channel]), len(model_win))
            model_win[:n_copy] = wf_test[channel][:n_copy]
            
            snr = compute_snr(data_win, model_win, delta_t)
            snrs.append(snr)
        
        print(f"  dlon={dlon:+.2f} rad: SNR_A={snrs[0]:.4f}, SNR_E={snrs[1]:.4f}")
    
    # Test 3: Vary ecliptic latitude by ±0.1 rad
    print("\n" + "-" * 70)
    print("TEST 3: Vary ecliptic latitude (±0.1 rad)")
    print("-" * 70)
    
    for dlat in [-0.1, -0.05, 0.05, 0.1]:
        params_test = params.copy()
        params_test['eclipticlatitude'] = params['eclipticlatitude'] + dlat
        
        wf_test = generate_waveform(params_test, delta_t, start_time, end_time)
        
        snrs = []
        for channel in ['LISA_A', 'LISA_E']:
            data_win = aet_data[channel][lo_idx:hi_idx]
            model_win = np.zeros(len(data_win))
            n_copy = min(len(wf_test[channel]), len(model_win))
            model_win[:n_copy] = wf_test[channel][:n_copy]
            
            snr = compute_snr(data_win, model_win, delta_t)
            snrs.append(snr)
        
        print(f"  dlat={dlat:+.2f} rad: SNR_A={snrs[0]:.4f}, SNR_E={snrs[1]:.4f}")
    
    # Test 4: Vary coalescence time by ±50 samples
    print("\n" + "-" * 70)
    print("TEST 4: Vary coalescence time (±50 samples)")
    print("-" * 70)
    
    for dtc in [-50, -25, 25, 50]:
        params_test = params.copy()
        params_test['tc'] = params['tc'] + dtc * delta_t
        
        wf_test = generate_waveform(params_test, delta_t, start_time, end_time)
        
        snrs = []
        for channel in ['LISA_A', 'LISA_E']:
            data_win = aet_data[channel][lo_idx:hi_idx]
            model_win = np.zeros(len(data_win))
            n_copy = min(len(wf_test[channel]), len(model_win))
            model_win[:n_copy] = wf_test[channel][:n_copy]
            
            snr = compute_snr(data_win, model_win, delta_t)
            snrs.append(snr)
        
        print(f"  dtc={dtc:+d} samples ({dtc*delta_t:+.0f}s): SNR_A={snrs[0]:.4f}, SNR_E={snrs[1]:.4f}")
    
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print("If SNR is high for reference params → issue is elsewhere")
    print("If SNR peaks at non-zero sky offset → orbit model mismatch")
    print("If SNR peaks at non-zero time offset → systematic time shift")
    print("=" * 70)


if __name__ == '__main__':
    main()
