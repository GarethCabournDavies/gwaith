"""Quick debug test for sky location variations."""
import sys
import os
import numpy as np
import h5py

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from diagnostics.test_sky_location import (
    load_signal_params, 
    load_noiseless_data, 
    generate_waveform, 
    compute_snr,
    DATA_FILE
)

def main():
    print("Loading data...")
    params = load_signal_params(DATA_FILE, signal_idx=0)
    aet_data, delta_t, epoch = load_noiseless_data(DATA_FILE)
    
    tc = params['tc']
    n_samples = len(aet_data['LISA_A'])
    
    print(f"Signal #0: M1={params['mass1']:.0f}, M2={params['mass2']:.0f}")
    print(f"  tc={tc:.1f}s, lat={params['eclipticlatitude']:.3f}, lon={params['eclipticlongitude']:.3f}")
    print(f"Data: n_samples={n_samples}, dt={delta_t:.3f}s")
    
    # Shorter window for quick test: 1 day before, 0.25 days after
    window_before = 1.0 * 86400
    window_after = 0.25 * 86400
    start_time = max(0.0, tc - window_before)
    end_time = min(n_samples * delta_t, tc + window_after)
    
    print(f"\nWindow: {start_time:.1f}s to {end_time:.1f}s")
    
    detector_times = epoch + delta_t * np.arange(n_samples)
    lo_idx = int(np.searchsorted(detector_times, start_time, side='left'))
    hi_idx = int(np.searchsorted(detector_times, end_time, side='right'))
    
    print(f"Data slice: [{lo_idx}, {hi_idx}) = {hi_idx - lo_idx} samples")
    
    # Test 1: Reference
    print("\n--- TEST 1: Reference parameters ---")
    wf_ref = generate_waveform(params, delta_t, start_time, end_time)
    print(f"Waveform generated. Lengths: A={len(wf_ref['LISA_A'])}, E={len(wf_ref['LISA_E'])}")
    
    for channel in ['LISA_A', 'LISA_E']:
        data_win = aet_data[channel][lo_idx:hi_idx]
        model_win = np.zeros(len(data_win))
        n_copy = min(len(wf_ref[channel]), len(model_win))
        model_win[:n_copy] = wf_ref[channel][:n_copy]
        
        print(f"  {channel}: data={len(data_win)}, model={len(model_win)}")
        snr = compute_snr(data_win, model_win, delta_t)
        print(f"  {channel}: SNR={snr}")
    
    # Test 2: Vary longitude
    print("\n--- TEST 2: Vary ecliptic longitude ---")
    for dlon in [-0.1, 0.0, 0.1]:
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
        
        print(f"  dlon={dlon:+.2f}: SNR_A={snrs[0]}, SNR_E={snrs[1]}")


if __name__ == '__main__':
    main()
