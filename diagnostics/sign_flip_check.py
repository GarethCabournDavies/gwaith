#!/usr/bin/env python3
"""Diagnostic: isolate the sign flip between hwilas FastBHB and LDC reference."""
import sys, os
import numpy as np
import h5py
import astropy.units as un

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA_FILE = 'examples/datasets/LDC2_sangria_hm_training.hdf'

# --- load signal 0 params ---
with h5py.File(DATA_FILE, 'r') as f:
    cat = f['sky/mbhb/cat']
    rec = cat[0]
    def s(k):
        return float(np.asarray(rec[k]).ravel()[0])
    p = {k: s(k) for k in ['Mass1','Mass2','Distance','CoalescenceTime',
         'EclipticLatitude','EclipticLongitude','InitialPolarAngleL',
         'InitialAzimuthalAngleL','PhaseAtCoalescence','Spin1','Spin2',
         'PolarAngleOfSpin1','PolarAngleOfSpin2']}
    ds = f['sky/mbhb/tdi']
    X, Y, Z = (np.asarray(ds[c], dtype=float).ravel() for c in ['X','Y','Z'])
    dt = float(ds.attrs['dt']); t0 = float(ds.attrs.get('t0', 0.0))

A_data = (Z - X)/np.sqrt(2); E_data = (X - 2*Y + Z)/np.sqrt(6); T_data = (X+Y+Z)/np.sqrt(3)
tc = p['CoalescenceTime']
lo = max(t0, tc - 1.0*86400); hi = min(t0 + dt*len(X), tc + 0.1*86400)
lo_i = int((lo - t0)/dt); hi_i = int((hi - t0)/dt)
data = {'LISA_A': A_data[lo_i:hi_i], 'LISA_E': E_data[lo_i:hi_i], 'LISA_T': T_data[lo_i:hi_i]}
print(f"window [{lo:.1f}, {hi:.1f}]  n={hi_i-lo_i}  dt={dt}")

spin1z = p['Spin1']*np.cos(p['PolarAngleOfSpin1'])
spin2z = p['Spin2']*np.cos(p['PolarAngleOfSpin2'])
print(f"M1={p['Mass1']:.0f} M2={p['Mass2']:.0f} D={p['Distance']:.0f} tc={tc:.1f}")

# --- reference: LDC FastBHB directly ---
from ldc.lisa import orbits
from ldc.waveform.lisabeta import FastBHB
lisa_orbits = orbits.Orbits.type(dict({"nominal_arm_length": 2.5e9*un.m,
    "initial_rotation": 0*un.rad, "initial_position": 0*un.rad, "orbit_type": "analytic"}))
fb_ref = FastBHB("MBHB", approx="IMRPhenomHM", T=hi, delta_t=dt, orbits=lisa_orbits,
                 modes=[(2,2),(2,1),(3,3),(3,2),(4,4),(4,3)])
template_ref = {'Mass1': p['Mass1'], 'Mass2': p['Mass2'], 'CoalescenceTime': tc,
    'Distance': p['Distance'], 'EclipticLatitude': p['EclipticLatitude'],
    'EclipticLongitude': p['EclipticLongitude'], 'Inclination': p['InitialPolarAngleL'],
    'Polarization': p['InitialAzimuthalAngleL'], 'Spin1x': 0.0, 'Spin1y': 0.0, 'Spin1z': spin1z,
    'Spin2x': 0.0, 'Spin2y': 0.0, 'Spin2z': spin2z,
    'Spin1': p['Spin1'], 'Spin2': p['Spin2'],
    'PhaseAtCoalescence': p['PhaseAtCoalescence'], 'Cadence': dt}
A_r, E_r, T_r = fb_ref.get_td_tdiaet(template=template_ref, tdi2=False)
n_copy = min(len(A_r), hi_i - lo_i)
ref = {'LISA_A': A_r[:n_copy], 'LISA_E': E_r[:n_copy], 'LISA_T': T_r[:n_copy]}

# --- hwilas path (current, with sign flip) ---
from hwilas.waveforms.fastbhb import get_td_waveform_from_fastbhb as get_fastbhb_waveform
hw = get_fastbhb_waveform("FastBHB",
    mass1=p['Mass1'], mass2=p['Mass2'], spin1z=spin1z, spin2z=spin2z,
    spin1_polar_angle=p['PolarAngleOfSpin1'], spin2_polar_angle=p['PolarAngleOfSpin2'],
    distance=p['Distance'], inclination=p['InitialPolarAngleL'],
    polarization=p['InitialAzimuthalAngleL'], coa_phase=p['PhaseAtCoalescence'],
    eclipticlongitude=p['EclipticLongitude'], eclipticlatitude=p['EclipticLatitude'],
    tc=tc, delta_t=dt, start_time=float(lo), end_time=float(hi))
hw = {k: np.asarray(v.data, dtype=float) for k, v in hw.items()}; n_hw = min(len(hw["LISA_A"]), n_copy)

print("\n--- comparison (first n_copy samples) ---")
for ch in ['LISA_A','LISA_E','LISA_T']:
    d = data[ch][:n_hw]; r = ref[ch][:n_hw]; h = hw[ch][:n_hw]
    def corr(a, b):
        a = a - a.mean(); b = b - b.mean()
        return float(np.dot(a, b)/np.sqrt(np.dot(a,a)*np.dot(b,b)))
    print(f"{ch}:  corr(data, ref)={corr(d,r):+.5f}   corr(data, hwilas)={corr(d,h):+.5f}")

# --- frequency-domain check on LISA_A: is hw = -ref or conj? ---
import numpy.fft as nf
ch = 'LISA_A'
rfd = nf.rfft(ref[ch][:n_hw]); hfd = nf.rfft(hw[ch][:n_hw])
print(f"\n{ch} FD:  max|hw+ref|/max|ref| = {np.max(np.abs(hfd+rfd))/np.max(np.abs(rfd)):.3e}")
print(f"{ch} FD:  corr(hw, -ref)        = {corr(hw[ch][:n_hw], -r)}")
# time-domain: is hw(t) == ref(-t)? (time-reversal test on a short stretch)
seg = n_hw//4
rev = np.zeros_like(r); rev[:seg] = r[seg-1::-1]
print(f"{ch} TD:  corr(hw, time-reversed ref segment) = {corr(hw[ch][:seg], rev[:seg]):+.5f}")
