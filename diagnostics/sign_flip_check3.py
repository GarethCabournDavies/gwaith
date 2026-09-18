import sys, os
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from hwilas.waveforms.fastbhb import get_td_waveform_from_fastbhb
p = dict(mass1=1015522.4376, mass2=796849.1091,
         spin1z=0.972661*np.cos(0.9089723456902256),
         spin2z=0.972862*np.cos(1.1816994499122777), distance=17758.367941273442,
         inclination=2.5775388853549464, polarization=4.094550225563854,
         coa_phase=4.275929308696054, eclipticlongitude=0.6109268516130327,
         eclipticlatitude=-0.5641023936335077, tc=4800021.15572853)
tc = p['tc']; dt = 5.0
hw = get_td_waveform_from_fastbhb("FastBHB", **p, delta_t=dt,
      start_time=tc-6*86400, end_time=tc+3*86400)
hwA = np.asarray(hw['LISA_A'].data, dtype=float)
times = tc-6*86400 + dt*np.arange(len(hwA))
i_pk = int(np.argmax(np.abs(hwA)))
print(f"peak at t={times[i_pk]:.1f} (dt_from_tc={times[i_pk]-tc:+.1f}s), h={hwA[i_pk]:.3e}")
seg = hwA[max(0,i_pk-6):i_pk+7]
print("h around peak:", np.array2string(seg, precision=4, suppress_small=True))
print("times around peak:", np.array2string(times[max(0,i_pk-6):i_pk+7], precision=1))
# zero crossings (sign changes)
zcs = np.where(np.diff(np.sign(hwA)) != 0)[0]
print(f"n_zero_crossings={len(zcs)}")
if len(zcs):
    print("first 8 zc times:", np.array2string(times[zcs[:8]], precision=1))
# coarse spectral peak
spec = np.abs(np.fft.rfft(hwA*np.hanning(len(hwA))))
fq = np.fft.rfftfreq(len(hwA), dt)
ipk = int(np.argmax(spec))
print(f"dominant freq={fq[ipk]:.4f} Hz (period {1/fq[ipk]:.1f}s) at t~{times[ipk]:.0f}")
