import sys, os
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from hwilas.waveforms.fastbhb import get_td_waveform_from_fastbhb
# minimal: one mode (2,2), small window around tc
p = dict(mass1=1015522.4376, mass2=796849.1091, spin1z=0.972661*np.cos(0.9089723456902256),
         spin2z=0.972862*np.cos(1.1816994499122777), distance=17758.367941273442,
         inclination=2.5775388853549464, polarization=4.094550225563854,
         coa_phase=4.275929308696054, eclipticlongitude=0.6109268516130327,
         eclipticlatitude=-0.5641023936335077, tc=4800021.15572853)
tc = p['tc']; dt = 5.0
hw = get_td_waveform_from_fastbhb("FastBHB", **p, delta_t=dt,
      start_time=tc-3600, end_time=tc+3600)
hwA = np.asarray(hw['LISA_A'].data, dtype=float)
times = tc-3600 + dt*np.arange(len(hwA))
# find peak
i_pk = int(np.argmax(np.abs(hwA)))
print(f"peak at t={times[i_pk]:.1f} (dt_from_tc={times[i_pk]-tc:+.1f}s), h={hwA[i_pk]:.3e}")
print("h(t) around peak:", np.array2string(hwA[max(0,i_pk-4):i_pk+5], precision=3, suppress_small=True))
# print zero crossings / first 10 samples
print("first samples:", np.array2string(hwA[:10], precision=3, suppress_small=True))
