"""Check signed aligned spins, explicit angles and requested harmonic modes."""
import numpy as np
from hwilas.waveforms import fastbhb


def test_direct_ssb_angles_preserve_signed_spins():
    p = dict(mass1=1e6, mass2=5e5, spin1z=-.5, spin2z=.4,
             inclination=1.1, polarization=.7, angle_convention='fastbhb')
    t = fastbhb._build_ldc_template(p)
    assert t['Spin1'] == -.5 and t['Spin2'] == .4
    assert t['Inclination'] == 1.1 and t['Polarization'] == .7
    assert 'InitialPolarAngleL' not in t and 'PolarAngleOfSpin1' not in t


def test_requested_modes_reach_generator(monkeypatch):
    seen = {}
    class Generator:
        def __init__(self, *args, **kwargs):
            seen.update(kwargs)
        def get_td_tdiaet(self, **kwargs):
            return (np.zeros(100),)*3
    monkeypatch.setattr(fastbhb, 'FastBHB', Generator)
    monkeypatch.setattr(fastbhb, '_get_lisa_orbits', lambda: None)
    result = fastbhb.get_td_waveform_from_fastbhb('FastBHB', delta_t=4., length=100,
               mode_array=[(2,2)], ifos=['LISA_A'], angle_convention='fastbhb')
    assert seen['modes'] == [(2,2)]
    assert len(result['LISA_A']) == 100
