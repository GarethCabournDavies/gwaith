"""Timing roots depend on each template, and HDF5 preserves the calculated cutoffs."""
import numpy as np
import h5py
from hwilas.template_banks.frequency import lower_frequency_for_duration
from hwilas.template_banks.io import write_bank, load_bank

SETTINGS=dict(f_lower=3e-5,f_upper=.09,pre_merger_days=1.)


def params(m1,m2,spin=0.):
    return dict(mass1=m1,mass2=m2,spin1z=spin,spin2z=0.,distance=1e4,
                eclipticlatitude=0.,eclipticlongitude=0.,inclination=1.,polarization=0.,coa_phase=0.)


def test_each_mass_and_spin_has_its_own_timing_root():
    from lisabeta.waveforms.bbh.pyIMRPhenomD import IMRPhenomDh22AmpPhase
    points=[params(1e6,5e5),params(5e6,2.5e6),params(1e6,5e5,.8)]
    cutoffs=[]
    for p in points:
        cutoff=lower_frequency_for_duration(p,SETTINGS)
        model=IMRPhenomDh22AmpPhase(np.array([1e-6]),p['mass1'],p['mass2'],p['spin1z'],p['spin2z'],p['distance'])
        assert abs(-float(model.compute_toff(cutoff))-86400)<1
        cutoffs.append(cutoff)
    assert len(set(cutoffs))==3
    assert cutoffs[1]<cutoffs[0]


def test_analysis_floor_is_respected():
    p=params(1e6,5e5)
    assert lower_frequency_for_duration(p,{**SETTINGS,'f_lower':.002})==.002


def test_bank_cutoffs_roundtrip_per_row(tmp_path):
    points=[params(1e6,5e5),params(5e6,2.5e6)]
    for p in points:
        p['f_lower']=lower_frequency_for_duration(p,SETTINGS)
    path=tmp_path/'bank.hdf'
    write_bank(path,points,SETTINGS)
    with h5py.File(path) as f:
        np.testing.assert_array_equal(f['f_lower'][:],[p['f_lower'] for p in points])
    assert [p['f_lower'] for p in load_bank(path)]==[p['f_lower'] for p in points]
