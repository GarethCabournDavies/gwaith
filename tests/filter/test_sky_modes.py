"""Sky/helicity branch proposals preserve folded coordinates and optimiser state."""
from types import SimpleNamespace
import builtins
import pytest
import numpy as np
import hwilas.search.sky_modes as sky_module
from hwilas.search.sky_modes import sky_mode_vectors
from tools.search import optimise_snr_peaks as module


def test_64_branches_and_identity():
    pytest.importorskip("gwreparam")
    p=np.array([5e5,.2,.1,0.,1.,.3,1.,.2,.1])
    for tc in [86400.,4e7]:
        branches=sky_mode_vectors(p,tc)
        assert branches.shape==(64,9)
        assert np.isfinite(branches).all()
        np.testing.assert_array_equal(branches[0],p)
        np.testing.assert_array_equal(branches[:,:4],np.tile(p[:4],(64,1)))
        assert len(np.unique(np.round(branches,8),axis=0))==64
        assert np.all((branches[:,4]>=0)&(branches[:,4]<2*np.pi))
        assert np.all(abs(branches[:,5])<=np.pi/2)
        assert np.all((branches[:,7]>=0)&(branches[:,7]<np.pi))


def test_singular_inclination_uses_sky_branches():
    for inc in [0.,np.pi/2,np.pi]:
        p=np.array([5e5,.2,.1,0.,1.,.3,inc,.2,.1])
        assert sky_mode_vectors(p,4e7).shape==(8,9)


def test_native_pso_sweep_updates_returned_personal_best(monkeypatch):
    pytest.importorskip("gwreparam")
    p=np.array([5e5,.2,.1,0.,1.,.3,1.,.2,.1])
    tc=4e7
    target=sky_mode_vectors(p,tc)[20]
    width=np.array([1e6,.2,2.,2.,2*np.pi,np.pi,np.pi,np.pi,2*np.pi])
    monkeypatch.setattr(module,'compute_statistic_core',lambda x,**kw: 10-np.sum(((x-target)/width)**2))
    lower=np.array([1e5,.1,-.99,-.99,0.,-np.pi/2,0.,0.,0.])
    upper=np.array([1e6,.25,.99,.99,2*np.pi,np.pi/2,np.pi,np.pi,2*np.pi])
    args=SimpleNamespace(optimizer='pso',seed=1,n_particles=1,n_iterations=1,n_processes=1,sky_sweep_every=1)
    cost,best=module.optimise_population(args,p[None,:].copy(),lower,upper,dict(tc_reference=tc),0)
    assert cost==-10.
    np.testing.assert_allclose(best,target)


def test_de_sweeps_reinsert_winning_branch(monkeypatch):
    p=np.zeros(9); target=np.ones(9)*.5
    calls=[]
    def branches(x,tc):
        calls.append(1)
        return x[None,:] if len(calls)==1 else np.vstack([x,target])
    monkeypatch.setattr(module,'sky_mode_vectors',branches)
    monkeypatch.setattr(module,'compute_statistic_core',lambda x,**kw: 10-np.sum((x-target)**2))
    args=SimpleNamespace(optimizer='de',seed=1,n_particles=5,n_iterations=2,n_processes=1,
        sky_sweep_every=1,de_strategy='best1bin',de_mutation=(.5,1.),de_recombination=.7,de_tol=0.,de_atol=0.)
    cost,best=module.optimise_population(args,np.zeros((5,9)),np.full(9,-1.),np.ones(9),dict(tc_reference=1.),0)
    assert cost==-10.
    np.testing.assert_allclose(best,target)


@pytest.mark.parametrize("missing", ["gwreparam", "missing_transitive_dependency"])
def test_optional_import_and_fallback(monkeypatch, caplog, missing):
    original_import = builtins.__import__

    def guarded_import(name, *args, **kwargs):
        if name.startswith("gwreparam"):
            raise ModuleNotFoundError(f"No module named {missing}", name=missing)
        return original_import(name, *args, **kwargs)

    sky_module._gwreparam_functions.cache_clear()
    monkeypatch.setattr(builtins, "__import__", guarded_import)
    p = np.array([5e5, .2, .1, 0., 1., .3, 1., .2, .1])
    try:
        if missing != "gwreparam":
            with pytest.raises(ModuleNotFoundError, match=missing):
                sky_mode_vectors(p, 4e7)
            return
        for tc in [86400., 4e7]:
            branches = sky_mode_vectors(p, tc)
            assert branches.shape == (8, 9)
            assert np.isfinite(branches).all()
            np.testing.assert_array_equal(branches[0], p)
            np.testing.assert_array_equal(branches[:, :4], np.tile(p[:4], (8, 1)))
            np.testing.assert_array_equal(branches[:, 8], np.full(8, p[8]))
            assert len(np.unique(np.round(branches, 8), axis=0)) == 8
            assert np.all((branches[:, 4] >= 0) & (branches[:, 4] < 2*np.pi))
            assert np.all(abs(branches[:, 5]) <= np.pi/2)
            assert np.all((branches[:, 7] >= 0) & (branches[:, 7] < np.pi))
            np.testing.assert_allclose(branches[4:, 6], np.pi-p[6])
        assert caplog.text.count("gwreparam is unavailable") == 1
    finally:
        sky_module._gwreparam_functions.cache_clear()


def test_lisa_sky_reflection_and_rotations():
    branches = sky_module.lisa_sky_branches(.4, .3, .2, .6)
    np.testing.assert_allclose(branches[0], [.4, .3, .2, .6])
    np.testing.assert_allclose(branches[1], [.4+np.pi/2, .3, .2+np.pi/2, .6])
    np.testing.assert_allclose(branches[4], [.4, -.3, np.pi-.2, -.6])
    np.testing.assert_allclose(branches[6], [.4+np.pi, -.3, np.pi-.2, -.6])
