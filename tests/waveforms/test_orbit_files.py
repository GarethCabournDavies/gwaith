"""ESA conversion caches by source content and keeps time/frame conventions explicit."""
from types import SimpleNamespace
import sys
import h5py
import numpy as np
from hwilas.waveforms.orbit_files import cached_esa_orbit_file, ROTATION, LINKS


def test_local_esa_conversion_and_content_addressed_cache(tmp_path,monkeypatch):
    files=[]
    for i in range(3):
        path=tmp_path/f'sc{i}.oem';path.write_text(str(i));files.append(path)
    calls=[]
    class Orbit:
        t_start=0.;t_end=1e9
        def __init__(self,*paths):calls.append(paths)
        def compute_position(self,t):
            assert t[0] in (10000.,10001.)
            return np.broadcast_to([1.,2.,3.],(len(t),3,3)).copy()
        def compute_velocity(self,t):return np.zeros((len(t),3,3))
        def compute_unit_vector(self,t,links):
            assert links==LINKS
            return np.broadcast_to([1.,0.,0.],(len(t),6,3)).copy()
        def compute_ltt(self,t,links):return np.full((len(t),6),8.3)
    monkeypatch.setitem(sys.modules,'lisaorbits',SimpleNamespace(OEMOrbits=Orbit))
    args=dict(oem_files=files,cache_dir=tmp_path/'cache')
    first=cached_esa_orbit_file(10000.,7200.,**args)
    second=cached_esa_orbit_file(10000.,7200.,**args)
    assert first==second and len(calls)==1
    with h5py.File(first) as f:
        assert f.attrs['data_time_origin']==10000.
        assert f.attrs['frame']=='ecliptic'
        assert f['tcb/x'].shape==(3,3,3)
        np.testing.assert_allclose(f['tcb/x'][0,0],ROTATION@np.array([1.,2.,3.]))
    files[0].write_text('changed-content')
    assert cached_esa_orbit_file(10000.,7200.,**args)!=first
    assert len(calls)==2
    assert cached_esa_orbit_file(10001.,7200.,**args)!=first
