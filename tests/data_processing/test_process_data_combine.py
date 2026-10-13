"""Timestamp-aligned combination and weighted subtraction, through core and CLI."""
import subprocess
import sys
import h5py
import numpy as np
import pytest

from hwilas.data_processing.sum_timeseries import sum_files

CHANNELS=['LISA_A','LISA_E','LISA_T']


def segment(path,epoch,values,dt=4.,fractional=True):
    with h5py.File(path,'w') as f:
        f.attrs.update(epoch=epoch,delta_t=dt,divide_by_laser_frequency=fractional,laser_frequency=2.816e14)
        for i,c in enumerate(CHANNELS):f[c]=np.asarray(values)*(i+1)


def test_sum_overlaps_gaps_and_clipping(tmp_path):
    a,b=tmp_path/'a.hdf',tmp_path/'b.hdf'
    segment(a,-4.,[10.,1.,2.,3.]);segment(b,8.,[4.,5.])
    out=tmp_path/'sum.hdf'
    sum_files([b,a],out,CHANNELS,start=0.,end=32.,delta_t=4.,block_size=2)
    with h5py.File(out) as f:
        for i,c in enumerate(CHANNELS):
            np.testing.assert_array_equal(f[c][:],np.array([1.,2.,7.,5.,0.,0.,0.,0.])*(i+1))
        assert f.attrs['epoch']==0 and f.attrs['delta_t']==4
        assert f.attrs['complete'] and f.attrs['divide_by_laser_frequency']
    # Output can be replaced on a rerun.
    sum_files([],out,CHANNELS,start=0,end=32,delta_t=4)
    with h5py.File(out) as f:assert np.count_nonzero(f['LISA_A'][:])==0


@pytest.mark.parametrize('fault',['cadence','units','alignment','duplicate'])
def test_sum_rejects_incompatible_inputs(tmp_path,fault):
    a,b=tmp_path/'a.hdf',tmp_path/'b.hdf'
    segment(a,0.,[1.,2.]);segment(b,1. if fault=='alignment' else 0.,[1.,2.],
                                dt=5. if fault=='cadence' else 4.,fractional=fault!='units')
    with pytest.raises(ValueError):
        sum_files([a,a] if fault=='duplicate' else [a,b],tmp_path/'out.hdf',CHANNELS)
    assert not (tmp_path/'out.hdf').exists()


def test_sum_cli(tmp_path):
    a=tmp_path/'a.hdf';segment(a,8.,[2.,3.]);out=tmp_path/'out.hdf'
    subprocess.run([sys.executable,'-m','tools.data_processing.process_data_combine',
        '--data-file',str(a),'--output-file',str(out),'--start-time','0','--end-time','24',
        '--delta-t','4'],check=True,capture_output=True,text=True)
    with h5py.File(out) as f:np.testing.assert_array_equal(f['LISA_A'][:],[0.,0.,2.,3.,0.,0.])


def test_contiguous_chunks_match_concatenation(tmp_path):
    a,b=tmp_path/'a.hdf',tmp_path/'b.hdf'
    segment(a,100.,[1.,2.,3.]);segment(b,112.,[4.,5.,6.])
    out=tmp_path/'combined.hdf'
    sum_files([b,a],out,CHANNELS)
    with h5py.File(out) as f:
        assert f.attrs['epoch']==100.
        assert f.attrs['analysis_end_time']==124.
        for i,c in enumerate(CHANNELS):
            np.testing.assert_array_equal(f[c][:],np.arange(1.,7.)*(i+1))


@pytest.mark.parametrize('factors,expected',[(None,5.),([],5.),([2.],10.),([2.,-1.],1.),([0.,1.],3.)])
def test_multiplication_factors(tmp_path,factors,expected):
    a,b=tmp_path/'a.hdf',tmp_path/'b.hdf'
    segment(a,4.,[2.]);segment(b,4.,[3.]);out=tmp_path/'combined.hdf'
    sum_files([a,b],out,CHANNELS,start=0,end=12,delta_t=4,multiplication_factors=factors)
    with h5py.File(out) as f:
        for i,c in enumerate(CHANNELS):
            np.testing.assert_array_equal(f[c][:],[0.,expected*(i+1),0.])


@pytest.mark.parametrize('factors',[[1.,2.,3.],[np.nan],[np.inf]])
def test_invalid_factors_are_rejected_before_writing(tmp_path,factors):
    a,b=tmp_path/'a.hdf',tmp_path/'b.hdf'
    segment(a,0.,[2.]);segment(b,0.,[3.]);out=tmp_path/'combined.hdf'
    with pytest.raises(ValueError,match='Multiplication factors'):
        sum_files([a,b],out,CHANNELS,multiplication_factors=factors)
    assert not out.exists()


def test_factors_cli_preserves_file_order(tmp_path):
    a,b=tmp_path/'a.hdf',tmp_path/'b.hdf'
    segment(a,0.,[2.]);segment(b,0.,[3.]);out=tmp_path/'combined.hdf'
    subprocess.run([sys.executable,'-m','tools.data_processing.process_data_combine',
        '--data-file',str(b),str(a),'--output-file',str(out),
        '--multiplication-factors','2','-1'],check=True,capture_output=True,text=True)
    with h5py.File(out) as f:np.testing.assert_array_equal(f['LISA_A'][:],[4.])


@pytest.mark.parametrize('factor_arguments,expected',[
    ([],12.),(['--multiplication-factors'],12.),
    (['--multiplication-factors','2'],24.),
    (['--multiplication-factors','1','-1','1'],6.),
])
def test_cli_factor_modes(tmp_path,factor_arguments,expected):
    inputs=[tmp_path/f'input{i}.hdf' for i in range(3)]
    for path,value in zip(inputs,[2.,3.,7.]):segment(path,0.,[value])
    out=tmp_path/'combined.hdf'
    subprocess.run([sys.executable,'-m','tools.data_processing.process_data_combine',
        '--data-file',*map(str,inputs),'--output-file',str(out),*factor_arguments],
        check=True,capture_output=True,text=True)
    with h5py.File(out) as f:
        for i,c in enumerate(CHANNELS):np.testing.assert_array_equal(f[c][:],[expected*(i+1)])


def test_cli_invalid_factor_count_leaves_no_output(tmp_path):
    inputs=[tmp_path/f'input{i}.hdf' for i in range(3)]
    for path in inputs:segment(path,0.,[1.])
    out=tmp_path/'combined.hdf'
    result=subprocess.run([sys.executable,'-m','tools.data_processing.process_data_combine',
        '--data-file',*map(str,inputs),'--output-file',str(out),'--multiplication-factors','1','-1'],
        capture_output=True,text=True)
    assert result.returncode!=0
    assert 'Multiplication factors' in result.stderr
    assert not out.exists()
