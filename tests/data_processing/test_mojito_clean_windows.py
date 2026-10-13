"""Mojito source-window selection and the clean-data Bash workflow."""
import os
from pathlib import Path
import shlex
import subprocess
import sys
import h5py
import numpy as np
import pytest
from tools.data_processing.mojito_clean_windows import plan_windows


def test_plan_uses_source_ids_and_overlapping_support(tmp_path):
    directory=tmp_path/'raw';directory.mkdir();catalogue=tmp_path/'catalogue.hdf'
    for i in [0,1,2]:(directory/f'MBHB_sample_source{i}_0.h5').touch()
    with h5py.File(catalogue,'w') as f:
        g=f.create_group('Binaries');g['ID']=[2,0,1]
        g['TimeCoalescencePhenomTPHMSSBFrame']=np.array([715.,10.,130.])*86400+800
    rows=plan_windows(directory,catalogue,tmp_path/'out',0,731*86400,122*86400)
    assert [(row[2]/86400,row[3]/86400) for row in rows]==[(0,122),(0,122),(122,244),(610,731)]
    assert all('source' in row[1] for row in rows)
    (directory/'MBHB_sample_source0_other.h5').touch()
    with pytest.raises(ValueError,match='Multiple raw files'):plan_windows(directory,catalogue,tmp_path/'out',0,731*86400,122*86400)


@pytest.mark.parametrize('merger_day,starts',[(102.,[0.]),(162.,[122.]),(160.,[0.,122.])])
def test_asymmetric_chunk_boundaries(tmp_path,merger_day,starts):
    directory=tmp_path/'raw';directory.mkdir()
    (directory/'MBHB_sample_source0_0.h5').touch()
    catalogue=tmp_path/'catalogue.hdf'
    with h5py.File(catalogue,'w') as f:
        g=f.create_group('Binaries');g['ID']=[0]
        g['TimeCoalescencePhenomTPHMSSBFrame']=[merger_day*86400+800]
    rows=plan_windows(directory,catalogue,tmp_path/'out',0.,244*86400.,122*86400.)
    assert [row[2]/86400 for row in rows]==starts


def test_bash_workflow_planning_and_real_sum(tmp_path):
    directory=tmp_path/'raw sources';directory.mkdir();catalogue=tmp_path/'catalogue.hdf'
    for i in [0,1,2]:(directory/f'MBHB_sample_source{i}_0.h5').touch()
    with h5py.File(catalogue,'w') as f:
        g=f.create_group('Binaries');g['ID']=[0,1,2]
        g['TimeCoalescencePhenomTPHMSSBFrame']=np.array([10.,130.,715.])*86400+800
    executables=tmp_path/'bin';executables.mkdir()
    for name,module in [('hwilas_mojito_clean_windows','mojito_clean_windows'),
                        ('hwilas_process_data_combine','process_data_combine')]:
        path=executables/name
        path.write_text('#!/bin/sh\nexec '+shlex.quote(sys.executable)+
                        f' -m tools.data_processing.{module} "$@"\n')
        path.chmod(0o755)
    # Stub only raw-data processing; planning and HDF5 summation use their real CLIs.
    processor=executables/'hwilas_process_mojito_data_chunk'
    processor.write_text(f'#!{sys.executable}\n'+r'''import argparse,re,h5py
p=argparse.ArgumentParser()
for name in ['data-file','output-file']:p.add_argument('--'+name)
for name in ['analysis-start-time','analysis-end-time','delta-t']:p.add_argument('--'+name,type=float)
a,_=p.parse_known_args()
source=int(re.search(r'source(\d+)_',a.data_file).group(1))
with h5py.File(a.output_file,'w') as f:
 f.attrs.update(epoch=a.analysis_start_time,delta_t=a.delta_t,divide_by_laser_frequency=True,laser_frequency=2.816e14)
 n=round((a.analysis_end_time-a.analysis_start_time)/a.delta_t)
 for i,c in enumerate(['LISA_A','LISA_E','LISA_T']):
  d=f.create_dataset(c,shape=(n,),dtype='f8',chunks=True,compression='gzip',fillvalue=0.)
  d[0]=(source+1)*(i+1)
''')
    processor.chmod(0o755)
    output=tmp_path/'workflow output'
    script=Path(__file__).resolve().parents[2]/'examples/data_processing/process_mojito_clean_data.sh'
    environment={**os.environ,'PATH':str(executables)+os.pathsep+os.environ['PATH']}
    subprocess.run(['bash',str(script),str(directory),str(catalogue),str(output)],
                   cwd=tmp_path,env=environment,check=True,capture_output=True,text=True,timeout=90)
    assert len(list((output/'data_processing').glob('*.hdf')))==4
    with h5py.File(output/'mojito_clean_summed.hdf') as f:
        assert len(f['LISA_A'])==731*86400//4
        assert f['LISA_A'][0]==3.
        assert f['LISA_A'][122*86400//4]==2.
        assert f['LISA_A'][500*86400//4]==0.
        assert f['LISA_A'][610*86400//4]==3.
        assert f.attrs['complete']
