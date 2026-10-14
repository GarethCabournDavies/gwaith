"""Exercise direct search workflows using stub pipeline commands."""
import os
from pathlib import Path
import subprocess
import pytest

ROOT = Path(__file__).resolve().parents[2]


@pytest.mark.parametrize('dataset,tdi', [('mojito','2'),('sangria','1.5')])
def test_basic_search_runs_complete_joint_workflow(tmp_path, dataset, tdi):
    import json
    import sys
    import tomllib
    scripts = tomllib.loads((ROOT/'pyproject.toml').read_text())['project']['scripts']
    record = tmp_path/'calls.jsonl'
    stub = '''import json, os, pathlib, sys
with pathlib.Path(os.environ['CALL_RECORD']).open('a') as f:
    f.write(json.dumps([pathlib.Path(sys.argv[0]).name, sys.argv[1:]]) + '\\n')
'''
    for name in scripts:
        command = tmp_path/name
        command.write_text(f'#!{sys.executable}\n'+stub)
        command.chmod(0o755)
    env = dict(os.environ, PATH=f'{tmp_path}:{os.environ["PATH"]}',
               CALL_RECORD=str(record), N_PROCESSES='2', DATA_TIME_ORIGIN='12345678.0', DATA_FILE=str(tmp_path/'data.hdf'),
               OUTPUT_DIR=str(tmp_path/'output'))
    subprocess.run(['bash',str(ROOT/f'examples/search/{dataset}_basic_search.sh')],
                   env=env, capture_output=True, text=True, check=True)
    calls = [json.loads(line) for line in record.read_text().splitlines()]
    searches = [args for name,args in calls if name=='hwilas_matched_filter_search']
    assert len(searches)==6  # Two partitions in each round and final residual search.
    for args in searches:
        assert args[args.index('--channels')+1:args.index('--channels')+3]==['LISA_A','LISA_E']
        assert args[args.index('--tdi-generation')+1]==tdi
        assert dataset+'_bank.hdf' in args[args.index('--bank-file')+1]
    fits = [args for name,args in calls if name=='hwilas_optimise_snr_peaks']
    assert len(fits)==2
    assert '--previous-parameter-file' in fits[1]
    names = [name for name,args in calls]
    assert names.count('hwilas_trigger_merge')==3
    assert names.count('hwilas_subtract_signals')==2
    assert names[-1]=='hwilas_check_search_residuals'
