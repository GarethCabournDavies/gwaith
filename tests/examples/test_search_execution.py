"""Exercise workflow submission without running a search or requiring Slurm."""
import os
from pathlib import Path
import subprocess
import pytest

ROOT = Path(__file__).resolve().parents[2]


def test_slurm_worker_allocates_requested_cpus(tmp_path):
    mock = tmp_path / 'sbatch'
    mock.write_text('#!/bin/bash\nprintf "%s\\n" "$@" > "$CAPTURE"\necho 12345\n')
    mock.chmod(0o755)
    capture = tmp_path / 'arguments'
    env = dict(os.environ, PATH=f'{tmp_path}:{os.environ["PATH"]}',
               RUN_MODE='slurm', N_PROCESSES='6', SLURM_PARTITION='qmcompute', CAPTURE=str(capture))
    subprocess.run(['bash','-c','source "$1"; run_search_job 6 "$2" hwilas_optimise_snr_peaks --event-fraction 0/4',
                    'bash',str(ROOT/'examples/search/search_execution.sh'),str(tmp_path/'worker-%j.log')],
                   env=env,capture_output=True,text=True,check=True)
    args = capture.read_text().splitlines()
    assert '--cpus-per-task=6' in args
    assert '--partition=qmcompute' in args
    assert '--wait' in args
    assert any('--wrap=hwilas_optimise_snr_peaks' in arg for arg in args)


@pytest.mark.parametrize('mode,workers', [('invalid','4'),('bash','0'),('slurm','x')])
def test_invalid_execution_settings_fail(mode,workers):
    env=dict(os.environ,RUN_MODE=mode,N_PROCESSES=workers)
    result=subprocess.run(['bash','-c','source "$1"','bash',
                           str(ROOT/'examples/search/search_execution.sh')],env=env,
                          capture_output=True,text=True)
    assert result.returncode != 0


def test_bash_runs_without_scheduler(tmp_path):
    env=dict(os.environ,RUN_MODE='bash',N_PROCESSES='3')
    result=subprocess.run(['bash','-c',
        'source "$1"; run_search_job 3 "$2/log" echo "$n_processes"',
        'bash',str(ROOT/'examples/search/search_execution.sh'),str(tmp_path)],
        env=env,capture_output=True,text=True,check=True)
    assert result.stdout.strip() == '3'


def test_optimisation_submits_all_groups_before_merging(tmp_path):
    env=dict(os.environ,RUN_MODE='slurm',N_PROCESSES='6',OPTIMISATION_JOBS='3')
    script = r"""source "$1"
run_search_job() { printf '%s\n' "$*" > "$CAPTURE/job-${*: -1}"; }
hwilas_combine_search_parameters() { printf '%s\n' "$*" > "$CAPTURE/parameters"; }
hwilas_process_data_combine() { printf '%s\n' "$*" > "$CAPTURE/models"; }
hwilas_optimise_snr_peaks --n-processes 6 --output-parameter-file fit.hdf --output-timeseries-file model.hdf
"""
    env['CAPTURE']=str(tmp_path)
    subprocess.run(['bash','-c',script,'bash',str(ROOT/'examples/search/search_execution.sh')],
                   env=env,capture_output=True,text=True,check=True)
    jobs=list(tmp_path.glob('job-*'))
    assert len(jobs)==3
    for part in range(3):
        text=(tmp_path/f'job-model_part_{part}.hdf').read_text()
        assert f'--event-fraction {part}/3' in text
        assert '--n-processes 6' in text
    assert '--same-round' in (tmp_path/'parameters').read_text()
    assert 'model_part_2.hdf' in (tmp_path/'models').read_text()


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
               CALL_RECORD=str(record), RUN_MODE='bash', N_PROCESSES='2',
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
