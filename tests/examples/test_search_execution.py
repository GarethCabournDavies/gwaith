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
    if dataset == 'mojito':
        final = tmp_path/'output/final'
        final.mkdir(parents=True)
        (final/'all_optimised_parameters.csv').write_text('combined_event_index,unused\n42,a\n7,b\n')
    subprocess.run(['bash',str(ROOT/f'examples/search/{dataset}_basic_search.sh')],
                   env=env, capture_output=True, text=True, check=True)
    calls = [json.loads(line) for line in record.read_text().splitlines()]
    searches = [args for name,args in calls if name=='hwilas_matched_filter_search']
    # Run two partitions for every enabled search stage; a completed round may be skipped.
    source = (ROOT/f'examples/search/{dataset}_basic_search.sh').read_text()
    enabled = sum(line.strip() == 'hwilas_matched_filter_search '+chr(92)
                  for line in source.splitlines())
    assert len(searches) == 2*enabled
    for args in searches:
        assert args[args.index('--channels')+1:args.index('--channels')+3]==['LISA_A','LISA_E']
        assert args[args.index('--tdi-generation')+1]==tdi
        assert dataset+'_bank.hdf' in args[args.index('--bank-file')+1]
        if dataset == 'mojito':
            residual_search = '/residual_test/' in args[args.index('--output-file')+1]
            assert args[args.index('--statistic-threshold')+1] == ('4' if residual_search else '6')
    fits = [args for name,args in calls if name=='hwilas_optimise_snr_peaks']
    enabled_fits = sum(line.strip() == 'hwilas_optimise_snr_peaks '+chr(92)
                       for line in source.splitlines())
    assert len(fits) == enabled_fits
    if dataset == 'mojito':
        for args in fits:
            for flag, value in {'--approximant':'BBHx', '--refinement-approximant':'BBHx',
                '--orbits':'esa', '--refinement-orbits':'esa', '--n-particles':'32', '--n-iterations':'100',
                '--polish-evaluations':'1000', '--optimisation-window-before':'86400',
                '--optimisation-window-after':'14400', '--refinement-generator-keywords':'response_nodes:4096'}.items():
                assert args[args.index(flag)+1] == value
            assert args[args.index('--fit-channels')+1:args.index('--fit-channels')+3] == ['LISA_A','LISA_E']
    if fits:
        assert '--previous-parameter-file' in fits[-1]
    residual_args = next(args for name,args in calls if name == 'hwilas_check_search_residuals')
    assert residual_args[residual_args.index('--signal-exclusion')+1] == '86400'
    assert '--output-statistic-plot' in residual_args
    names = [name for name,args in calls]
    assert names.count('hwilas_trigger_merge')==3
    assert names.count('hwilas_subtract_signals')==2
    if dataset == 'mojito':
        assert names[-3:] == ['hwilas_check_search_residuals', 'hwilas_export_pe_seed', 'hwilas_export_pe_seed']
        exports = [args for name, args in calls if name == 'hwilas_export_pe_seed']
        for row_index, exported in enumerate(exports):
            assert '--pe-config' not in exported
            assert '--sampler-config' not in exported
            for flag, value in {
                '--parameters': str(tmp_path/'output/final/all_optimised_parameters.hdf'),
                '--row-index': str(row_index), '--target-time-origin': '12345678.0',
                '--native-phase-convention': 'pi-over-2-minus-phase',
                '--out': str(tmp_path/f'output/final/pe/seed_{row_index}.yaml'),
            }.items():
                assert exported[exported.index(flag)+1] == value
    else:
        assert names[-1]=='hwilas_check_search_residuals'
