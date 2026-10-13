import json
import subprocess
import sys
from pathlib import Path


def test_notebook_clean_filter_preserves_source_and_local_file(tmp_path):
    notebook={'metadata':{'widgets':{'generated':'state'},'kernelspec':{'name':'python3'}},
              'cells':[{'cell_type':'markdown','source':['Explanation'],'metadata':{}},
                       {'cell_type':'code','source':['print(1)'],'execution_count':7,
                        'metadata':{'tags':['keep'],'execution':{'time':'generated'}},
                        'outputs':[{'output_type':'stream','text':'1'}]}]}
    text=json.dumps(notebook);local=tmp_path/'executed.ipynb';local.write_text(text)
    script=Path(__file__).resolve().parents[1]/'tools/git/strip_notebook_outputs.py'
    result=subprocess.run([sys.executable,str(script)],input=text,capture_output=True,text=True,check=True)
    cleaned=json.loads(result.stdout)
    assert local.read_text()==text
    assert cleaned['cells'][1]['source']==['print(1)']
    assert cleaned['cells'][1]['outputs']==[] and cleaned['cells'][1]['execution_count'] is None
    assert cleaned['cells'][1]['metadata']=={'tags':['keep']}
    assert cleaned['metadata']=={'kernelspec':{'name':'python3'}}
