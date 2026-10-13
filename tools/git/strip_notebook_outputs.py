"""Git clean filter: remove notebook execution results without editing local files."""
import json
import sys


def strip_outputs(notebook):
    notebook.get('metadata',{}).pop('widgets',None)
    notebook.get('metadata',{}).pop('signature',None)
    for cell in notebook.get('cells',[]):
        cell.get('metadata',{}).pop('execution',None)
        if cell.get('cell_type')=='code':
            cell['outputs']=[]
            cell['execution_count']=None
    return notebook


def main():
    notebook=strip_outputs(json.load(sys.stdin))
    json.dump(notebook,sys.stdout,ensure_ascii=False,indent=1)
    sys.stdout.write('\n')


if __name__=='__main__':main()
