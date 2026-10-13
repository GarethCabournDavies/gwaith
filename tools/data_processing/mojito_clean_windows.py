"""Plan source-specific preprocessing chunks from the Mojito catalogue."""
import argparse
from pathlib import Path
import re
import h5py
import numpy as np


def plan_windows(directory,catalogue,output_directory,start,end,chunk_duration,before_days=40.,after_days=20.,time_offset=800.):
    if not np.isfinite([start,end,chunk_duration,before_days,after_days,time_offset]).all() or end<=start or chunk_duration<=0 or min(before_days,after_days)<0 or before_days+after_days<=0:
        raise ValueError('Require increasing finite times and positive chunk/window durations')
    with h5py.File(catalogue,'r') as f:
        group=f['Binaries'] if 'Binaries' in f else f
        times=group['TimeCoalescencePhenomTPHMSSBFrame'][:] if 'TimeCoalescencePhenomTPHMSSBFrame' in group else group['tc'][:]
        ids=group['ID'][:]
    if len(set(map(int,ids)))!=len(ids) or not np.isfinite(times).all():
        raise ValueError('Catalogue source IDs must be unique and times finite')
    mergers={int(source):float(tc)-time_offset for source,tc in zip(ids,times)}
    files={}
    for path in sorted(Path(directory).glob('*.h5')):
        match=re.search(r'(?:^|_)source(\d+)(?:_|\.)',path.name)
        if not match:continue
        source=int(match.group(1))
        if source not in mergers:raise ValueError(f'Unknown catalogue source {source}: {path}')
        if source in files:raise ValueError(f'Multiple raw files for source {source}; select one version')
        files[source]=path
    if not files:raise ValueError(f'No sourceN .h5 files found in {directory}')
    missing=set(mergers)-set(files)
    if missing:raise ValueError(f'Missing raw files for catalogue source IDs {sorted(missing)}')
    rows=[]
    for source,path in files.items():
        tc=mergers[source]
        for index in range(int(np.ceil((end-start)/chunk_duration))):
            lo=start+index*chunk_duration;hi=min(lo+chunk_duration,end)
            # Retain the complete core chunk when it intersects the source support window.
            if hi<=tc-before_days*86400. or lo>=tc+after_days*86400.:continue
            output=Path(output_directory)/f'{path.stem}_chunk_{index}_downsampled.hdf'
            rows.append((str(path),str(output),lo,hi))
    if not rows:raise ValueError('No source windows intersect the requested time range')
    return rows


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data-directory',required=True)
    p.add_argument('--catalogue-file',required=True)
    p.add_argument('--output-directory',required=True)
    p.add_argument('--start-time',type=float,default=0.)
    p.add_argument('--end-time',type=float,required=True)
    p.add_argument('--chunk-duration',type=float,default=122*86400.)
    p.add_argument('--before-merger-days',type=float,default=40.)
    p.add_argument('--after-merger-days',type=float,default=20.)
    p.add_argument('--catalogue-time-offset',type=float,default=800.)
    a=p.parse_args()
    rows=plan_windows(a.data_directory,a.catalogue_file,a.output_directory,a.start_time,a.end_time,
                      a.chunk_duration,a.before_merger_days,a.after_merger_days,a.catalogue_time_offset)
    # Validate the whole plan before emitting rows consumed by the Bash workflow.
    for path,output,start,end in rows:
        if any(character in path+output for character in '\t\n'):
            raise ValueError('Paths cannot contain tabs or newlines')
    for path,output,start,end in rows:print(f'{path}\t{output}\t{start:.17g}\t{end:.17g}')


if __name__=='__main__':main()
