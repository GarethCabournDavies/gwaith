"""Add processed source segments on a common grid, leaving uncovered samples zero."""
import json
import logging
from pathlib import Path

import h5py
import numpy as np

logger=logging.getLogger(__name__)


def _grid_index(seconds,dt):
    index=round(seconds/dt)
    if not np.isfinite(seconds) or not np.isclose(seconds/dt,index,rtol=0,atol=1e-6):
        raise ValueError('Times must align with the output sampling grid')
    return index


def sum_files(paths,output_file,channels,start=None,end=None,delta_t=None,block_size=262144,
              multiplication_factors=None):
    """Stream overlapping segments into zero-filled HDF5 datasets, bounded in memory."""
    paths=[Path(path) for path in paths]
    factors=np.atleast_1d(np.asarray([] if multiplication_factors is None else multiplication_factors,dtype=float))
    if factors.ndim!=1 or len(factors) not in (0,1,len(paths)) or not np.isfinite(factors).all():
        raise ValueError('Multiplication factors must be finite, with 0, 1 or one value per input file')
    if len(factors)==0:
        factors=np.ones(len(paths))
    elif len(factors)==1:
        factors=np.full(len(paths),factors[0])
    if len({path.resolve() for path in paths})!=len(paths):
        raise ValueError('Duplicate input files would count the same signal twice')
    if Path(output_file).resolve() in {path.resolve() for path in paths}:
        raise ValueError('Output must differ from input files')
    if block_size<1 or not channels:
        raise ValueError('Positive block size and at least one channel are required')
    segments=[];metadata={};units=None;laser=None
    for path in paths:
        with h5py.File(path,'r') as f:
            dt=float(f.attrs['delta_t']);epoch=float(f.attrs['epoch'])
            if not np.isfinite(dt) or dt<=0 or not np.isfinite(epoch):
                raise ValueError(f'Invalid sampling metadata: {path}')
            if delta_t is None:
                delta_t=dt
            if dt!=delta_t:
                raise ValueError(f'Cadence differs in {path}: {dt} != {delta_t}')
            fractional=bool(f.attrs['divide_by_laser_frequency'])
            frequency=float(f.attrs['laser_frequency'])
            if not np.isfinite(frequency) or frequency<=0:
                raise ValueError(f'Invalid laser frequency: {path}')
            if units is not None and (fractional!=units or frequency!=laser):
                raise ValueError(f'Frequency units or laser frequency differ: {path}')
            units,laser=fractional,frequency
            if not metadata:
                metadata=dict(f.attrs)
            lengths=[]
            for c in channels:
                if c not in f or f[c].ndim!=1:
                    raise ValueError(f'Missing one-dimensional channel {c}: {path}')
                lengths.append(len(f[c]))
            if len(set(lengths))!=1:
                raise ValueError(f'Channel lengths differ: {path}')
            segments.append((path,epoch,lengths[0]))
    if delta_t is None or not np.isfinite(delta_t) or delta_t<=0:
        raise ValueError('Supply a positive cadence for an empty sum')
    if start is None:
        if not segments:raise ValueError('An empty sum needs explicit start/end times')
        start=min(epoch for _,epoch,_ in segments)
    if end is None:
        if not segments:raise ValueError('An empty sum needs explicit start/end times')
        end=max(epoch+length*delta_t for _,epoch,length in segments)
    if not np.isfinite([start,end]).all() or end<=start:
        raise ValueError('End must be finite and later than start')
    length=_grid_index(end-start,delta_t)
    if length<1:raise ValueError('Output grid is empty')
    offsets=[_grid_index(epoch-start,delta_t) for _,epoch,_ in segments]
    Path(output_file).parent.mkdir(parents=True,exist_ok=True)
    with h5py.File(output_file,'w') as out:
        out.attrs.update(metadata)
        out.attrs.update(epoch=float(start),delta_t=float(delta_t),
            analysis_start_time=float(start),analysis_end_time=float(end),
            divide_by_laser_frequency=units if units is not None else True,
            laser_frequency=laser if laser is not None else 2.816e14,
            source='summed_mojito_sources',source_segment='summed',source_segment_count=len(paths),
            combination='sum',input_files=json.dumps([str(path) for path in paths]),
            multiplication_factors=json.dumps(factors.tolist()),complete=False)
        # Unwritten HDF5 chunks read as zeros without allocating full arrays in RAM.
        for c in channels:
            out.create_dataset(c,shape=(length,),dtype='f8',chunks=(min(block_size,length),),
                               compression='gzip',compression_opts=1,shuffle=True,fillvalue=0.)
        for (path,epoch,input_length),offset,factor in zip(segments,offsets,factors):
            lo=max(0,offset);hi=min(length,offset+input_length)
            logger.info('Adding %s × %g at epoch %.3f; output samples %d:%d',path,factor,epoch,lo,hi)
            if hi<=lo or factor==0:continue
            with h5py.File(path,'r') as source:
                for c in channels:
                    for j in range(lo,hi,block_size):
                        stop=min(j+block_size,hi)
                        values=np.asarray(source[c][j-offset:stop-offset],dtype=float)
                        if not np.isfinite(values).all():
                            raise ValueError(f'Nonfinite samples in {path}, channel {c}')
                        combined=out[c][j:stop]+factor*values
                        if not np.isfinite(combined).all():
                            raise ValueError('Summed samples are nonfinite')
                        out[c][j:stop]=combined
        out.attrs['complete']=True
    return length
