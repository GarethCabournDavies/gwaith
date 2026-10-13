"""Manual partitions preserve global event identity and merge complete catalogues."""
import h5py
import numpy as np
import pytest
from hwilas.search.partitions import event_partition, partition_range
from tools.search.combine_search_parameters import merge_partitions


@pytest.mark.parametrize('count,pieces', [(19,4),(3,8),(0,3),(20,1)])
def test_partitions_cover_events_exactly_once(count,pieces):
    ids=np.sort(np.concatenate([event_partition(count,f'{p}/{pieces}') for p in range(pieces)]))
    np.testing.assert_array_equal(ids,np.arange(count))


@pytest.mark.parametrize('fraction', ['2/2','-1/2','3/2','1/0','2','a/b','1/2/3'])
def test_invalid_partition(fraction):
    with pytest.raises(ValueError):event_partition(19,fraction)


def test_fraction_selects_interleaved_groups():
    np.testing.assert_array_equal(event_partition(10,'0/3'),[0,3,6,9])
    np.testing.assert_array_equal(event_partition(10,'1/3'),[1,4,7])
    np.testing.assert_array_equal(event_partition(10,'2/3'),[2,5,8])


def shard(path,ids,signature='same'):
    with h5py.File(path,'w') as f:
        f.attrs.update(selected_event_count=3,optimisation_signature=signature,pre_merger_days=1.)
        f['global_event_index']=ids
        f['tc']=np.asarray(ids)*100.
        f['optimised_stat']=100-np.asarray(ids)


def test_merge_sorts_global_event_ids_and_preserves_metadata(tmp_path):
    a,b=tmp_path/'a.hdf',tmp_path/'b.hdf'
    shard(a,[2]);shard(b,[0,1])
    out=tmp_path/'merged.hdf'
    table=merge_partitions([a,b],out)
    np.testing.assert_array_equal(table['global_event_index'],[0,1,2])
    with h5py.File(out) as f:
        assert f.attrs['pre_merger_days']==1.
        assert f.attrs['event_fraction']=='0/1'


@pytest.mark.parametrize('ids,signature', [([1],'same'),([0,1],'same'),([1,2],'other')])
def test_merge_rejects_missing_duplicate_or_incompatible_events(tmp_path,ids,signature):
    a,b=tmp_path/'a.hdf',tmp_path/'b.hdf'
    shard(a,[0]);shard(b,ids,signature)
    with pytest.raises(ValueError):merge_partitions([a,b],tmp_path/'out.hdf')


def test_cli_partitions_preserve_seeds_and_model_sum(tmp_path,monkeypatch):
    from tools.search import optimise_snr_peaks as module
    from hwilas.template_banks.io import write_bank
    from hwilas.data_processing.sum_timeseries import sum_files
    source,candidates,bank=[tmp_path/name for name in ['data.hdf','candidates.hdf','bank.hdf']]
    with h5py.File(source,'w') as f:
        f.attrs.update(epoch=0.,delta_t=4.,divide_by_laser_frequency=True,laser_frequency=2.816e14)
        for c in ['LISA_A','LISA_E','LISA_T']:f[c]=np.zeros(64)
    with h5py.File(candidates,'w') as f:
        f['template_id']=[0,1,2];f['statistic']=[100.,80.,60.]
        for c in ['LISA_A','LISA_E']:
            f['trigger_id_'+c]=[0,1,2];f['time_'+c]=[16.,32.,64.]
    p=dict(mass1=1e6,mass2=5e5,spin1z=0.,spin2z=0.,distance=1e4,
           eclipticlatitude=.2,eclipticlongitude=1.,inclination=1.,polarization=.2,coa_phase=0.)
    write_bank(bank,[p,p,p],dict(f_lower=.001,f_upper=.09,pre_merger_days=1.))
    def event(task):
        args,idx,count,path,time,channel_time,tid,stat,params,dt,flag,previous=task
        record={key:0. for key in module.PARAMETERS}
        record.update(tc=time,mass1=1e6,mass2=5e5,global_event_index=idx,initial_template_id=tid,
            initial_stat=stat,optimised_stat=stat+1,distance=1e4,amplitude_phase=0.,initial_time=time,
            initial_fit_stat=stat,f_lower=.001,optimisation_window_start=0.,optimisation_window_end=16.,
            in_previous_window=False,previous_event_index=-1,polish_evaluations=1,polish_success=True)
        signal=np.random.default_rng(args.seed+idx).normal(size=4)
        return record,idx*8.,{c:signal for c in args.output_timeseries_channels}
    monkeypatch.setattr(module,'_optimise_event',event)
    def run(fraction,offset,label):
        params,model=tmp_path/(label+'.hdf'),tmp_path/(label+'_model.hdf')
        monkeypatch.setattr('sys.argv',['optimise','--input-file',str(candidates),'--bank-file',str(bank),
            '--data-files',str(source),'--psd-file','unused','--statistic-threshold','0',
            '--minimum-peak-separation','0','--event-fraction',f'{offset}/{fraction}',
            '--output-parameter-file',str(params),'--output-timeseries-file',str(model)])
        module.main()
        return params,model
    complete=run(1,0,'complete');a=run(2,0,'part0');b=run(2,1,'part1')
    merged=tmp_path/'merged.hdf'
    merge_partitions([b[0],a[0]],merged)
    with h5py.File(complete[0]) as expected,h5py.File(merged) as actual:
        for k in expected:np.testing.assert_array_equal(expected[k][:],actual[k][:])
    merged_model=tmp_path/'model.hdf'
    sum_files([a[1],b[1]],merged_model,['LISA_A','LISA_E','LISA_T'])
    with h5py.File(complete[1]) as expected,h5py.File(merged_model) as actual:
        for c in ['LISA_A','LISA_E','LISA_T']:np.testing.assert_allclose(expected[c][:],actual[c][:],atol=1e-15)


@pytest.mark.parametrize('count,pieces', [(19,4),(3,8),(0,3),(20,1)])
def test_contiguous_partitions_cover_bank(count,pieces):
    ranges=[partition_range(count,f'{p}/{pieces}') for p in range(pieces)]
    ids=[i for start,end in ranges for i in range(start,end)]
    assert ids == list(range(count))
