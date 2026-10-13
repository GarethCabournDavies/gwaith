"""Inclusive multi-round seeds retain nearby signals and flag possible duplicates."""
import csv
import h5py
import numpy as np
import pytest
from hwilas.search.candidate_summary import combine_candidates, nearby_round_events
from tools.search.collect_search_candidates import main


def write_candidates(path,times,stats,label='network_snr'):
    with h5py.File(path,'w') as f:
        f.attrs['statistic_name']=label
        f['end_time']=times; f['statistic']=stats
        f['template_id']=np.zeros(len(times),dtype=int)


def test_neighbours_are_cross_round_and_never_vetoed(tmp_path):
    a,b=tmp_path/'a.hdf',tmp_path/'b.hdf'
    write_candidates(a,[10.,11.,100.],[50.,60.,70.])
    write_candidates(b,[12.,200.],[30.,20.])
    out=tmp_path/'all.hdf'
    rows=combine_candidates([a,b],out,3.)
    assert len(rows['statistic'])==5
    np.testing.assert_array_equal(rows['close_other_round'],[True,True,False,True,False])
    np.testing.assert_array_equal(rows['search_round'],[0,0,0,1,1])
    np.testing.assert_array_equal(rows['nearest_other_round_index'],[3,3,3,1,2])
    with h5py.File(out) as f:assert 'round_metadata' in f


def test_summary_cli_preserves_round_ids_and_statistic_labels(tmp_path,monkeypatch):
    a,b=tmp_path/'a.hdf',tmp_path/'b.hdf'
    write_candidates(a,[10.],[50.],'first');write_candidates(b,[12.],[80.],'second')
    out=tmp_path/'all.hdf';csv_file=tmp_path/'all.csv'
    monkeypatch.setattr('sys.argv',['collect','--input-files',str(a),str(b),
        '--output-file',str(out),'--csv-file',str(csv_file)])
    main()
    with open(csv_file) as f:
        reader=csv.DictReader(f)
        assert reader.fieldnames[:7]==['event_id','time_s','time_days','statistic','mchirp','eta','chieff']
        rows=list(reader)
    assert [r['search_round'] for r in rows]==['1','0']
    assert [r['statistic_name'] for r in rows]==['second','first']
    assert all(r['close_other_round']=='True' for r in rows)


def test_empty_round_and_no_cross_round_neighbour(tmp_path):
    a,b=tmp_path/'a.hdf',tmp_path/'b.hdf'
    write_candidates(a,[10.],[50.]);write_candidates(b,[],[])
    rows=combine_candidates([a,b],tmp_path/'out.hdf')
    assert not rows['close_other_round'][0]
    assert rows['nearest_other_round_index'][0]==-1
    assert np.isnan(rows['nearest_other_round_separation_seconds'][0])


def test_invalid_window():
    with pytest.raises(ValueError):nearby_round_events([10.],[0],-1.)


def test_one_hour_window_boundary():
    flags=nearby_round_events([0.,3600.,3600.01],[0,1,1],3600.)
    np.testing.assert_array_equal(flags['close_other_round'],[True,True,False])


def test_exclusion_prefers_retained_earlier_rounds():
    from hwilas.search.candidate_summary import retain_earlier_round_events
    # The stronger later candidate is excluded; excluded rows do not veto round 2.
    keep=retain_earlier_round_events([0.,3600.,3601.,7200.],[0,1,1,2],3600.)
    np.testing.assert_array_equal(keep,[True,False,True,False])
    np.testing.assert_array_equal(retain_earlier_round_events([0.,3600.,7200.],[0,1,2],3600.),[True,False,True])


def test_filtered_catalogue_preserves_row_identity(tmp_path):
    a,b=tmp_path/'a.hdf',tmp_path/'b.hdf'
    write_candidates(a,[0.],[50.]);write_candidates(b,[100.,4000.],[100.,20.])
    rows=combine_candidates([a,b],tmp_path/'out.hdf',3600.,True)
    np.testing.assert_array_equal(rows['end_time'],[0.,4000.])
    np.testing.assert_array_equal(rows['combined_event_index'],[0,2])
    np.testing.assert_array_equal(rows['round_event_index'],[0,1])


def test_round1_csv_excludes_prior_candidates(tmp_path,monkeypatch):
    from tools.search.candidate_hdf_to_csv import main as export
    a,b=tmp_path/'a.hdf',tmp_path/'b.hdf'
    write_candidates(a,[0.],[50.]);write_candidates(b,[3600.,3601.],[100.,20.])
    out=tmp_path/'round1.csv'
    monkeypatch.setattr('sys.argv',['export','--input-file',str(b),'--output-file',str(out),'--previous-event-files',str(a)])
    export()
    with open(out) as f:rows=list(csv.DictReader(f))
    assert len(rows)==1 and rows[0]['time_s']=='3601.0' and rows[0]['event_id']=='1'
