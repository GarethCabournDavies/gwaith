"""Candidate CSVs retain search statistics and distinguish template parameters from fits."""
import csv
import h5py
import numpy as np
from hwilas.template_banks.io import write_bank
from tools.search.candidate_hdf_to_csv import write_candidate_csv


def test_candidate_csv_orders_snr_and_includes_template_parameters(tmp_path):
    p = dict(mass1=1e6,mass2=5e5,spin1z=.3,spin2z=-.3,distance=1e4,
        eclipticlatitude=.2,eclipticlongitude=1.,inclination=1.,polarization=.2,coa_phase=.1)
    bank=tmp_path/'bank.hdf'
    write_bank(bank,[p],dict(f_lower=.001,f_upper=.09,pre_merger_days=1.))
    rows=dict(template_id=np.array([0,0]), end_time=np.array([86400.,172800.]),
        snr=np.array([5.,13.]), snr_LISA_A=np.array([3.,5.]), snr_LISA_E=np.array([4.,12.]),
        time_LISA_A=np.array([86400.,172800.]),time_LISA_E=np.array([86400.,172800.]),
        complex_snr_LISA_E=np.array([4j,12j]))
    out=tmp_path/'candidates.csv'
    write_candidate_csv(out,rows,bank)
    with out.open() as f:
        reader=csv.DictReader(f)
        assert reader.fieldnames[:7]==['event_id','time_s','time_days','statistic','mchirp','eta','chieff']
        result=list(reader)
    assert result[0]['event_id']=='1' and float(result[0]['statistic'])==13.
    assert float(result[0]['time_days'])==2.
    assert float(result[0]['statistic_phase_LISA_E_rad'])==np.pi/2
    assert float(result[0]['chieff'])==.1
    assert float(result[0]['mass1'])==1e6
    assert float(result[0]['distance'])==1e4
    assert result[0]['parameter_source']=='template'
    assert list(result[0]).index('distance') < list(result[0]).index('parameter_source')


def test_legacy_coincidence_csv_and_empty_header(tmp_path):
    rows=dict(template_id=np.array([0]),snr_LISA_A=np.array([3.]),snr_LISA_E=np.array([4.]),
        time_LISA_A=np.array([10.]),time_LISA_E=np.array([12.]))
    out=tmp_path/'legacy.csv'
    write_candidate_csv(out,rows,statistic='legacy peak coincidence')
    with out.open() as f:
        row=next(csv.DictReader(f))
    assert float(row['time_s'])==11. and float(row['statistic'])==5.
    write_candidate_csv(out,{key:value[:0] for key,value in rows.items()})
    with out.open() as f:
        reader=csv.DictReader(f)
        assert 'statistic' in reader.fieldnames and list(reader)==[]


def test_optimised_export_normalises_legacy_statistic_name(tmp_path):
    from tools.search.candidate_hdf_to_csv import write_optimised_csv
    rows=dict(tc=np.array([86400.]),optimised_stat=np.array([50.]),
        mass1=np.array([1e6]),mass2=np.array([5e5]),spin1z=np.array([.3]),spin2z=np.array([0.]),
        initial_fit_snr=np.array([40.]))
    out=tmp_path/'optimised.csv'
    write_optimised_csv(out,rows)
    with out.open() as f:
        reader=csv.DictReader(f)
        assert reader.fieldnames[:7]==['event_id','time_s','time_days','statistic','mchirp','eta','chieff']
        row=next(reader)
    assert row['initial_fit_stat']=='40.0' and 'initial_fit_snr' not in row
