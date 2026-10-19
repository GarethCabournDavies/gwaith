"""Exercise statistic-ordered row selection, physical conventions and native MBHBS readers."""
import math
import os
import sys
from pathlib import Path

import h5py
import numpy as np
import pytest
import yaml

from hwilas.search.pe_export import export_pe_seed

ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture
def source(tmp_path):
    path = tmp_path / 'fits.hdf'
    with h5py.File(path, 'w') as f:
        values = dict(combined_event_index=42, search_round=1, mass1=2e6,
                      mass2=1e6, spin1z=.2, spin2z=-.3, distance=12000.,
                      tc=2000000., eclipticlongitude=1.2, eclipticlatitude=-.4,
                      inclination=2., polarization=.6, coa_phase=.7,
                      optimisation_window_start=1900000., optimisation_window_end=2100000.)
        for key, value in values.items():
            f[key] = [value, value+1] if key == 'combined_event_index' else [value, value]
        f['optimised_stat'] = [20., 10.]
        f.create_group('round_metadata/1').attrs.update(
            waveform_model='phenomthm', f_ref=.0001, orbit='esa', data_time_origin=800.)
    return path


def run(source, tmp_path, **kwargs):
    return export_pe_seed(source, kwargs.pop('row_index', 0), tmp_path/'seed.yaml', 100., **kwargs)


def test_conventions_and_preservation(source, tmp_path):
    seed = run(source, tmp_path)
    assert seed['t_ref'] == 2000700.
    assert seed['phi_ref'] == pytest.approx(math.pi/2-.7)
    assert seed['f_ref'] == 0.
    assert seed['distance'] == 12000.
    assert seed['beta'] == -.4
    assert seed['chi2z'] == -.3
    p = yaml.safe_load((tmp_path/'seed.yaml').read_text())['provenance']
    assert p['derived_seed']['q'] == .5
    assert p['native_search']['combined_event_index'] == 42
    assert p['native_metadata']['f_ref'] == .0001
    assert p['reference_transfer']['approximate']
    assert p['truth'] is None
    assert {f.name for f in tmp_path.iterdir()} == {'fits.hdf', 'seed.yaml'}
    with h5py.File(source, 'a') as f:
        f['distance'][0] = 15000.
    run(source, tmp_path)
    assert yaml.safe_load((tmp_path/'seed.yaml').read_text())['distance'] == 15000.


@pytest.mark.parametrize('row_index', [-1, 2])
def test_out_of_range_row(source, tmp_path, row_index):
    with pytest.raises(ValueError, match='outside catalogue'):
        run(source, tmp_path, row_index=row_index)


@pytest.mark.parametrize('key,value', [('mass2', 3e6), ('distance', -1.),
                                      ('tc', np.nan), ('eclipticlatitude', 2.),
                                      ('spin1z', 1.1)])
def test_invalid_physical_values(source, tmp_path, key, value):
    with h5py.File(source, 'a') as f:
        f[key][0] = value
    with pytest.raises(ValueError):
        run(source, tmp_path)
    assert not (tmp_path/'seed.yaml').exists()


def test_external_provider_requires_convention(source, tmp_path):
    with h5py.File(source, 'a') as f:
        f['round_metadata/1'].attrs.update(waveform_model='bbhx', f_ref=0.)
    with pytest.raises(ValueError, match='phase convention'):
        run(source, tmp_path)
    seed = run(source, tmp_path, native_phase_convention='direct')
    assert seed['phi_ref'] == .7


def test_missing_origin_is_not_guessed(source, tmp_path):
    with h5py.File(source, 'a') as f:
        del f['round_metadata/1'].attrs['data_time_origin']
    with pytest.raises(ValueError, match='data_time_origin'):
        run(source, tmp_path)


def test_selection_uses_statistic_order_without_ids(source, tmp_path):
    with h5py.File(source, 'a') as f:
        del f['combined_event_index']
        f['optimised_stat'][:] = [10., 20.]
        f['distance'][:] = [100., 25000.]
    assert run(source, tmp_path)['distance'] == 25000.


def test_named_output_contains_only_starting_parameters(source, tmp_path):
    output = tmp_path/'nested/custom-event.yaml'
    contents = export_pe_seed(source, 0, output, 100.)
    assert list(output.parent.iterdir()) == [output]
    assert set(contents) == {'m1', 'm2', 'chi1z', 'chi2z', 'distance', 'inc',
                             'lam', 'beta', 'psi', 'phi_ref', 't_ref', 'f_ref',
                             'injection_frame', 'provenance'}
    assert contents['provenance']['native_search']['optimisation_window_start'] == 1900000.


def test_cli_requires_single_event(source, tmp_path, monkeypatch):
    from tools.search import export_pe_seed as cli
    args = ['hwilas_export_pe_seed', '--parameters', str(source),
            '--target-time-origin', '100', '--out', str(tmp_path/'custom.yaml')]
    monkeypatch.setattr(sys, 'argv', args)
    with pytest.raises(SystemExit) as exc:
        cli.main()
    assert exc.value.code == 2
    monkeypatch.setattr(sys, 'argv', [*args, '--row-index', '0'])
    cli.main()
    assert yaml.safe_load((tmp_path/'custom.yaml').read_text())['provenance']['row_index'] == 0


def test_actual_mbhbs_parameter_interface(source, tmp_path):
    """Verify starting values with MBHBS's real parameter conversion, not a PE run."""
    checkout = os.environ.get('MBHBS_SOURCE')
    if not checkout:
        pytest.skip('Set MBHBS_SOURCE to exercise the real MBHBS parameter interface')
    sys.path.insert(0, checkout)
    from mbhbs.bbhx_utils import get_bbhx_parameter_dict
    from bbhx.utils.constants import PC_SI
    seed = run(source, tmp_path)
    # The receiving configuration loader owns the documented Mpc-to-metre conversion.
    seed['distance'] *= 1e6*PC_SI
    parameters = get_bbhx_parameter_dict(seed, parameter_frame=seed['injection_frame'])
    assert parameters['distance'] == pytest.approx(12000.*1e6*PC_SI)
    assert parameters['m1'] == 2e6
    assert parameters['f_ref'] == 0.
