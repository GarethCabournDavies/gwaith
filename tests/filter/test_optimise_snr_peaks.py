from types import SimpleNamespace
import h5py
import numpy as np
import pytest
from gwaith.types import FrequencySeries
from tools.search import optimise_snr_peaks as module


def test_reject_nested_event_and_particle_workers(monkeypatch,capsys):
    monkeypatch.setattr('sys.argv',['optimise','--input-file','unused',
        '--bank-file','unused','--data-files','unused','--psd-file','unused',
        '--event-workers','2','--n-processes','2'])
    with pytest.raises(SystemExit) as error:
        module.load_arguments()
    assert error.value.code==2
    assert 'Use event workers or particle workers, not both' in capsys.readouterr().err


def test_coherent_physical_distance_amplitude():
    h = FrequencySeries(np.array([0j,1+2j,3-1j]),delta_f=.25)
    amplitude = 2*np.exp(.4j)
    data = {c:FrequencySeries(amplitude*h.data,delta_f=.25) for c in ('A','E')}
    rho, fitted = module.coherent_fit(dict(A=h,E=h),data,
                                     FrequencySeries(np.ones(3),delta_f=.25),['A','E'])
    assert np.isfinite(rho)
    np.testing.assert_allclose(fitted,amplitude.real)
    assert np.isreal(fitted)


def test_loudest_event_selection(tmp_path):
    path = tmp_path/'events.hdf'
    with h5py.File(path,'w') as f:
        f['template_id'] = [0,1,2]
        for c in ('A','E'):
            f['trigger_id_'+c] = [0,1,2]
            f['snr_'+c] = [10.,30.,20.]
            f['time_'+c] = [100.,101.,200.]
    args = SimpleNamespace(input_file=path,optimisation_channels=['A','E'],
                           statistic_threshold=0.,minimum_peak_separation=10.)
    stats,times,_,ids = module.get_events_to_optimise(args)
    np.testing.assert_array_equal(ids,[1,2])
    assert np.all(np.diff(stats) <= 0)
    args.minimum_peak_separation = 0.
    stats,times,_,ids = module.get_events_to_optimise(args)
    np.testing.assert_array_equal(ids,[1,2,0])
    assert np.all(np.diff(stats) <= 0)


@pytest.mark.parametrize('previous_kind', ['hdf', 'csv', 'parameter'])
def test_excludes_candidates_near_previous_events(tmp_path, previous_kind):
    candidates = tmp_path/'candidates.hdf'
    with h5py.File(candidates, 'w') as f:
        f['template_id'] = [0, 1, 2]
        for c in ('A', 'E'):
            f['trigger_id_'+c] = [0, 1, 2]
            f['snr_'+c] = [30., 20., 10.]
            f['time_'+c] = [100., 150., 300.]

    args = dict(
        input_file=candidates,
        optimisation_channels=['A', 'E'],
        statistic_threshold=0.,
        minimum_peak_separation=40.,
    )

    if previous_kind == 'hdf':
        previous = tmp_path/'previous.hdf'
        with h5py.File(previous, 'w') as f:
            f['tc'] = [130.]
        args['previous_event_files'] = [str(previous)]
    elif previous_kind == 'parameter':
        previous = tmp_path/'previous_parameters.hdf'
        with h5py.File(previous, 'w') as f:
            f['tc'] = [130.]
        args['previous_parameter_file'] = str(previous)
    else:
        previous = tmp_path/'previous.csv'
        previous.write_text('event_id,time_s,statistic\n0,130.0,99.0\n')

        args['previous_event_files'] = [str(previous)]

    args = SimpleNamespace(**args)
    stats, times, _, ids = module.get_events_to_optimise(args)
    np.testing.assert_array_equal(ids, [2])
    np.testing.assert_array_equal(times, [300.])
    np.testing.assert_array_equal(stats, [np.sqrt(200.)])


def test_file_coverage_is_half_open():
    np.testing.assert_array_equal(module.identify_file_of_coinc(
        np.array([0.,10.]),np.array([10.,20.]),np.array([10.,19.])),[1,1])

@pytest.mark.parametrize('field', ['statistic', 'snr'])
def test_selection_uses_stored_candidate_statistic(tmp_path, field):
    path = tmp_path/'stored_statistic.hdf'
    with h5py.File(path, 'w') as f:
        f['template_id'] = [0, 1]
        f[field] = [60., 40.]
        for c in ('A', 'E'):
            f['trigger_id_'+c] = [0, 1]
            f['snr_'+c] = [1., 100.]
            f['time_'+c] = [100., 200.]
    args = SimpleNamespace(input_file=path, optimisation_channels=['A', 'E'],
                           statistic_threshold=50., minimum_peak_separation=0.)
    stats, _, _, ids = module.get_events_to_optimise(args)
    np.testing.assert_array_equal(ids, [0])
    np.testing.assert_array_equal(stats, [60.])
