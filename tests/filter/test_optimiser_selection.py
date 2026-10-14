"""PSO/DE CLI selection shares bounds, population size and the physical objective."""
from types import SimpleNamespace
from multiprocessing.pool import ThreadPool
import pickle
from functools import partial
import numpy as np
import pytest
from tools.search import optimise_snr_peaks as module


def cli(monkeypatch, *options):
    monkeypatch.setattr('sys.argv', ['optimise', '--input-file', 'unused', '--bank-file', 'unused',
        '--data-files', 'unused', '--psd-file', 'unused', '--output-parameter-file', 'unused', *options])
    return module.load_arguments()


def test_pso_default_and_de_options(monkeypatch):
    assert cli(monkeypatch).optimizer == 'pso'
    args = cli(monkeypatch, '--optimiser', 'DE', '--population-size', '24', '--n-iterations', '50')
    assert (args.optimizer, args.n_particles, args.n_iterations) == ('de', 24, 50)


@pytest.mark.parametrize('options', [('--population-size', '4'), ('--de-mutation', '.5', '2'),
    ('--de-recombination', '1.1'), ('--de-tol', '-1')])
def test_invalid_de_settings(monkeypatch, options):
    with pytest.raises(SystemExit) as error:
        cli(monkeypatch, '--optimizer', 'de', *options)
    assert error.value.code == 2


def test_de_real_solver_serial_and_parallel_agree(monkeypatch):
    args = SimpleNamespace(optimizer='de', seed=42, n_particles=24, n_iterations=50,
        n_processes=1, de_strategy='best1bin', de_mutation=(.5, 1.), de_recombination=.7,
        de_tol=0., de_atol=0.)
    monkeypatch.setattr(module, 'compute_statistic_core',
                        lambda x, target: 10.-np.sum((x-target)**2))
    lower, upper = np.full(2, -2.), np.full(2, 2.)
    population = np.random.default_rng(8).uniform(lower, upper, (24, 2))
    kwargs = dict(target=np.array([.25, -.5]))
    serial = module.optimise_population(args, population.copy(), lower, upper, kwargs, 0)
    class Context:
        def Pool(self, n):
            return ThreadPool(n)
    monkeypatch.setattr(module, 'get_context', lambda method: Context())
    args.n_processes = 2
    parallel = module.optimise_population(args, population.copy(), lower, upper, kwargs, 0)
    assert serial[0] == pytest.approx(-10., abs=1e-8)
    np.testing.assert_allclose(serial[1], kwargs['target'], atol=1e-4)
    assert serial[0] == parallel[0]
    np.testing.assert_array_equal(serial[1], parallel[1])
    objective = partial(module._de_objective, objective_kwargs=kwargs)
    assert pickle.loads(pickle.dumps(objective))(serial[1]) == serial[0]


def test_pso_dispatch_preserves_settings(monkeypatch):
    args = SimpleNamespace(optimizer='pso', seed=42, n_particles=5, n_iterations=3, n_processes=2)
    population = np.zeros((5, 9)); lower = np.full(9, -1.); upper = -lower
    class Swarm:
        def __init__(self, **kwargs):
            assert kwargs['n_particles'] == 5
            assert kwargs['options'] == {'c1': .5, 'c2': .3, 'w': .9}
        def optimize(self, objective, **kwargs):
            assert objective is module.compute_minus_statistic
            assert kwargs['iters'] == 3 and kwargs['n_processes'] == 2
            return -10., np.zeros(9)
    monkeypatch.setattr(module.ps.single, 'GlobalBestPSO', Swarm)
    assert module.optimise_population(args, population, lower, upper, {}, 0)[0] == -10.
