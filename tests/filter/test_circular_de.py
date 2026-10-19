"""Differential evolution crosses angular seams without opening physical bounds."""
import numpy as np
import pytest
from scipy.optimize import differential_evolution
from tools.search.optimise_snr_peaks import CircularDEStrategy

STRATEGIES = [base+cross for base in ('best1', 'rand1', 'rand2', 'best2',
    'randtobest1', 'currenttobest1') for cross in ('bin', 'exp')]


def test_mutation_uses_short_difference_and_wraps_phase():
    class RNG:
        def uniform(self, *args, size=None):
            return np.zeros(size) if size is not None else .5
        def shuffle(self, values):
            pass
        def integers(self, dimensions):
            return 0
    bounds = (np.array([0.]), np.array([2*np.pi]))
    strategy = CircularDEStrategy(['coa_phase'], *bounds, 'best1bin', (.5, 1.5), 1.)
    population = np.array([[2*np.pi-.1], [.1], [2*np.pi-.1], [1.], [2.], [3.]])
    np.testing.assert_allclose(strategy(0, population, RNG()), [0.], atol=1e-12)
    # A truncated phase prior keeps its ordinary difference and remains unwrapped.
    strategy = CircularDEStrategy(['coa_phase'], np.array([0.]), np.array([5.]),
                                   'best1bin', (.5, 1.5), 1.)
    np.testing.assert_allclose(strategy(0, population, RNG()), [np.pi], atol=1e-12)


@pytest.mark.parametrize('strategy_name', STRATEGIES)
def test_real_de_converges_with_angles_and_physical_coordinate(strategy_name):
    lower = np.zeros(4)
    upper = np.array([2*np.pi, np.pi, 2*np.pi, 10.])
    strategy = CircularDEStrategy(['eclipticlongitude', 'polarization', 'coa_phase', 'mchirp'],
                                  lower, upper, strategy_name, (.5, 1.), .8)
    rng = np.random.default_rng(42)
    population = rng.uniform(lower, upper, (32, 4))
    population[:, :3] = upper[:3]-rng.uniform(.01, .3, (32, 3))
    def objective(position):
        return (1-np.cos(position[0]-.05) + 1-np.cos(2*(position[1]-.05)) +
                1-np.cos(position[2]-.05) + (position[3]-2.)**2)
    result = differential_evolution(objective, list(zip(lower, upper)), init=population,
        strategy=strategy, seed=42, maxiter=250, tol=0., polish=False, updating='deferred')
    assert result.fun < 1e-5
    assert np.all(result.population >= lower)
    assert np.all(result.population <= upper)
