"""Angular seam handling preserves ordinary physical parameter bounds."""
from types import SimpleNamespace
import numpy as np
from tools.search.optimise_snr_peaks import CircularStar


def test_shortest_displacements_and_mixed_boundaries():
    lower = np.zeros(4)
    upper = np.array([2*np.pi, np.pi, 2*np.pi, 10.])
    topology = CircularStar(['eclipticlongitude', 'polarization', 'coa_phase', 'mchirp'], lower, upper)
    position = np.array([[2*np.pi-.1, np.pi-.1, .1, 9.]])
    target = np.array([[.1, .1, 2*np.pi-.1, 1.]])
    np.testing.assert_allclose(topology.displacement(target, position, (lower, upper)),
                               [[.2, .2, -.2, -8.]])
    swarm = SimpleNamespace(position=position, velocity=np.array([[.2, .2, -.2, 3.]]))
    np.testing.assert_allclose(topology.compute_position(swarm, (lower, upper)),
                               [[.1, .1, 2*np.pi-.1, 8.]])


def test_restricted_phase_prior_does_not_wrap():
    topology = CircularStar(['coa_phase'], np.array([0.]), np.array([1.]))
    assert not topology.circular[0]
    np.testing.assert_allclose(topology.displacement(np.array([[.1]]), np.array([[.9]]),
                                                    (np.array([0.]), np.array([1.]))), [[-.8]])


def test_velocity_moves_across_the_phase_seam(monkeypatch):
    monkeypatch.setattr(np.random, 'uniform', lambda **kwargs: np.ones(kwargs['size']))
    bounds = (np.array([0.]), np.array([2*np.pi]))
    topology = CircularStar(['coa_phase'], *bounds)
    swarm = SimpleNamespace(position=np.array([[2*np.pi-.1]]),
        pbest_pos=np.array([[.1]]), best_pos=np.array([.1]), velocity=np.zeros((1,1)),
        options={'w':0., 'c1':.5, 'c2':.5})
    velocity = topology.compute_velocity(swarm, vh=lambda velocity, *args, **kwargs: velocity,
                                         bounds=bounds)
    np.testing.assert_allclose(velocity, [[.2]])


def test_actual_pso_converges_across_phase_seam():
    import pyswarms as ps
    np.random.seed(42)
    bounds = (np.array([0.]), np.array([2*np.pi]))
    optimiser = ps.single.GlobalBestPSO(n_particles=12, dimensions=1,
        options={'c1':.5, 'c2':.3, 'w':.9}, bounds=bounds,
        init_pos=np.linspace(2*np.pi-.3, 2*np.pi-.01, 12)[:,None])
    optimiser.top = CircularStar(['coa_phase'], *bounds)
    def objective(position):
        return 1-np.cos(position[:,0]-.05)
    cost, position = optimiser.optimize(objective, iters=80, verbose=False)
    assert cost < 1e-6
    assert 0 <= position[0] < 2*np.pi
