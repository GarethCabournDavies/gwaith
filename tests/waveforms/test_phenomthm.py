"""Exercise the mode/response adapter and its public sampling contract."""
import numpy as np
import pytest

from hwilas.waveforms import get_fd_waveform
from hwilas.waveforms.phenomthm import equatorial_angles
from hwilas.waveforms.sampling import detector_grid
from hwilas.waveforms.unified_generation import generate_waveform


def parameters():
    return dict(mass1=1e6, mass2=5e5, spin1z=.2, spin2z=-.1,
                distance=10000., inclination=1., polarization=.5,
                eclipticlongitude=1.2, eclipticlatitude=.3, coa_phase=.4,
                tc=800., f_lower=1e-4, f_ref=2e-4, response_nodes=128)


def test_equatorial_basis_rotation():
    ra, dec, psi = equatorial_angles(0., 0., 0.)
    assert ra == pytest.approx(0.)
    assert dec == pytest.approx(0.)
    assert psi == pytest.approx(-np.deg2rad(23.439291111))


def test_grid_preserves_epoch_and_excludes_end():
    times, dt = detector_grid(dict(start_time=123., end_time=139., delta_t=4.))
    np.testing.assert_array_equal(times, [123., 127., 131., 135.])
    assert dt == 4.
    with pytest.raises(ValueError):
        detector_grid(dict(delta_t=4., end_time=13.))


def test_real_phenomthm_waveform_channels_and_grid():
    pytest.importorskip('phenomxpy')
    pytest.importorskip('tdispline')
    p = parameters()
    times = np.arange(128)*4.
    raw = get_fd_waveform('PhenomTHM', delta_t=4., length=128,
                          ifos=['LISA_X', 'LISA_Y', 'LISA_Z', 'LISA_A'], **p)
    assert all(len(w) == len(times) and np.isfinite(w.data).all() for w in raw.values())
    assert np.max(np.abs(raw['LISA_A'].data)) > 0
    np.testing.assert_allclose(raw['LISA_A'].data,
                               (raw['LISA_Z'].data-raw['LISA_X'].data)/np.sqrt(2))
    unified = generate_waveform(p, 'PhenomTHM', times, channels=['LISA_A'], tdi_generation='2.0')
    np.testing.assert_allclose(unified['LISA_A'], raw['LISA_A'].data)


def test_invalid_grid_rejected_by_unified_interface():
    with pytest.raises(ValueError, match='uniformly sampled'):
        generate_waveform(parameters(), 'PhenomTHM', [0., 4., 9.])
