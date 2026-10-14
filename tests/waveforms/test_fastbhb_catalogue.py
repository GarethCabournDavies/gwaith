"""Reference conventions preserve polarization tensors and a common intrinsic waveform."""
import numpy as np
import pytest
from hwilas.waveforms.fastbhb_catalogue import EQ_TO_ECL, ecliptic_polarization, spherical_harmonic


def basis(lon, lat, psi):
    u = np.array([np.sin(lon), -np.cos(lon), 0.])
    v = np.array([-np.sin(lat)*np.cos(lon), -np.sin(lat)*np.sin(lon), np.cos(lat)])
    return np.cos(psi)*u+np.sin(psi)*v, -np.sin(psi)*u+np.cos(psi)*v


@pytest.mark.parametrize('ra,dec,psi', [(1.1,.4,.6), (4.1,-.8,2.5), (.2,1.3,.1)])
def test_polarization_rotation_preserves_both_tensor_components(ra, dec, psi):
    direction = EQ_TO_ECL @ np.array([np.cos(dec)*np.cos(ra), np.cos(dec)*np.sin(ra), np.sin(dec)])
    lon, lat = np.arctan2(direction[1], direction[0]) % (2*np.pi), np.arcsin(direction[2])
    mapped = ecliptic_polarization(dict(right_ascension=ra, eclipticlongitude=lon,
                                      eclipticlatitude=lat, polarization=psi))
    eq1, eq2 = [EQ_TO_ECL @ x for x in basis(ra, dec, psi)]
    ecl1, ecl2 = basis(lon, lat, mapped)
    np.testing.assert_allclose(np.outer(eq1,eq1)-np.outer(eq2,eq2),
                               np.outer(ecl1,ecl1)-np.outer(ecl2,ecl2), atol=1e-14)
    np.testing.assert_allclose(np.outer(eq1,eq2)+np.outer(eq2,eq1),
                               np.outer(ecl1,ecl2)+np.outer(ecl2,ecl1), atol=1e-14)


def test_harmonics_match_installed_lisabeta_convention():
    lal = pytest.importorskip('lal')
    for l,m in [(2,2),(2,1),(3,3),(3,2),(4,4),(4,3)]:
        for signed_m in [m,-m]:
            for theta in [0., .7, 2.1, np.pi]:
                assert spherical_harmonic(l,signed_m,theta,.3) == pytest.approx(
                    lal.SpinWeightedSphericalHarmonic(theta,.3,-2,l,signed_m), abs=1e-14)


def test_analytic_spline_reproduces_native_fastbhb_without_fitting():
    pytest.importorskip('ldc.waveform.lisabeta')
    pytest.importorskip('tdispline.numpy.tdi')
    from hwilas.waveforms.fastbhb_catalogue import FastBHBCatalogueComparison, AnalyticEquatorialOrbits
    ra,dec=1.1,.4
    direction=EQ_TO_ECL @ np.array([np.cos(dec)*np.cos(ra),np.cos(dec)*np.sin(ra),np.sin(dec)])
    row=dict(source_id=0,mass1=1e6,mass2=5e5,spin1z=.2,spin2z=-.1,distance=1e4,
             inclination=1.,polarization=.6,coa_phase=.3,right_ascension=ra,declination=dec,
             eclipticlongitude=np.arctan2(direction[1],direction[0])%(2*np.pi),
             eclipticlatitude=np.arcsin(direction[2]),phase_reference_frequency=3e-5,
             tc=10*86400.,data_time_origin=12345678.0)
    generator=FastBHBCatalogueComparison()
    epoch=row['tc']-900.;n=512;dt=4.
    gen,_,start,_=generator._generator(row,epoch,dt)
    config=dict(gen.wvf_pars['LISAconst']);config['Orbitt0']+=start
    generator.esa_orbits=AnalyticEquatorialOrbits(config,row['data_time_origin'])
    native=generator.native(row,epoch,n,dt)
    responses=generator.responses(row,epoch,n,dt)
    from hwilas.search.phase_diagnostics import alignment_metrics
    control=alignment_metrics(native,responses['Analytic + TDI spline'],dt)
    assert control['best_common_lag_seconds'] == 0.
    assert control['zero_lag']['AE_overlap'] > .995
    assert control['zero_lag']['AE_amplitude_ratio'] == pytest.approx(1.,abs=.02)
    for channel in native:
        np.testing.assert_array_equal(responses['Analytic + TDI spline'][channel],
                                      responses['ESA + TDI spline'][channel])
