"""Verify installed-package discovery, lazy imports and public plugin dispatch."""
import sys
from importlib.metadata import EntryPoint

import numpy as np
import pytest

from hwilas.waveforms import available_approximants, get_fd_waveform
from hwilas.waveforms import plugins
from hwilas.waveforms.unified_generation import generate_waveform


def test_installed_provider_discovery_and_unified_generation(tmp_path, monkeypatch):
    info = tmp_path / 'example_waveforms-1.0.dist-info'
    info.mkdir()
    (info / 'METADATA').write_text('Metadata-Version: 2.1\nName: example-waveforms\nVersion: 1.0\n')
    (info / 'entry_points.txt').write_text('[hwilas.waveforms]\nExampleModel = external_waveform_fixture:generate\n')
    (tmp_path / 'external_waveform_fixture.py').write_text(
        'import numpy as np\nfrom gwaith.types import TimeSeries\n'
        'def generate(approximant, tdi, **params):\n'
        '    dt = params["delta_t"]\n    start = params.get("start_time", 0.)\n'
        '    n = params.get("length")\n'
        '    if n is None: n = round((params["end_time"]-start)/dt)\n'
        '    return {c: TimeSeries(np.arange(n, dtype=float), delta_t=dt, epoch=start)\n'
        '            for c in params.get("ifos", ["LISA_A", "LISA_E", "LISA_T"])}\n')
    monkeypatch.syspath_prepend(str(tmp_path))
    assert 'ExampleModel' in available_approximants()
    assert 'external_waveform_fixture' not in sys.modules
    raw = get_fd_waveform('examplemodel', length=8, delta_f=1/32., start_time=100.)
    assert raw['LISA_A'].delta_t == 4.
    assert float(raw['LISA_A'].epoch) == 100.
    signal = dict(mass1=1e6, mass2=5e5, spin1z=0., spin2z=0., distance=1000.,
                  inclination=1., polarization=.2, eclipticlongitude=.3,
                  eclipticlatitude=.4, phi_ref=0., tc=120.)
    result = generate_waveform(signal, 'EXAMPLEMODEL', 100.+np.arange(8)*4., channels=['LISA_A'])
    np.testing.assert_array_equal(result['LISA_A'], raw['LISA_A'].data)
    monkeypatch.delitem(sys.modules, 'external_waveform_fixture')


@pytest.mark.parametrize('names', [('FastBHB',), ('PhenomTHM',), ('Example', 'EXAMPLE')])
def test_ambiguous_names_rejected(monkeypatch, names):
    monkeypatch.setattr(plugins.metadata, 'entry_points', lambda **kw: [
        EntryPoint(name=n, value='unused:generate', group=plugins.ENTRY_POINT_GROUP) for n in names])
    with pytest.raises(ValueError, match='Duplicate or reserved'):
        available_approximants()


def test_bbhx_is_not_builtin(monkeypatch):
    monkeypatch.setattr(plugins.metadata, 'entry_points', lambda **kw: [])
    assert 'BBHx' not in available_approximants()
    with pytest.raises(ValueError, match='entry point'):
        get_fd_waveform('BBHx', delta_t=4., length=8)


def test_provider_import_failure_propagates(monkeypatch):
    entry = EntryPoint(name='Broken', value='absent_waveform_provider:generate', group=plugins.ENTRY_POINT_GROUP)
    monkeypatch.setattr(plugins.metadata, 'entry_points', lambda **kw: [entry])
    with pytest.raises(ModuleNotFoundError, match='absent_waveform_provider'):
        get_fd_waveform('Broken', delta_t=4., length=8)
