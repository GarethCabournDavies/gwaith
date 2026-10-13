"""Network SNR retention sets per-template frequency and time support."""
import numpy as np
import pytest
from scipy.integrate import cumulative_trapezoid
from hwilas.template_banks.frequency import lower_frequency_for_snr, duration_from_frequency
from hwilas.template_banks.io import write_bank, load_bank


def test_cutoff_retains_snr_not_snr_squared():
    f = np.linspace(.001, .1, 10001)
    density = np.ones(len(f))
    cutoff = lower_frequency_for_snr(f, density, .99)
    remaining = (.1-cutoff)/(.1-.001)
    assert .99 <= np.sqrt(remaining) < .9901
    assert remaining < .99


def test_each_template_has_its_own_psd_weighted_cutoff():
    f = np.geomspace(1e-5, .1, 32768)
    cutoffs = []
    for centre in (.001, .01):
        density = np.exp(-((np.log(f/centre))/.5)**2)
        cutoff = lower_frequency_for_snr(f, density)
        cumulative = cumulative_trapezoid(density, f, initial=0.)
        discarded = cumulative[np.searchsorted(f, cutoff)]/cumulative[-1]
        assert np.sqrt(1-discarded) >= .99
        cutoffs.append(cutoff)
    assert cutoffs[1] > 9*cutoffs[0]


@pytest.mark.parametrize('density', [np.zeros(3), np.array([1., -1., 1.]), np.array([1., np.nan, 1.])])
def test_invalid_snr_density(density):
    with pytest.raises(ValueError):
        lower_frequency_for_snr(np.array([1., 2., 3.]), density)


def test_cutoff_duration_and_bank_roundtrip(tmp_path):
    from tests.template_banks.test_start_frequency import params, SETTINGS
    p = params(1e6, 5e5)
    p['f_lower'] = .0002
    p['pre_merger_seconds'] = duration_from_frequency(p, p['f_lower'])
    assert p['pre_merger_seconds'] > 0
    path = tmp_path/'bank.hdf'
    write_bank(path, [p], SETTINGS)
    restored = load_bank(path)[0]
    assert restored['f_lower'] == p['f_lower']
    assert restored['pre_merger_seconds'] == p['pre_merger_seconds']


def test_generator_uses_full_response_and_saves_cutoff(tmp_path, monkeypatch):
    from hwilas.template_banks import stochastic as sb
    from hwilas.search.config import load_config
    from pathlib import Path
    cfg = load_config(Path(__file__).resolve().parents[2]/'examples/template_banks/mojito_bank.toml')
    path = tmp_path/'psd.txt'
    np.savetxt(path, [[1e-5, 1.], [.125, 1.]])
    cfg['bank'].update(psd_A=str(path), psd_E=str(path), psd_units='Hz')
    cfg['waveform']['laser_frequency'] = 1.
    sb.init_worker(cfg)
    calls = []
    def waveform(params, f, settings, modes):
        calls.append((f.copy(), settings.copy()))
        amplitude = np.exp(-((np.log(f/.002))/.5)**2)
        return {c: amplitude.astype(complex) for c in sb.CHANNELS}
    monkeypatch.setattr(sb, 'frequency_template', waveform)
    monkeypatch.setattr(sb, 'short_template', lambda *a, **k: pytest.fail('SNR cutoff must not use truncated templates'))
    from tests.template_banks.test_start_frequency import params
    p, W, sig, error = sb.generate(params(1e6, 5e5))
    assert error is None
    assert len(calls) == 2
    assert calls[0][0][0] == cfg['waveform']['f_lower']
    assert p['f_lower'] > cfg['waveform']['f_lower']
    assert p['pre_merger_seconds'] == duration_from_frequency(p, p['f_lower'])
    bank = sb.Bank(cfg)
    bank.add(p, W, sig)
    assert bank.best_match(p, W, sig) == pytest.approx(1., abs=1e-6)
