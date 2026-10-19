"""Labelled PSD inputs can include independent unsmoothed channels."""
import numpy as np
from matplotlib.axes import Axes
from tools.plotting.plot_psd import main


def test_five_labelled_psds(tmp_path, monkeypatch):
    path = tmp_path / 'psd.txt'
    np.savetxt(path, [[0., 0.], [1e-5, 2.], [.01, 1.], [.1, .5]])
    output = tmp_path / 'psds.png'
    labels = []
    limits = {}
    original_xlim = Axes.set_xlim
    original_ylim = Axes.set_ylim

    def record_xlim(self, *args, **kwargs):
        if len(args) == 2:
            limits["x"] = args
        return original_xlim(self, *args, **kwargs)

    def record_ylim(self, *args, **kwargs):
        if len(args) == 2:
            limits["y"] = args
        return original_ylim(self, *args, **kwargs)

    monkeypatch.setattr(Axes, "set_xlim", record_xlim)
    monkeypatch.setattr(Axes, "set_ylim", record_ylim)
    original_plot = Axes.plot

    def record_plot(self, *args, **kwargs):
        labels.append(kwargs['label'])
        return original_plot(self, *args, **kwargs)

    monkeypatch.setattr(Axes, 'plot', record_plot)
    monkeypatch.setattr('sys.argv', ['plot_psd', '--psd-files',
                        f'A unsmoothed:{path}', f'E unsmoothed:{path}',
                        f'T unsmoothed:{path}', f'A smoothed:{path}',
                        f'E smoothed:{path}', '--f-min', '1e-5', '--f-max', '0.1',
                        '--y-min', '1e-47', '--y-max', '1e-36',
                        '--output-file', str(output)])
    main()
    assert labels == ['A unsmoothed', 'E unsmoothed', 'T unsmoothed',
                      'A smoothed', 'E smoothed']
    assert limits == {'x': (1e-5, .1), 'y': (1e-47, 1e-36)}
    assert output.read_bytes().startswith(b'\x89PNG\r\n\x1a\n')
