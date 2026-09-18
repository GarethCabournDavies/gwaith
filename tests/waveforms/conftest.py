"""Test configuration for the PhenomXPy + TDISpline waveform tests.

The ``hwilas`` package imports ``gwaith`` at import time, but these tests only
exercise :mod:`hwilas.waveforms.phenomx_tdi` with mocked dependencies. To allow
the suite to run in minimal environments (where the heavy optional stack is not
installed), a lightweight stub for ``gwaith.types.TimeSeries`` is registered if
the real package cannot be imported. When gwaith is installed this has no effect.
"""

import sys
import types


def _install_gwaith_stub():
    try:  # pragma: no cover - only reached when gwaith is missing
        import gwaith.types  # noqa: F401

        return  # real gwaith available, nothing to do
    except Exception:
        pass

    class _TimeSeries:
        def __init__(self, data=None, delta_t=None, epoch=None, name=None):
            self.data = data
            self.delta_t = delta_t
            self.epoch = epoch
            self.name = name

    gwaith_mod = types.ModuleType("gwaith")
    types_mod = types.ModuleType("gwaith.types")
    types_mod.TimeSeries = _TimeSeries
    gwaith_mod.types = types_mod
    sys.modules.setdefault("gwaith", gwaith_mod)
    sys.modules.setdefault("gwaith.types", types_mod)


_install_gwaith_stub()
