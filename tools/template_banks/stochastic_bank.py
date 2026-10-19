"""Generate a stochastic FastBHB bank from TOML parameter priors."""
import argparse
import json
from hwilas import add_hwilas_common_options, init_logging
from hwilas.search.config import load_config
from hwilas.template_banks.stochastic import build
from hwilas.waveforms.cli import add_waveform_arguments, apply_waveform_config


def main():
    p = argparse.ArgumentParser(description=__doc__)
    add_hwilas_common_options(p)
    p.add_argument("--config", required=True)
    p.add_argument("--nproc", type=int, default=4)
    p.add_argument("--seed", type=int, help="Override the proposal seed in the config")
    add_waveform_arguments(p)
    p.set_defaults(waveform_type=None,orbit=None)
    a = p.parse_args()
    init_logging(a.verbose)
    cfg = apply_waveform_config(a, load_config(a.config))
    if a.seed is not None:
        cfg["bank"]["seed"] = a.seed
    print(json.dumps(build(cfg, a.nproc), indent=2))


if __name__ == "__main__":
    main()
