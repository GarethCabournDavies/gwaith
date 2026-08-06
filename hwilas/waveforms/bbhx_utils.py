"""Utilities for interfacing with BBHx."""

from typing import Any

import numpy as np

from .bbhx_conversion import _core_transforms, convert_frame_parameters

ARGS_KEYS = [
    "m1",
    "m2",
    "chi1z",
    "chi2z",
    "distance",
    "phi_ref",
    "f_ref",
    "inc",
    "lam",
    "beta",
    "psi",
    "t_ref",
]

KWARGS_KEYS = ["freqs", "modes", "direct", "fill", "squeeze", "length"]
BBHX_FRAME = "SSB"


def get_bbhx_parameter_dict(
    parameters: dict[str, Any],
    *,
    parameter_frame: str,
    t0: float = 0.0,
) -> dict[str, float]:
    """Return BBHx-ready parameters in the BBHx evaluation frame."""
    converted = convert_frame_parameters(
        parameters,
        source_frame=parameter_frame,
        target_frame=BBHX_FRAME,
        t0=t0,
    )
    missing = [name for name in ARGS_KEYS if name not in converted]
    if missing:
        msg = (
            "Cannot build BBHx parameters because required values are missing: "
            f"{missing}."
        )
        raise ValueError(msg)
    return {name: converted[name] for name in ARGS_KEYS}


def transform_to_bbhx_parameters(
    x: np.ndarray,
    input_parameters: list[str],
    injection_params: dict[str, float] | None = None,
):
    """Generate the parameters for the BBHX model from the input x.

    BBHx expects: m1, m2, chi1z, chi2z, distance, phi_ref, f_ref, inc, lam, beta, psi, t_ref
    """
    parameters = _core_transforms(
        x,
        input_parameters,
        injection_params,
    )
    keys_to_remove = set(parameters.keys()) - set(ARGS_KEYS)
    for key in keys_to_remove:
        del parameters[key]
    return parameters


def build_bbhx_parameter_array(injection_params: dict[str, float]) -> np.ndarray:
    """Return parameters ordered as expected by BBHX."""
    return np.array([injection_params[param] for param in ARGS_KEYS], dtype=float)
