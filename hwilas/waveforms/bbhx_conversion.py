"""Coordinate and parameter-conversion helpers."""

from typing import Any, Literal

import numpy as np
from bbhx.utils.transform import LISA_to_SSB, SSB_to_LISA

FRAME_DEPENDENT_PARAMETERS = ("t_ref", "lam", "beta", "psi")


def normalize_frame_name(
    frame: str,
    *,
    field_name: str = "frame",
) -> Literal["SSB", "LISA"]:
    """Normalize and validate a frame name."""
    normalized = str(frame).strip().upper()
    if normalized not in {"SSB", "LISA"}:
        valid = "LISA, SSB"
        msg = f"Invalid {field_name} '{frame}'. Valid options: {valid}."
        raise ValueError(msg)
    return normalized


def convert_frame_parameters(
    parameters: dict[str, Any],
    *,
    source_frame: str,
    target_frame: str,
    t0: float = 0.0,
) -> dict[str, Any]:
    """Convert frame-dependent parameters between SSB and LISA."""
    normalized_source = normalize_frame_name(source_frame, field_name="source frame")
    normalized_target = normalize_frame_name(target_frame, field_name="target frame")
    converted = dict(parameters)

    if normalized_source == normalized_target:
        return converted

    available = [name for name in FRAME_DEPENDENT_PARAMETERS if name in converted]
    if not available:
        return converted
    if len(available) != len(FRAME_DEPENDENT_PARAMETERS):
        missing = sorted(set(FRAME_DEPENDENT_PARAMETERS) - set(available))
        msg = (
            "Frame conversion requires all of "
            f"{FRAME_DEPENDENT_PARAMETERS}. Missing: {missing}."
        )
        raise ValueError(msg)

    if normalized_source == "LISA" and normalized_target == "SSB":
        t_ref, lam, beta, psi = LISA_to_SSB(
            converted["t_ref"],
            converted["lam"],
            converted["beta"],
            converted["psi"],
            t0=t0,
        )
    elif normalized_source == "SSB" and normalized_target == "LISA":
        t_ref, lam, beta, psi = SSB_to_LISA(
            converted["t_ref"],
            converted["lam"],
            converted["beta"],
            converted["psi"],
            t0=t0,
        )
    else:
        msg = (
            f"Unsupported frame conversion {normalized_source} -> {normalized_target}."
        )
        raise ValueError(msg)

    converted["t_ref"] = t_ref
    converted["lam"] = np.mod(lam, 2 * np.pi)
    converted["beta"] = beta
    converted["psi"] = np.mod(psi, np.pi)
    if "sin_beta" in converted:
        converted["sin_beta"] = np.sin(beta)
    return converted


def convert_to_ssb_frame(parameters: dict[str, float], t=0.0) -> dict[str, float]:
    """Convert time, sky location, and polarization parameters to the SSB frame."""
    return convert_frame_parameters(
        parameters,
        source_frame="LISA",
        target_frame="SSB",
        t0=t,
    )


def convert_to_lisa_frame(parameters: dict[str, float], t=0.0) -> dict[str, float]:
    """Convert time, sky location, and polarization parameters to the LISA frame."""
    return convert_frame_parameters(
        parameters,
        source_frame="SSB",
        target_frame="LISA",
        t0=t,
    )


def _core_transforms(
    x: np.ndarray,
    input_parameters: list[str],
    injection_params: dict[str, float] | None = None,
) -> dict[str, float]:
    """Core transforms for converting from sampling parameters to source parameters."""
    output_parameters = dict(zip(input_parameters, x.T, strict=True))
    if "M" in output_parameters and "q" in output_parameters:
        total_mass = output_parameters["M"]
        q = output_parameters["q"]
        output_parameters["m1"] = total_mass / (1 + q)
        output_parameters["m2"] = total_mass * q / (1 + q)

    if "mchirp" in output_parameters and "q" in output_parameters:
        mchirp = output_parameters["mchirp"]
        q = output_parameters["q"]
        output_parameters["m1"] = mchirp * (1 + q) ** (1 / 5) / q ** (3 / 5)
        output_parameters["m2"] = mchirp * q ** (2 / 5) * (1 + q) ** (1 / 5)

    if "delta_phi" in output_parameters:
        # phase = delta_phi - sign(cos inc) * psi
        if "cos_inc" in output_parameters:
            cos_inc = output_parameters["cos_inc"]
        elif "inc" in output_parameters:
            cos_inc = np.cos(output_parameters["inc"])
        elif injection_params is not None:
            if "inc" in injection_params:
                cos_inc = np.cos(injection_params["inc"])
            elif "cos_inc" in injection_params:
                cos_inc = injection_params["cos_inc"]
            else:
                msg = "Cannot determine cos_inc for phase transformation."
                raise ValueError(msg)
        else:
            msg = "Cannot determine cos_inc for phase transformation."
            raise ValueError(msg)
        if "psi" in output_parameters:
            psi = output_parameters["psi"]
        elif injection_params is not None and "psi" in injection_params:
            psi = injection_params["psi"]
        else:
            msg = "Cannot determine psi for phase transformation."
            raise ValueError(msg)
        output_parameters["phi_ref"] = np.mod(
            output_parameters["delta_phi"] - np.sign(cos_inc) * psi, 2 * np.pi
        )

    if "sin_beta" in output_parameters:
        output_parameters["beta"] = np.arcsin(output_parameters["sin_beta"])

    if "cos_inc" in output_parameters:
        output_parameters["inc"] = np.arccos(output_parameters["cos_inc"])
    return output_parameters


def generate_all_parameters(
    x: np.ndarray,
    input_parameters: list[str],
    injection_params: dict[str, float] | None = None,
):
    """Generate all known parameters from the input x.

    This includes parameters that are not used by BBHx
    """
    output_parameters = _core_transforms(
        x,
        input_parameters,
        injection_params,
    )

    if "m1" in output_parameters and "m2" in output_parameters:
        output_parameters["M"] = output_parameters["m1"] + output_parameters["m2"]
        output_parameters["q"] = output_parameters["m2"] / output_parameters["m1"]
        output_parameters["mchirp"] = (
            output_parameters["m1"] * output_parameters["m2"]
        ) ** (3 / 5) / (output_parameters["m1"] + output_parameters["m2"]) ** (1 / 5)

    if "beta" in output_parameters:
        output_parameters["sin_beta"] = np.sin(output_parameters["beta"])

    if "inc" in output_parameters:
        output_parameters["cos_inc"] = np.cos(output_parameters["inc"])

    if "phi_ref" in output_parameters:
        if "psi" in output_parameters:
            psi = output_parameters["psi"]
        elif injection_params is not None and "psi" in injection_params:
            psi = injection_params["psi"]
        else:
            msg = "Cannot determine psi for phase transformation."
            raise ValueError(msg)
        if "cos_inc" in output_parameters:
            cos_inc = output_parameters["cos_inc"]
        elif "inc" in output_parameters:
            cos_inc = np.cos(output_parameters["inc"])
        elif injection_params is not None:
            if "inclination" in injection_params:
                cos_inc = np.cos(injection_params["inc"])
            elif "cos_inc" in injection_params:
                cos_inc = injection_params["cos_inc"]
            else:
                msg = "Cannot determine cos_inc for phase transformation."
                raise ValueError(msg)
        else:
            msg = "Cannot determine cos_inc for phase transformation."
            raise ValueError(msg)
        output_parameters["delta_phi"] = np.mod(
            output_parameters["phi_ref"] + np.sign(cos_inc) * psi, 2 * np.pi
        )

    return output_parameters
