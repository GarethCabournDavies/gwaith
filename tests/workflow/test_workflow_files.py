"""Tests for configurable input_files/output_files on workflow streams."""

import unittest.mock

import pytest

from gwaith.workflow.generator import build_workflow


def _config(data):
    mock_config = unittest.mock.MagicMock()
    mock_config.data = data
    return mock_config


def _base_data(noise_cfg=None, sine_cfg=None, rounds=3):
    return {
        "workflow": {"rounds": rounds, "name": "files-test"},
        "processor": {"options": {"input-file": "/data/in.hdf"}},
        "executables": {"sine": "fit_sine", "noise": "estimate_psd"},
        "sine": {"options": {}, **(sine_cfg or {})},
        "noise": {"options": {}, **(noise_cfg or {})},
    }


def _build(data, tmp_path):
    return build_workflow(_config(data), "files-test", str(tmp_path))


def _node(workflow, uid):
    return next(node for node in workflow.nodes if node.uid == uid)


def _arg(command, flag):
    return command[command.index(flag) + 1]


def test_noise_default_output_unchanged(tmp_path):
    workflow = _build(_base_data(), tmp_path)
    noise = _node(workflow, "node_noise_r000_estimate_psd")
    assert _arg(noise.command, "--output").endswith("round_000_noise.hdf")
    assert noise.produces == ["file_noise_r000_psd"]
    assert workflow.final_noise_file == "file_noise_r002_psd"


def test_noise_extra_outputs(tmp_path):
    data = _base_data(
        noise_cfg={
            "output_files": [
                {"id": "noise_model", "arg": "output-model", "extension": "json"}
            ]
        }
    )
    workflow = _build(data, tmp_path)
    noise = _node(workflow, "node_noise_r001_estimate_psd")
    assert _arg(noise.command, "--output-model").endswith(
        "round_001_noise_noise_model.json"
    )
    assert "file_noise_r001_noise_model" in noise.produces
    # Non-signal outputs never feed the subtraction.
    residual = _node(workflow, "node_central_r001_combined_residual")
    assert "file_noise_r001_noise_model" not in residual.uses


def test_fit_consumes_previous_round_noise(tmp_path):
    data = _base_data(
        noise_cfg={
            "output_files": [
                {"id": "model", "arg": "output-model", "extension": "json"}
            ]
        },
        sine_cfg={
            "input_files": [
                {"from": "noise", "arg": "psd-file", "initial": "/data/psd0.hdf"},
                {"from": "noise:model", "arg": "noise-model"},
            ]
        },
    )
    workflow = _build(data, tmp_path)

    fit0 = _node(workflow, "node_sine_r000_fit")
    assert _arg(fit0.command, "--psd-file") == "/data/psd0.hdf"
    assert "--noise-model" not in fit0.command  # optional and not yet available

    fit2 = _node(workflow, "node_sine_r002_fit")
    assert "file_noise_r001_psd" in fit2.uses
    assert "file_noise_r001_model" in fit2.uses
    assert _arg(fit2.command, "--psd-file").endswith("round_001_noise.hdf")
    files = {f.uid: f for f in workflow.files}
    assert "node_sine_r002_fit" in files["file_noise_r001_psd"].used_by


def test_noise_consumes_same_round_signal_output(tmp_path):
    data = _base_data(
        sine_cfg={"output_files": [{"id": "glitches", "arg": "output-glitches"}]},
        noise_cfg={
            "input_files": [
                {"from": "sine:glitches", "arg": "glitch-file"},
                {"from": "noise", "arg": "previous-psd"},
                {
                    "path": "/data/mask.hdf",
                    "arg": "mask",
                    "dataset_arg": "mask-dataset",
                    "dataset": "m",
                },
            ]
        },
    )
    workflow = _build(data, tmp_path)
    noise0 = _node(workflow, "node_noise_r000_estimate_psd")
    assert "file_sine_r000_glitches" in noise0.uses
    assert "--previous-psd" not in noise0.command
    assert _arg(noise0.command, "--mask") == "/data/mask.hdf"
    assert _arg(noise0.command, "--mask-dataset") == "m"
    noise1 = _node(workflow, "node_noise_r001_estimate_psd")
    assert "file_noise_r000_psd" in noise1.uses


def test_required_input_without_initial_fails(tmp_path):
    data = _base_data(
        sine_cfg={
            "input_files": [{"from": "noise", "arg": "psd-file", "required": True}]
        }
    )
    with pytest.raises(ValueError, match="psd-file"):
        _build(data, tmp_path)


@pytest.mark.parametrize(
    "entry, message",
    [
        ({"from": "nope", "arg": "x"}, "unknown stream"),
        ({"from": "noise:nope", "arg": "x"}, "unknown output"),
        ({"from": "noise"}, "arg"),
        ({"arg": "x"}, "exactly one"),
    ],
)
def test_invalid_input_files(tmp_path, entry, message):
    data = _base_data(sine_cfg={"input_files": [entry]})
    with pytest.raises(ValueError, match=message):
        _build(data, tmp_path)


def test_global_input_files_shared_by_streams(tmp_path):
    data = _base_data(
        sine_cfg={
            "input_files": [
                {
                    "from": "inputs:orbits",
                    "arg": "orbits",
                    "dataset_arg": "orbits-dataset",
                }
            ]
        },
        noise_cfg={"input_files": [{"from": "inputs:orbits", "arg": "orbits"}]},
    )
    data["workflow"]["input_files"] = {
        "orbits": {"path": "/data/orbits.h5", "dataset": "esa"}
    }
    workflow = _build(data, tmp_path)
    for uid in ("node_sine_r000_fit", "node_sine_r002_fit"):
        fit = _node(workflow, uid)
        assert _arg(fit.command, "--orbits") == "/data/orbits.h5"
        assert _arg(fit.command, "--orbits-dataset") == "esa"
        assert "file_inputs_final_orbits" in fit.uses
    noise = _node(workflow, "node_noise_r001_estimate_psd")
    assert _arg(noise.command, "--orbits") == "/data/orbits.h5"
    files = {f.uid: f for f in workflow.files}
    assert files["file_inputs_final_orbits"].produced_by is None


@pytest.mark.parametrize(
    "entry, message",
    [
        ({"from": "inputs:nope", "arg": "x"}, "unknown output"),
        ({"from": "inputs", "arg": "x"}, "must name a file"),
    ],
)
def test_invalid_global_input_reference(tmp_path, entry, message):
    data = _base_data(sine_cfg={"input_files": [entry]})
    data["workflow"]["input_files"] = {"orbits": "/data/orbits.h5"}
    with pytest.raises(ValueError, match=message):
        _build(data, tmp_path)
