"""Workflow graph construction and output generation."""

from __future__ import annotations

import re
import shlex
from pathlib import Path
from typing import Any

from .helpers import make_file_uid, make_node_uid, register_file, register_node
from .types import GwaithFile, GwaithNode, GwaithStream, GwaithWorkflow

_GLOBAL_SECTIONS = {"workflow", "executables", "default_resources"}
_INFRA_SECTIONS = {"processor", "subtraction", "noise"}
_ROUND_TAG_TEMPLATE = "round_{round_index:03d}"
_DEFAULT_DATASET = "data"
_DEFAULT_SIGNAL_DATASET = "signal"
_DEFAULT_PSD_DATASET = "psd"


def _safe_name(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9]+", "_", str(value)).strip("_") or "item"


def _round_tag(round_index: int) -> str:
    return _ROUND_TAG_TEMPLATE.format(round_index=int(round_index))


def _as_path(value: str | Path) -> str:
    return str(Path(value).expanduser())


def _script_header() -> str:
    return "#!/usr/bin/env bash\nset -euo pipefail\n\n"


def _write_executable(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")
    path.chmod(path.stat().st_mode | 0o111)


def _render_command(command: list[str]) -> str:
    return shlex.join([str(part) for part in command]) + "\n"


def _normalise_stream_decimation(stream_cfg: dict[str, Any]) -> int:
    value = stream_cfg.get("stream_decimation", stream_cfg.get("stream-decimation", 1))
    decimation = int(value)
    if decimation < 1:
        raise ValueError("stream_decimation must be >= 1")
    return decimation


def _normalise_stream_offset(stream_cfg: dict[str, Any]) -> int:
    return int(
        stream_cfg.get(
            "stream_decimation_offset",
            stream_cfg.get("stream-decimation-offset", 0),
        )
    )


def _normalise_skip_rounds(stream_cfg: dict[str, Any]) -> set[int]:
    raw = stream_cfg.get("skip_rounds", stream_cfg.get("skip-rounds", []))
    if raw is None:
        return set()
    if not isinstance(raw, list):
        raise ValueError("skip_rounds must be a list")
    return {int(value) for value in raw}


def _default_signal_output(options: dict[str, Any]) -> dict[str, Any]:
    """Return the primary output of a signal stream (--output-signal)."""
    return {
        "id": "signal",
        "arg": "output-signal",
        "dataset_arg": "output-dataset",
        "dataset": str(options.get("output-dataset", _DEFAULT_SIGNAL_DATASET)),
        "kind": "signal",
        "extension": "hdf",
        "include_in_subtraction": True,
    }


def _default_noise_output(options: dict[str, Any]) -> dict[str, Any]:
    """Return the primary output of the noise stream (--output)."""
    return {
        "id": "psd",
        "arg": "output",
        # output-dataset is passed through the noise options unchanged.
        "dataset_arg": None,
        "dataset": str(options.get("output-dataset") or _DEFAULT_PSD_DATASET),
        "kind": "psd",
        "extension": "hdf",
        "include_in_subtraction": False,
    }


def _normalise_stream_outputs(
    stream_cfg: dict[str, Any],
    options: dict[str, Any],
    primary: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    """Return normalised output artifact definitions for a stream.

    The first entry is the stream's primary output (by default the signal
    artifact written via --output-signal/--output-dataset). Additional
    outputs may be provided under output_files/output-files, each with:

    ``id``
        Name of the output, used in file names and by ``input_files``.
    ``arg``
        CLI flag (without ``--``) used to pass the output path.
    ``dataset_arg`` / ``dataset``
        Optional flag and value naming the dataset inside the file.
    ``kind``
        Logical role (default ``artifact``).
    ``extension``
        File extension (default ``hdf``).
    ``include_in_subtraction``
        Whether the file is subtracted from the data (default: kind == signal).
    """
    if primary is None:
        primary = _default_signal_output(options)
    outputs: list[dict[str, Any]] = [primary]

    raw_outputs = stream_cfg.get("output_files", stream_cfg.get("output-files", []))
    if raw_outputs is None:
        raw_outputs = []
    if not isinstance(raw_outputs, list):
        raise ValueError("output_files must be a list")

    for index, raw_output in enumerate(raw_outputs):
        if not isinstance(raw_output, dict):
            raise ValueError("Each output_files entry must be a mapping")

        output_id = _safe_name(
            raw_output.get("id") or raw_output.get("name") or f"output_{index}"
        )
        if output_id == primary["id"]:
            # The primary output is always present and reserved.
            continue
        if any(existing["id"] == output_id for existing in outputs):
            raise ValueError(f"Duplicate output_files id: {output_id}")

        arg_name = str(
            raw_output.get("arg")
            or raw_output.get("output-arg")
            or f"output-{output_id}"
        )
        arg_name = arg_name.lstrip("-")

        dataset_arg = raw_output.get(
            "dataset_arg", raw_output.get("output-dataset-arg")
        )
        dataset_value = raw_output.get("dataset")
        kind = str(raw_output.get("kind", "artifact"))
        extension = str(raw_output.get("extension", "hdf")).lstrip(".") or "hdf"
        include_in_subtraction = bool(
            raw_output.get(
                "include_in_subtraction",
                raw_output.get("include-in-subtraction", kind == "signal"),
            )
        )

        outputs.append(
            {
                "id": output_id,
                "arg": arg_name,
                "dataset_arg": str(dataset_arg) if dataset_arg else None,
                "dataset": str(dataset_value) if dataset_value is not None else None,
                "kind": kind,
                "extension": extension,
                "include_in_subtraction": include_in_subtraction,
            }
        )

    return outputs


def _normalise_stream_inputs(stream_cfg: dict[str, Any]) -> list[dict[str, Any]]:
    """Return normalised input file definitions for a stream.

    Inputs are given under input_files/input-files. Each entry is a mapping:

    ``from``
        ``<stream>`` or ``<stream>:<output_id>``. The node receives the most
        recent file with that output id produced by the named stream (e.g.
        ``noise:psd``). If ``output_id`` is omitted the stream's primary
        output is used.
    ``path``
        A fixed file used every round, instead of ``from``.
    ``arg``
        CLI flag (without ``--``) used to pass the file path. Required.
    ``dataset_arg`` / ``dataset``
        Optional flag and value naming the dataset inside the file.
    ``initial``
        Fixed file used until ``from`` has produced anything.
    ``required``
        If true, fail when no file is available for a node (default false;
        the argument is otherwise omitted for that node).
    """
    raw_inputs = stream_cfg.get("input_files", stream_cfg.get("input-files", []))
    if raw_inputs is None:
        raw_inputs = []
    if not isinstance(raw_inputs, list):
        raise ValueError("input_files must be a list")

    inputs: list[dict[str, Any]] = []
    for raw_input in raw_inputs:
        if not isinstance(raw_input, dict):
            raise ValueError("Each input_files entry must be a mapping")
        arg_name = raw_input.get("arg") or raw_input.get("input-arg")
        if not arg_name:
            raise ValueError("Each input_files entry must define 'arg'")
        source = raw_input.get("from")
        path = raw_input.get("path")
        if bool(source) == bool(path):
            raise ValueError(
                f"input_files entry '{arg_name}' must define exactly one of "
                "'from' or 'path'"
            )
        source_stream = source_output = None
        if source:
            source_stream, _, source_output = str(source).partition(":")
            source_stream = _safe_name(source_stream)
            source_output = _safe_name(source_output) if source_output else None
        dataset_arg = raw_input.get("dataset_arg", raw_input.get("dataset-arg"))
        dataset_value = raw_input.get("dataset")
        initial = raw_input.get("initial")
        inputs.append(
            {
                "arg": str(arg_name).lstrip("-"),
                "source_stream": source_stream,
                "source_output": source_output,
                "path": _as_path(path) if path else None,
                "initial": _as_path(initial) if initial else None,
                "dataset_arg": str(dataset_arg).lstrip("-") if dataset_arg else None,
                "dataset": str(dataset_value) if dataset_value is not None else None,
                "required": bool(raw_input.get("required", False)),
            }
        )
    return inputs


_GLOBAL_INPUT_STREAM = "inputs"


def _normalise_global_inputs(workflow_cfg: dict[str, Any]) -> dict[str, dict[str, Any]]:
    """Return named analysis-level input files from workflow.input_files.

    Accepts a mapping of ``name: path`` or ``name: {path, dataset}``. Streams
    reference these with ``input_files: [{from: inputs:<name>, arg: ...}]``.
    """
    raw = workflow_cfg.get("input_files", workflow_cfg.get("input-files", {}))
    if raw is None:
        return {}
    if not isinstance(raw, dict):
        raise ValueError("workflow.input_files must be a mapping of name to path")
    inputs: dict[str, dict[str, Any]] = {}
    for raw_name, value in raw.items():
        name = _safe_name(raw_name)
        if isinstance(value, dict):
            path = value.get("path")
            dataset = value.get("dataset")
        else:
            path, dataset = value, None
        if not path:
            raise ValueError(f"workflow.input_files '{raw_name}' needs a path")
        inputs[name] = {
            "path": _as_path(path),
            "dataset": str(dataset) if dataset is not None else None,
        }
    return inputs


def _stream_runs_on_round(stream_cfg: dict[str, Any], round_index: int) -> bool:
    if round_index in stream_cfg["skip_rounds"]:
        return False
    return (round_index - stream_cfg["decimation_offset"]) % stream_cfg[
        "decimation"
    ] == 0


def _is_hdf5_file(gwaith_file: GwaithFile) -> bool:
    """Return True if the file is an HDF5 container (by path extension)."""
    local_path = gwaith_file.local_path
    if not local_path:
        return False
    return local_path.endswith(".hdf") or local_path.endswith(".hdf5")


def _data_file_args(input_file: GwaithFile) -> list[str]:
    """Return the CLI args to point a tool at the input file's data.

    The flag names depend on the file type: HDF5 files are addressed with
    ``--data```, while JSON timeseries files (produced by
    the subtraction step) are addressed with ``--timeseries-file``/
    ``--timeseries-name``. Tools that accept both sets of options resolve to a
    single input via ``_resolve_input_file``, so emitting only the relevant
    pair keeps the command unambiguous.
    """
    if not input_file.local_path:
        return []
    if _is_hdf5_file(input_file):
        return [
            "--data",
            input_file.local_path,
        ]
    return ["--timeseries-file", input_file.local_path]


def _output_file_args(output_cfg: dict[str, Any], output_path: str) -> list[str]:
    """Return the CLI args used to pass an output file to a tool."""
    args = [f"--{output_cfg['arg']}", output_path]
    dataset_arg = output_cfg.get("dataset_arg")
    dataset_value = output_cfg.get("dataset")
    if dataset_arg and dataset_value:
        args.extend([f"--{dataset_arg}", dataset_value])
    return args


def _latest_output_uid(
    outputs_by_round: dict[int, dict[str, str]],
    output_id: str,
    max_round_inclusive: int,
) -> str | None:
    """Return the most recent file uid for ``output_id`` up to a round."""
    candidates = [
        round_index
        for round_index, outputs in outputs_by_round.items()
        if round_index <= max_round_inclusive and output_id in outputs
    ]
    if not candidates:
        return None
    return outputs_by_round[max(candidates)][output_id]


def _build_cli_args(
    options: dict[str, Any], excluded: set[str] | None = None
) -> list[str]:
    excluded = excluded or set()
    args: list[str] = []
    for key, value in options.items():
        if key in excluded:
            continue
        flag = f"--{key}"
        if isinstance(value, bool):
            if value:
                args.append(flag)
            continue
        if value == "" or value is None:
            args.append(flag)
            continue
        if isinstance(value, list):
            for item in value:
                args.extend([flag, str(item)])
            continue
        args.extend([flag, str(value)])
    return args


def _serialise_file(gwaith_file: GwaithFile) -> dict[str, Any]:
    return {
        "uid": gwaith_file.uid,
        "path": gwaith_file.local_path,
        "final_path": gwaith_file.final_path,
        "dataset": gwaith_file.dataset,
        "kind": gwaith_file.kind,
        "stream": gwaith_file.stream,
        "round": gwaith_file.round_index,
        "produced_by": gwaith_file.produced_by,
        "used_by": list(gwaith_file.used_by),
        "retention_level": gwaith_file.retention_level,
    }


def _serialise_node(gwaith_node: GwaithNode) -> dict[str, Any]:
    return {
        "uid": gwaith_node.uid,
        "name": gwaith_node.name,
        "stream": gwaith_node.stream,
        "round": gwaith_node.round_index,
        "command": list(gwaith_node.command),
        "uses": list(gwaith_node.uses),
        "produces": list(gwaith_node.produces),
        "resources": dict(gwaith_node.resources),
    }


def _serialise_stream(gwaith_stream: GwaithStream) -> dict[str, Any]:
    return {
        "uid": gwaith_stream.uid,
        "name": gwaith_stream.name,
        "environment": gwaith_stream.environment,
        "decimation": gwaith_stream.decimation,
        "skip_rounds": list(gwaith_stream.skip_rounds),
        "nodes": list(gwaith_stream.nodes),
    }


def _parse_config(
    config_parser, workflow_name: str | None, workflow_dir: str | None
) -> dict[str, Any]:
    data = config_parser.data
    workflow_cfg = data.get("workflow", {})
    if not isinstance(workflow_cfg, dict):
        raise ValueError("workflow section must be a mapping")

    processor_cfg = data.get("processor", {})
    if not isinstance(processor_cfg, dict):
        raise ValueError("processor section must be a mapping")
    processor_options = processor_cfg.get("options", {}) or {}

    input_file = (
        processor_options.get("input-file")
        or processor_options.get("data")
        or workflow_cfg.get("input-file")
        or workflow_cfg.get("data")
    )
    input_dataset = str(processor_options.get("input-dataset") or _DEFAULT_DATASET)

    executables = data.get("executables", {})
    if not isinstance(executables, dict):
        raise ValueError("executables section must be a mapping")

    rounds = int(workflow_cfg.get("rounds", 1))
    if rounds < 1:
        raise ValueError("workflow.rounds must be >= 1")

    name = str(workflow_name or workflow_cfg.get("name") or "gwaith_workflow")
    target_dir = (
        Path(workflow_dir or workflow_cfg.get("output_dir") or name)
        .expanduser()
        .resolve()
    )

    subtraction_cfg = (
        data.get("subtraction", {})
        if isinstance(data.get("subtraction", {}), dict)
        else {}
    )
    noise_cfg = data.get("noise", {}) if isinstance(data.get("noise", {}), dict) else {}
    noise_options = dict(noise_cfg.get("options", {}) or {})

    signal_streams: list[dict[str, Any]] = []
    for section_name, section_cfg in data.items():
        if section_name in _GLOBAL_SECTIONS or section_name in _INFRA_SECTIONS:
            continue
        if not isinstance(section_cfg, dict):
            continue
        if section_name not in executables:
            continue
        options = dict(section_cfg.get("options", {}) or {})
        signal_streams.append(
            {
                "uid": _safe_name(section_name),
                "section_name": section_name,
                "name": section_name,
                "executable": str(executables[section_name]),
                "decimation": _normalise_stream_decimation(section_cfg),
                "decimation_offset": _normalise_stream_offset(section_cfg),
                "skip_rounds": _normalise_skip_rounds(section_cfg),
                "resources": dict(section_cfg.get("resources", {}) or {}),
                "options": options,
                "outputs": _normalise_stream_outputs(section_cfg, options),
                "inputs": _normalise_stream_inputs(section_cfg),
                "initial_prior": options.get("prior-file")
                or options.get("priors")
                or options.get("init-params"),
            }
        )

    if any(stream["uid"] == _GLOBAL_INPUT_STREAM for stream in signal_streams):
        raise ValueError(f"'{_GLOBAL_INPUT_STREAM}' is reserved and cannot be a stream")
    if not signal_streams:
        raise ValueError(
            "At least one signal stream must be defined in the parsed config"
        )

    return {
        "name": name,
        "workflow_dir": target_dir,
        "rounds": rounds,
        "input_file": _as_path(input_file) if input_file else None,
        "input_dataset": input_dataset,
        "global_retention_level": workflow_cfg.get("global-retention-level"),
        "global_inputs": _normalise_global_inputs(workflow_cfg),
        "signal_streams": signal_streams,
        "subtraction": {
            "executable": str(
                executables.get("subtraction", "gwaith_subtract_signals")
            ),
            "options": dict(subtraction_cfg.get("options", {}) or {}),
            "resources": dict(subtraction_cfg.get("resources", {}) or {}),
        },
        "noise": {
            "enabled": "noise" in executables,
            "executable": str(executables.get("noise", "gwaith_estimate_psd")),
            "options": noise_options,
            "resources": dict(noise_cfg.get("resources", {}) or {}),
            "outputs": _normalise_stream_outputs(
                noise_cfg, noise_options, primary=_default_noise_output(noise_options)
            ),
            "inputs": _normalise_stream_inputs(noise_cfg),
        },
    }


def build_workflow_graph(
    config_parser, workflow_name: str | None = None, workflow_dir: str | None = None
) -> GwaithWorkflow:
    cfg = _parse_config(
        config_parser, workflow_name=workflow_name, workflow_dir=workflow_dir
    )
    target_dir: Path = cfg["workflow_dir"]

    files: dict[str, GwaithFile] = {}
    nodes: dict[str, GwaithNode] = {}
    streams: dict[str, GwaithStream] = {}

    original_data_uid = make_file_uid("processor", None, "input_data")
    original_data_file = GwaithFile(
        uid=original_data_uid,
        local_path=cfg["input_file"],
        final_path=cfg["input_file"],
        dataset=cfg["input_dataset"],
        kind="input_data",
        round_index=None,
        stream_uid="processor",
        retention_level=cfg["global_retention_level"],
    )
    register_file(files, original_data_file)

    streams["central"] = GwaithStream(uid="central", environment="bash", name="central")
    if cfg["noise"]["enabled"]:
        streams["noise"] = GwaithStream(uid="noise", environment="bash", name="noise")

    initial_prior_by_stream: dict[str, str | None] = {}
    subtractable_output_uids_by_stream_round: dict[str, dict[int, list[str]]] = {}
    prior_uid_by_stream_round: dict[str, dict[int, str]] = {}
    next_data_by_stream: dict[str, str] = {}

    def _latest_produced_round(stream_uid: str, max_round_inclusive: int) -> int | None:
        """Find the latest round where the stream produced output (ran on that round)."""
        if max_round_inclusive < 0:
            return None
        stream = streams[stream_uid]
        # Compute the latest round that matches the decimation pattern
        candidate = (
            (max_round_inclusive - stream.decimation_offset) // stream.decimation
        ) * stream.decimation + stream.decimation_offset
        # Iterate backwards from candidate, checking skip_rounds. This accounts for skipped rounds.
        for round_index in range(candidate, -1, -stream.decimation):
            if round_index in stream.skip_rounds:
                continue
            return round_index
        return None

    for stream_cfg in cfg["signal_streams"]:
        stream_uid = stream_cfg["uid"]
        streams[stream_uid] = GwaithStream(
            uid=stream_uid,
            environment="bash",
            name=stream_cfg["name"],
            decimation=stream_cfg["decimation"],
            decimation_offset=stream_cfg["decimation_offset"],
            skip_rounds=sorted(stream_cfg["skip_rounds"]),
        )
        subtractable_output_uids_by_stream_round[stream_uid] = {}
        prior_uid_by_stream_round[stream_uid] = {}
        next_data_by_stream[stream_uid] = original_data_uid
        initial_prior_by_stream[stream_uid] = None

        initial_prior = stream_cfg.get("initial_prior")
        if initial_prior:
            prior_uid = make_file_uid(stream_uid, None, "initial_prior")
            prior_file = GwaithFile(
                uid=prior_uid,
                local_path=_as_path(initial_prior),
                final_path=_as_path(initial_prior),
                dataset=None,
                kind="initial_prior",
                round_index=None,
                stream_uid=stream_uid,
                retention_level=cfg["global_retention_level"],
            )
            register_file(files, prior_file)
            initial_prior_by_stream[stream_uid] = prior_uid

    # Every output produced by every stream, keyed by stream, round and output id
    outputs_by_stream_round: dict[str, dict[int, dict[str, str]]] = {
        stream_uid: {} for stream_uid in streams
    }
    primary_output_by_stream: dict[str, str] = {
        stream_cfg["uid"]: stream_cfg["outputs"][0]["id"]
        for stream_cfg in cfg["signal_streams"]
    }
    output_ids_by_stream: dict[str, set[str]] = {
        stream_cfg["uid"]: {o["id"] for o in stream_cfg["outputs"]}
        for stream_cfg in cfg["signal_streams"]
    }
    global_input_uids: dict[str, str] = {}
    for name, global_input in cfg["global_inputs"].items():
        uid = make_file_uid(_GLOBAL_INPUT_STREAM, None, name)
        register_file(
            files,
            GwaithFile(
                uid=uid,
                local_path=global_input["path"],
                final_path=global_input["path"],
                dataset=global_input["dataset"],
                kind="input_file",
                round_index=None,
                stream_uid=_GLOBAL_INPUT_STREAM,
                retention_level=cfg["global_retention_level"],
            ),
        )
        global_input_uids[name] = uid
    output_ids_by_stream[_GLOBAL_INPUT_STREAM] = set(global_input_uids)

    if cfg["noise"]["enabled"]:
        primary_output_by_stream["noise"] = cfg["noise"]["outputs"][0]["id"]
        output_ids_by_stream["noise"] = {o["id"] for o in cfg["noise"]["outputs"]}

    consumers = [(s_cfg["uid"], s_cfg["inputs"]) for s_cfg in cfg["signal_streams"]]
    if cfg["noise"]["enabled"]:
        consumers.append(("noise", cfg["noise"]["inputs"]))
    for consumer_uid, inputs in consumers:
        for input_cfg in inputs:
            source = input_cfg["source_stream"]
            if source is None:
                continue
            if source not in output_ids_by_stream:
                raise ValueError(
                    f"input_files for {consumer_uid} references unknown stream "
                    f"'{source}'"
                )
            if input_cfg["source_output"] is None:
                if source == _GLOBAL_INPUT_STREAM:
                    raise ValueError(
                        f"input_files for {consumer_uid} must name a file, e.g. "
                        f"'{_GLOBAL_INPUT_STREAM}:orbits'"
                    )
                input_cfg["source_output"] = primary_output_by_stream[source]
            if input_cfg["source_output"] not in output_ids_by_stream[source]:
                raise ValueError(
                    f"input_files for {consumer_uid} references unknown output "
                    f"'{input_cfg['source_output']}' of stream '{source}'"
                )

    def _fixed_input_uid(consumer_uid: str, label: str, path: str) -> str:
        uid = make_file_uid(consumer_uid, None, label)
        if uid not in files:
            register_file(
                files,
                GwaithFile(
                    uid=uid,
                    local_path=path,
                    final_path=path,
                    kind="input_file",
                    round_index=None,
                    stream_uid=consumer_uid,
                    retention_level=cfg["global_retention_level"],
                ),
            )
        return uid

    def _resolve_inputs(
        consumer_uid: str, inputs: list[dict[str, Any]], round_index: int
    ) -> tuple[list[str], list[str]]:
        """Return (CLI args, used file uids) for a node's input_files.

        Fit nodes see outputs from earlier rounds only, since all fits in a
        round run in parallel and before the noise node. The noise node also
        sees signal-stream outputs from its own round.
        """
        args: list[str] = []
        uses: list[str] = []
        for input_cfg in inputs:
            file_uid = None
            source = input_cfg["source_stream"]
            dataset = input_cfg["dataset"]
            if source == _GLOBAL_INPUT_STREAM:
                file_uid = global_input_uids[input_cfg["source_output"]]
                dataset = dataset or files[file_uid].dataset
            elif source is None:
                file_uid = _fixed_input_uid(
                    consumer_uid, f"input_{input_cfg['arg']}", input_cfg["path"]
                )
            else:
                same_round = consumer_uid == "noise" and source != "noise"
                max_round = round_index if same_round else round_index - 1
                file_uid = _latest_output_uid(
                    outputs_by_stream_round[source],
                    input_cfg["source_output"],
                    max_round,
                )
                if file_uid is None and input_cfg["initial"]:
                    file_uid = _fixed_input_uid(
                        consumer_uid,
                        f"initial_{input_cfg['arg']}",
                        input_cfg["initial"],
                    )
            if file_uid is None:
                if input_cfg["required"]:
                    raise ValueError(
                        f"No file available for --{input_cfg['arg']} of "
                        f"{consumer_uid} in round {round_index}; set 'initial'"
                    )
                continue
            uses.append(file_uid)
            args.extend([f"--{input_cfg['arg']}", files[file_uid].local_path])
            if input_cfg["dataset_arg"] and dataset:
                args.extend([f"--{input_cfg['dataset_arg']}", dataset])
        return args, uses

    round_entries: list[dict[str, Any]] = []
    latest_params_by_stream: dict[str, str | None] = {
        stream_cfg["uid"]: initial_prior_by_stream[stream_cfg["uid"]]
        for stream_cfg in cfg["signal_streams"]
    }
    final_residual_uid: str = original_data_uid
    final_noise_uid: str | None = None

    for round_index in range(cfg["rounds"]):
        round_tag = _round_tag(round_index)
        round_dir = target_dir / round_tag
        active_fit_nodes: list[str] = []
        prep_nodes: list[str] = []
        active_stream_uids: list[str] = []

        for stream_cfg in cfg["signal_streams"]:
            stream_uid = stream_cfg["uid"]
            if not _stream_runs_on_round(stream_cfg, round_index):
                continue

            active_stream_uids.append(stream_uid)
            fit_node_uid = make_node_uid(stream_uid, round_index, "fit")
            fit_node = GwaithNode(
                uid=fit_node_uid,
                round_index=round_index,
                stream_uid=stream_uid,
                name=f"{stream_uid}_fit_{round_tag}",
                resources=stream_cfg["resources"],
            )

            input_data_uid = next_data_by_stream[stream_uid]
            input_file = files[input_data_uid]
            fit_node.uses.append(input_data_uid)

            latest_prior_round = _latest_produced_round(stream_uid, round_index - 1)
            prior_uid = None
            if latest_prior_round is not None:
                prior_uid = prior_uid_by_stream_round[stream_uid].get(
                    latest_prior_round
                )
            if prior_uid is None:
                prior_uid = initial_prior_by_stream[stream_uid]
            if prior_uid and prior_uid in files:
                fit_node.uses.append(prior_uid)
            else:
                prior_uid = None

            likelihood_path = str(
                round_dir / f"{round_tag}_{stream_uid}_likelihood.yaml"
            )
            stream_outputs: list[dict[str, Any]] = stream_cfg["outputs"]
            output_paths: dict[str, str] = {}
            for output_cfg in stream_outputs:
                output_paths[output_cfg["id"]] = str(
                    round_dir
                    / f"{round_tag}_{stream_uid}_{output_cfg['id']}.{output_cfg['extension']}"
                )

            extra_input_args, extra_input_uids = _resolve_inputs(
                stream_uid, stream_cfg["inputs"], round_index
            )
            fit_node.uses.extend(extra_input_uids)

            command = [stream_cfg["executable"], *_data_file_args(input_file)]
            command.extend(extra_input_args)
            if prior_uid:
                command.extend(["--priors", files[prior_uid].local_path])
            command.extend(
                _build_cli_args(
                    stream_cfg["options"],
                    excluded={
                        "prior-file",
                        "priors",
                        "init-params",
                        "output-dataset",
                    },
                )
            )
            for output_cfg in stream_outputs:
                command.extend(
                    _output_file_args(output_cfg, output_paths[output_cfg["id"]])
                )
            command.extend(["--output-params", likelihood_path])
            fit_node.command = command

            likelihood_uid = make_file_uid(stream_uid, round_index, "likelihood")
            produced_output_uids: list[str] = []
            subtractable_output_uids: list[str] = []
            for output_cfg in stream_outputs:
                output_id = output_cfg["id"]
                output_uid = make_file_uid(stream_uid, round_index, output_id)
                output_path = output_paths[output_id]
                output_file = GwaithFile(
                    uid=output_uid,
                    local_path=output_path,
                    final_path=output_path,
                    produced_by=fit_node_uid,
                    retention_level=cfg["global_retention_level"],
                    dataset=output_cfg.get("dataset"),
                    kind=output_cfg.get("kind", "artifact"),
                    round_index=round_index,
                    stream_uid=stream_uid,
                )
                register_file(files, output_file)
                produced_output_uids.append(output_uid)
                outputs_by_stream_round[stream_uid].setdefault(round_index, {})[
                    output_id
                ] = output_uid
                if output_cfg.get("include_in_subtraction", False):
                    subtractable_output_uids.append(output_uid)

            likelihood_file = GwaithFile(
                uid=likelihood_uid,
                local_path=likelihood_path,
                final_path=likelihood_path,
                produced_by=fit_node_uid,
                retention_level=cfg["global_retention_level"],
                dataset=None,
                kind="likelihood",
                round_index=round_index,
                stream_uid=stream_uid,
            )
            register_file(files, likelihood_file)
            fit_node.produces.extend(produced_output_uids)
            fit_node.produces.append(likelihood_uid)
            register_node(nodes, files, fit_node)
            streams[stream_uid].nodes.append(fit_node_uid)
            active_fit_nodes.append(fit_node_uid)

            subtractable_output_uids_by_stream_round[stream_uid][round_index] = (
                subtractable_output_uids
            )
            prior_uid_by_stream_round[stream_uid][round_index] = likelihood_uid
            latest_params_by_stream[stream_uid] = likelihood_uid

        residual_uid = make_file_uid("central", round_index, "combined_residual")
        residual_path = str(round_dir / f"{round_tag}_combined_residual.hdf")
        residual_node_uid = make_node_uid("central", round_index, "combined_residual")
        residual_node = GwaithNode(
            uid=residual_node_uid,
            round_index=round_index,
            stream_uid="central",
            name=f"combined_residual_{round_tag}",
            resources=cfg["subtraction"]["resources"],
        )
        residual_signal_uids: list[str] = []
        for stream_cfg in cfg["signal_streams"]:
            stream_uid = stream_cfg["uid"]
            latest_signal_round = _latest_produced_round(stream_uid, round_index)
            if latest_signal_round is None:
                continue
            for signal_uid in subtractable_output_uids_by_stream_round[stream_uid].get(
                latest_signal_round, []
            ):
                if signal_uid in files:
                    residual_signal_uids.append(signal_uid)
        residual_node.uses.append(original_data_uid)
        residual_node.uses.extend(residual_signal_uids)
        if residual_signal_uids:
            residual_command = [
                cfg["subtraction"]["executable"],
                "--data",
                files[original_data_uid].local_path,
            ]
            for signal_uid in residual_signal_uids:
                residual_command.extend(
                    ["--signal-files", files[signal_uid].local_path]
                )
            residual_command.extend(_build_cli_args(cfg["subtraction"]["options"]))
            residual_command.extend(["--output", residual_path])
        else:
            residual_command = [
                "cp",
                files[original_data_uid].local_path,
                residual_path,
            ]
        residual_node.command = residual_command
        residual_file = GwaithFile(
            uid=residual_uid,
            local_path=residual_path,
            final_path=residual_path,
            produced_by=residual_node_uid,
            retention_level=cfg["global_retention_level"],
            dataset=cfg["input_dataset"],
            kind="combined_residual",
            round_index=round_index,
            stream_uid="central",
        )
        register_file(files, residual_file)
        residual_node.produces.append(residual_uid)
        register_node(nodes, files, residual_node)
        streams["central"].nodes.append(residual_node_uid)
        final_residual_uid = residual_uid

        noise_node_uid: str | None = None
        if cfg["noise"]["enabled"]:
            noise_node_uid = make_node_uid("noise", round_index, "estimate_psd")
            noise_node = GwaithNode(
                uid=noise_node_uid,
                round_index=round_index,
                stream_uid="noise",
                name=f"noise_{round_tag}",
                resources=cfg["noise"]["resources"],
            )
            extra_input_args, extra_input_uids = _resolve_inputs(
                "noise", cfg["noise"]["inputs"], round_index
            )
            noise_node.uses.append(residual_uid)
            noise_node.uses.extend(extra_input_uids)
            noise_node.command = [
                cfg["noise"]["executable"],
                "--data",
                residual_path,
                *extra_input_args,
            ]
            noise_node.command.extend(_build_cli_args(cfg["noise"]["options"]))

            for index, output_cfg in enumerate(cfg["noise"]["outputs"]):
                output_id = output_cfg["id"]
                # Keep the historical file name for the primary PSD output.
                suffix = "noise" if index == 0 else f"noise_{output_id}"
                output_path = str(
                    round_dir / f"{round_tag}_{suffix}.{output_cfg['extension']}"
                )
                output_uid = make_file_uid("noise", round_index, output_id)
                noise_node.command.extend(_output_file_args(output_cfg, output_path))
                register_file(
                    files,
                    GwaithFile(
                        uid=output_uid,
                        local_path=output_path,
                        final_path=output_path,
                        produced_by=noise_node_uid,
                        retention_level=cfg["global_retention_level"],
                        dataset=output_cfg.get("dataset"),
                        kind=output_cfg.get("kind", "artifact"),
                        round_index=round_index,
                        stream_uid="noise",
                    ),
                )
                noise_node.produces.append(output_uid)
                outputs_by_stream_round["noise"].setdefault(round_index, {})[
                    output_id
                ] = output_uid
                if index == 0:
                    final_noise_uid = output_uid

            register_node(nodes, files, noise_node)
            streams["noise"].nodes.append(noise_node_uid)

        if round_index + 1 < cfg["rounds"]:
            next_round_dir = target_dir / _round_tag(round_index + 1)
            for stream_cfg in cfg["signal_streams"]:
                stream_uid = stream_cfg["uid"]
                prep_node_uid = make_node_uid(
                    "central", round_index, f"prep_{stream_uid}"
                )
                prep_node = GwaithNode(
                    uid=prep_node_uid,
                    round_index=round_index,
                    stream_uid="central",
                    name=f"prepare_{stream_uid}_input_{round_tag}",
                    resources=cfg["subtraction"]["resources"],
                )
                prep_node.uses.append(original_data_uid)
                other_signal_uids: list[str] = []
                for other_stream_cfg in cfg["signal_streams"]:
                    other_stream_uid = other_stream_cfg["uid"]
                    if other_stream_uid == stream_uid:
                        continue
                    latest_signal_round = _latest_produced_round(
                        other_stream_uid, round_index
                    )
                    if latest_signal_round is None:
                        continue
                    for signal_uid in subtractable_output_uids_by_stream_round[
                        other_stream_uid
                    ].get(latest_signal_round, []):
                        if signal_uid in files:
                            other_signal_uids.append(signal_uid)

                prep_node.uses.extend(other_signal_uids)

                next_data_uid = make_file_uid(stream_uid, round_index + 1, "input_data")
                next_data_path = str(
                    next_round_dir
                    / f"{_round_tag(round_index + 1)}_{stream_uid}_input_data.hdf"
                )

                if other_signal_uids:
                    prep_command = [
                        cfg["subtraction"]["executable"],
                        "--data",
                        files[original_data_uid].local_path,
                    ]
                    for signal_uid in other_signal_uids:
                        prep_command.extend(
                            ["--signal-files", files[signal_uid].local_path]
                        )
                    prep_command.extend(_build_cli_args(cfg["subtraction"]["options"]))
                    prep_command.extend(["--output", next_data_path])
                else:
                    prep_command = [
                        "cp",
                        files[original_data_uid].local_path,
                        next_data_path,
                    ]
                prep_node.command = prep_command

                next_data_file = GwaithFile(
                    uid=next_data_uid,
                    local_path=next_data_path,
                    final_path=next_data_path,
                    produced_by=prep_node_uid,
                    retention_level=cfg["global_retention_level"],
                    dataset=cfg["input_dataset"],
                    kind="input_data",
                    round_index=round_index + 1,
                    stream_uid=stream_uid,
                )
                register_file(files, next_data_file)
                prep_node.produces.append(next_data_uid)
                register_node(nodes, files, prep_node)
                streams["central"].nodes.append(prep_node_uid)
                prep_nodes.append(prep_node_uid)
                next_data_by_stream[stream_uid] = next_data_uid
            else:
                next_data_by_stream[stream_uid] = original_data_uid

        round_entries.append(
            {
                "round": round_index,
                "active_streams": active_stream_uids,
                "fit_nodes": active_fit_nodes,
                "prep_nodes": prep_nodes,
                "combined_residual_node": residual_node_uid,
                "combined_residual_file": residual_uid,
                "noise_node": noise_node_uid,
            }
        )

    workflow = GwaithWorkflow(
        cp=config_parser,
        name=cfg["name"],
        workflow_dir=str(target_dir),
        rounds=cfg["rounds"],
        input_data={
            "data_file": cfg["input_file"],
            "data_dataset": cfg["input_dataset"],
        },
        global_retention_level=cfg["global_retention_level"],
    )
    workflow.streams = list(streams.values())
    workflow.nodes = list(nodes.values())
    workflow.files = list(files.values())
    workflow.round_entries = round_entries
    workflow.final_residual_file = final_residual_uid
    workflow.final_noise_file = final_noise_uid
    workflow.final_params = latest_params_by_stream
    return workflow


def build_workflow(
    config_parser,
    workflow_name: str | None = None,
    workflow_dir: str | None = None,
) -> GwaithWorkflow:
    """Build a workflow graph.

    Args:
            config_parser: Configuration parser with workflow settings
            workflow_name: Optional name for the workflow
            workflow_dir: Optional output directory for the workflow

    Returns:
            GwaithWorkflow
    """
    return build_workflow_graph(
        config_parser, workflow_name=workflow_name, workflow_dir=workflow_dir
    )
