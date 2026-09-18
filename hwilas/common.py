"""Common utility functions for the Hwilas package."""

import argparse
import importlib
import inspect
import logging
from importlib.metadata import PackageNotFoundError, version

__version__ = "0.1.0"

def _resolve_version() -> str:
    try:
        return version("hwilas")
    except PackageNotFoundError:
        return __version__


def add_hwilas_common_options(
    parser: argparse.ArgumentParser,
) -> argparse.ArgumentParser:
    """Add common Hwilas CLI options.

    - `--version`: print package version
    - `-v/--verbose`: increase logging level (`-v` => INFO, `-vv` => DEBUG)
    """
    parser.add_argument(
        "--version",
        action="version",
        version=f"%(prog)s {_resolve_version()}",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="count",
        default=0,
        help="Increase log verbosity (use -v for INFO, -vv for DEBUG)",
    )
    return parser


_LOG_FORMAT = "%(levelname)s:%(asctime)s:%(name)s:%(message)s"
_LOG_DATEFMT = "%Y-%m-%dT%H:%M:%S%z"

_DEFAULT_REDUCE_LOGGING_LEVEL = [
    "matplotlib.font_manager",
    "Pegasus",
]



def init_logging(
    verbosity: int = 0,
    default_level: int = logging.WARNING,
    reduce_logging_level: list[str] | None = _DEFAULT_REDUCE_LOGGING_LEVEL
) -> int:
    """Initialize unified logging format and return selected level.

    You can also pass a list to reduce_logging_level if there are
    certain loggers which spam too much.
    (looking at you matplotlib.font_manager 👀)
    """

    # the logging.WHATEVER levels are just wrappers for ints separated by 10
    # We reduce the level according to the number in verbosity.
    # verbosity is usually set by args.verbose in the common_options above,
    # which is an integer count of --verboses
    level = default_level - verbosity * 10
    reduced_level = default_level - (verbosity - 1) * 10

    if reduce_logging_level is not None:
        # This is a list of named loggers that we need to make a bit quieter
        for logger_name in reduce_logging_level:
            logging.getLogger(logger_name).setLevel(reduced_level)

    logging.basicConfig(level=level, format=_LOG_FORMAT, datefmt=_LOG_DATEFMT)
    return level


class _MissingModule:
    def __init__(self, name):
        self._name = name

    def __getattr__(self, attr):
        frame = inspect.currentframe()
        outer = inspect.getouterframes(frame, 2)
        func = outer[1][3] if len(outer) > 1 else "<unknown>"
        msg = (
            f"The function '{func}' tried to access attribute '{attr}' "
            f"of module '{self._name}', but it is not installed. "
            f"Please install '{self._name}' to enable this functionality."
        )
        raise ImportError(msg)


def import_optional(module_name):
    """Import a module or return a stub that raises helpful errors."""
    try:
        return importlib.import_module(module_name)
    except ImportError:
        return _MissingModule(module_name)


__all__ = [
    "__version__",
    "add_hwilas_common_options",
    "import_optional",
    "init_logging",
]