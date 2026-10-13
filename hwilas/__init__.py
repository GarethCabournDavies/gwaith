"""Hwilas package."""
import warnings

# Suppress this constants advisory, it is _very_ noisy
warnings.filterwarnings(
    "ignore",
    message=r"The following constants differ between lisaconstants and the version of astropy you have installed:",
    category=UserWarning,
    module=r"lisaconstants\.compat\.astropy",
)

from .common import (
    __version__,
    add_hwilas_common_options,
    import_optional,
    init_logging,
)
from .data_processing.truth import (
    add_truth_arguments,
    load_truth_file,
)

__all__ = [
    "__version__",
    "add_hwilas_common_options",
    "add_truth_arguments",
    "import_optional",
    "init_logging",
    "load_truth_file",
]
