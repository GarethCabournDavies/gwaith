"""Unit tests for the hwilas.common module."""

import argparse
import importlib
import logging
from unittest import mock

import numpy as np
import pytest

from hwilas import common


def test_add_hwilas_common_options():
    """Test that common options are added to the parser."""
    parser = argparse.ArgumentParser()
    common.add_hwilas_common_options(parser)
    args = parser.parse_args(["-vv"])
    assert args.verbose == 2

    with pytest.raises(SystemExit):
        parser.parse_args(["--version"])


def test_init_logging():
    """Test logging initialization."""
    assert common.init_logging(0) == logging.WARNING
    assert common.init_logging(1) == logging.INFO
    assert common.init_logging(2) == logging.DEBUG
    assert common.init_logging(100) == logging.DEBUG


def test_import_optional_existing():
    """Test importing an existing module."""
    module = common.import_optional("numpy")
    import numpy as np

    assert module == np


def test_import_optional_missing():
    """Test importing a missing module."""
    module_name = "this_module_does_not_exist"
    # Ensure it's not actually there
    with pytest.raises(ImportError):
        importlib.import_module(module_name)

    module = common.import_optional(module_name)
    assert isinstance(module, common._MissingModule)

    with pytest.raises(ImportError) as excinfo:
        module.some_function()

    assert module_name in str(excinfo.value)
    assert "some_function" in str(excinfo.value)

@mock.patch("hwilas.common.version")
def test_resolve_version_found(mock_version):
    """Test that the version is resolved when the package is installed."""
    mock_version.return_value = "0.1.0"
    assert common._resolve_version() == "0.1.0"
    mock_version.assert_called_once_with("hwilas")

@mock.patch("hwilas.common.version")
def test_resolve_version_not_found(mock_version):
    """Test that the version is resolved when the package is not installed."""
    mock_version.side_effect = importlib.metadata.PackageNotFoundError
    # Fallback to the hardcoded version
    assert common._resolve_version() == common.__version__
