"""Unit tests for the types.argparse module."""

import argparse

import pytest

from hwilas.types.argparse import KeyValueAction


def test_key_value_action_basic():
    """Test the basic functionality of KeyValueAction."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--kwargs", action=KeyValueAction, nargs="+")
    args = parser.parse_args(["--kwargs", "key1:value1", "key2:value2"])
    assert args.kwargs == {"key1": "value1", "key2": "value2"}


def test_key_value_action_with_type():
    """Test KeyValueAction with a type converter."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--kwargs", action=KeyValueAction, nargs="+", type=int)
    args = parser.parse_args(["--kwargs", "key1:10", "key2:-5"])
    assert args.kwargs == {"key1": 10, "key2": -5}


def test_key_value_action_invalid_format():
    """Test that KeyValueAction raises an error for invalid format."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--kwargs", action=KeyValueAction, nargs="+")
    with pytest.raises(SystemExit):
        parser.parse_args(["--kwargs", "key1_value1"])


def test_key_value_action_conversion_error():
    """Test that KeyValueAction raises an error for a failed conversion."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--kwargs", action=KeyValueAction, nargs="+", type=int)
    with pytest.raises(SystemExit):
        parser.parse_args(["--kwargs", "key1:not_an_int"])


def test_key_value_action_multiple_calls():
    """Test that multiple uses of the argument accumulate."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--kwargs", action=KeyValueAction)
    args = parser.parse_args(["--kwargs", "key1:val1", "--kwargs", "key2:val2"])
    assert args.kwargs == {"key1": "val1", "key2": "val2"}
