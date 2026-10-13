#!/usr/bin/env bash
set -euo pipefail
# Git runs clean filters from the checkout root; only standard-library Python is needed.
git config --local filter.hwilas-notebook.clean 'python3 tools/git/strip_notebook_outputs.py'
git config --local filter.hwilas-notebook.smudge cat
git config --local filter.hwilas-notebook.required true
echo 'Notebook source will be staged without outputs; local executed notebooks are preserved.'
