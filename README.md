<div align="center">
  <img src="logo/hwilas_logo.png" alt="Hwilas logo" width="400">
</div>

[![Coverage](https://github.com/uk-lisa-gs/hwilas/actions/workflows/coverage.yml/badge.svg)](https://github.com/uk-lisa-gs/hwilas/actions/workflows/coverage.yml)
[![Python](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/downloads/)
[![License](https://img.shields.io/github/license/uk-lisa-gs/hwilas)](LICENSE)

# Hwilas: Hierarchical Wave Inspiral LISA Analysis & Search

Here we provide a basic search for MBHB systems in LISA.

This is a template bank-based search, which utilises clustering techniques to reduce filesizes and
computational effort, as well as simple time slides for false alarm rate estimation.

The search is hierarchical, meaning we peform a first pass, remove N signals from the data, and then
perform the search again. This removes the very loudest signals from the data.

Hwilas (`/ˈʍiːlas/`, `wee-lass`) is the Cornish dialect word for search or seek. 

## Installation

Hwilas has three dependencies that are installed directly from git over **SSH**
(`gwaith`, `bbhx`, and — for the test extras — `pycbc-bbhx-plugin`). This avoids
the common headless-system failure where pip's HTTPS `git clone` falls back to a
missing GUI credential helper (e.g. `/usr/libexec/openssh/gnome-ssh-askpass`)
and aborts with:

```text
Username for 'https://github.com':
fatal: cannot exec '/usr/libexec/openssh/gnome-ssh-askpass': No such file or directory
```

Because the clones go over SSH, you **must have a GitHub SSH key set up** on the
machine you install from. Add your public key to your GitHub account and confirm
the connection works:

```bash
ssh -T git@github.com   # should print "Hi <username>! You've successfully authenticated..."
```

Then install as usual (add `--user` if you do not want to write to a system
prefix, and `.[test]` to pull in the optional test dependencies):

```bash
pip install --user .
# or, with test extras:
pip install --user ".[test]"
```

> **No SSH key available?** If you cannot configure an SSH key (e.g. on a shared
> node), you can instead keep the clones non-interactive over HTTPS by installing
> with `GIT_TERMINAL_PROMPT=0 GIT_ASKPASS= SSH_ASKPASS= pip install --user .` —
> but note that requires temporarily reverting the dependency URLs in
> `pyproject.toml` back to `git+https://...`.
