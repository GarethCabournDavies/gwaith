<div align="center">
  <img src="logo/hwilas_logo.png" alt="Hwilas logo" width="400">
</div>

[![Python](https://img.shields.io/badge/python-3.12%2B-blue.svg)](pyproject.toml)
[![Unit tests](https://github.com/uk-lisa-gs/hwilas/actions/workflows/test_suite.yml/badge.svg)](https://github.com/uk-lisa-gs/hwilas/actions/workflows/test_suite.yml)
[![Documentation](https://github.com/uk-lisa-gs/hwilas/actions/workflows/docs.yml/badge.svg)](https://github.com/uk-lisa-gs/hwilas/actions/workflows/docs.yml)
[![Coverage](https://codecov.io/gh/uk-lisa-gs/hwilas/graph/badge.svg)](https://codecov.io/gh/uk-lisa-gs/hwilas)
[![License](https://img.shields.io/github/license/uk-lisa-gs/hwilas)](LICENSE)

# Hwilas: Hierarchical Wave Inspiral LISA Analysis & Search

Here we provide a basic search for MBHB systems in LISA.

This is a template bank-based search, which utilises clustering techniques to reduce filesizes and
computational effort, as well as simple time slides for false alarm rate estimation.

The search is hierarchical, meaning we peform a first pass, remove N signals from the data, and then
perform the search again. This removes the very loudest signals from the data.

Hwilas (`/ˈʍiːlas/`, `wee-lass`) is the Cornish dialect word for search or seek. 

## Installation

Hwilas installs `gwaith` directly from git over **SSH**. The optional `tdi`
extra (`pip install -e '.[tdi]'`) also installs the private `tdispline`
repository and requires access to it. The test extra does not install
`tdispline`; tests requiring it skip when it is unavailable.
SSH avoids the common headless-system failure where
pip's HTTPS `git clone` falls back to a missing GUI credential helper (e.g.
`/usr/libexec/openssh/gnome-ssh-askpass`) and aborts with:

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

Sky-mode sweeps use eight sky proposals without additional dependencies. For the
full 64 sky/helicity branches, install the optional private `gwreparam` dependency
when your GitHub account has access:

```bash
pip install 'gwreparam @ git+ssh://git@github.com/uk-lisa-gs/gwreparam.git'
```

The optimiser automatically uses the full exploration when `gwreparam` is
installed. The eight-mode fallback preserves phase and explores sky reflections
and rotations; it does not explore the additional helicity branches. These are
candidate proposals evaluated with the waveform objective, not assumed exact
waveform symmetries. Set `--sky-sweep-every 0` to disable sweeps.

For Mojito preparation, search configuration and diagnostics, see
[the search workflow](docs/mojito_search.md).

To export a stable search event as MBHBS parameter/configuration YAML, see
[the PE export command and current MBHBS capability limits](docs/mbhbs_pe_export.md).

## CI and coverage

GitHub Actions runs the unit tests and coverage on Python 3.12. The coverage
workflow saves XML and HTML reports as the `coverage-reports` artifact and uploads
the results to [Codecov](https://codecov.io/gh/uk-lisa-gs/hwilas).
Enable this repository in Codecov and add its upload token as the `CODECOV_TOKEN`
repository secret to populate the coverage percentage badge.

If Git dependencies require private repository access, set `GH_DEPENDENCY_TOKEN`
to a read-only token with access to those repositories. Otherwise CI uses the
standard `GITHUB_TOKEN`, whose access is limited to this repository.

To measure coverage locally after installing the test extras:

```bash
python -m pip install coverage
python -m coverage run --source=hwilas,tools -m pytest tests
python -m coverage report -m
python -m coverage html
```

The documentation workflow builds the Markdown guides with MkDocs and saves the
site as the `documentation` artifact; it does not publish a hosted site. Build it
locally with `python -m pip install 'mkdocs>=1.6,<2'` followed by
`python -m mkdocs build --site-dir /tmp/hwilas-docs`.
