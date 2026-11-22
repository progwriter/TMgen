# Development

## Setup

You need Python 3.11+ and a C compiler. Create a virtual environment and
install the package in editable mode with the development tools (ruff, mypy,
pytest, matplotlib, seaborn):

```
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
```

The editable install builds the Cython extensions in place. After editing a
`.pyx` file, rebuild them by running `pip install -e .` again; Python-only
changes need nothing.

The package version comes from git tags (setuptools-scm), so an untagged
checkout reports a `0.1.devN` version.

## Layout

- `src/tmgen/models.pyx`, `tm.pyx`, `hmc.pyx`: the compiled modules.
- `src/tmgen/*.pyi`, `*.pxd`: type stubs for the public API and the Cython
  interface. Ruff and mypy cannot see `.pyx` files, so when you change a
  signature there, update the matching `.pyi` (and `.pxd`) by hand.
- `src/tmgen/plot.py`, `src/tmgen/inject/`: plain Python.
- `tests/`, `docs/`, `tools/` (release helpers).

## Checks

CI runs these on every push (Linux, macOS and Windows, Python 3.11-3.15):

```
ruff check
ruff format --check     # `ruff format` fixes formatting
mypy                    # strict, covers src, tests and tools
pytest                  # tests plus the docs/ doctests
```

Notes:

- Ruff sorts imports differently depending on whether the extensions are
  built, so run it after `pip install -e ".[dev]"`, as CI does.
- Tests and doctests run in a temporary directory (`conftest.py`), so they
  can write files freely.
- A `package` CI job builds the sdist and wheel and imports the installed
  wheel, so packaging mistakes show up before a release.

## Documentation

The docs are Sphinx pages in `docs/` and are also published on Read the Docs.
To build them locally:

```
pip install sphinx sphinx_rtd_theme
python -m sphinx -b html docs docs/_build
```

## Contributing changes

- Use [Conventional Commits](https://www.conventionalcommits.org/) (`feat`,
  `fix`, `docs`, `refactor`, `test`, `build`, `ci`, ...): imperative subject up
  to 50 characters, with a body explaining what and why.
- Keep commits small and self-contained; each should pass the checks above.
- Add an entry under **Unreleased** in `CHANGELOG.md` for every user-visible
  change, in the same commit.
- Update the README, docs, stubs and tests together with behavior changes.

## Releasing

See [RELEASING.md](RELEASING.md).
