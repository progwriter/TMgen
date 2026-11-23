# TMgen agent guide

TMgen generates network traffic matrices. Python 3.11+, with one Cython extension (`hmc`).

## Layout

- `src/tmgen/tm.py`, `models.py`, `plot.py`, `inject/`: pure Python (NumPy).
- `src/tmgen/hmc.pyx`: the only compiled module; `hmc.pyi` (typed API) is hand-written. No `.pxd` files are shipped: the compiled module is not a cimport API.
- `tests/`, `docs/` (Sphinx; `docs/quickstart.rst` doctests run under pytest), `tools/` (release helpers, type-checked and tested).
- `agentdocs/` (git-ignored): local notes such as `PLAN.md` and `API_CRITIQUE.md`. Never commit them. Keep `agentdocs/PLAN.md` current as work lands.

## Setup and checks

`DEVELOPMENT.md` is the human-facing version of this section and the commit rules; keep them consistent.

```
uv venv && uv pip install -e '.[dev]'
uv pip install -e .   # rebuild after editing hmc.pyx
```

CI runs these (plus the pre-commit hooks in `.pre-commit-config.yaml`, which add whitespace checks and a strict Sphinx build, `python -m sphinx -W -b html docs docs/_build`), and every commit should pass them:

```
ruff check
ruff format --check
mypy
pytest            # tests + docs doctests, each in a temp dir (conftest.py)
```

CI also builds the docs strictly in a `docs` job and measures coverage (`pytest --cov`, threshold in `pyproject.toml`) on one test leg.

Gotchas:

- Ruff's import sorting depends on whether the extension is built, so run `ruff check` after building.
- Ruff and mypy do not see `.pyx` files. When `hmc.pyx`'s signature changes, update `hmc.pyi` by hand.

## Code style

- Annotate everything (mypy strict). Python 3.11 syntax (`X | None`, built-in generics).
- Prefer `attrs` for data classes, `argparse` with named arguments for scripts, and `logging` over `print`.
- Comments only for non-obvious why, never how or history. Update a comment when you change the code it explains.
- Vocabulary: *node* (not PoP), *ingress* / *egress* (not origin / destination / OD), *epoch* (not time step or TM), *entry* (one ingress-egress volume in one epoch). Parameters: `num_nodes`, `num_epochs`, `total_traffic` (sum of a matrix), `mean_entry` (mean of an entry). Rename parameters with `tmgen._compat.renamed_arguments` so old keywords keep working for one release.
- Update README, docs, stubs and tests together with behavior changes.
- Add a `CHANGELOG.md` entry under **Unreleased** for every user-visible change (new features, behavior changes, fixes, removals), in the same commit.

## Commits

- Conventional Commits (`feat`, `fix`, `docs`, `style`, `refactor`, `perf`, `test`, `build`, `ci`, `chore`). Imperative, capitalized subject up to 50 characters, no trailing period; body wrapped at 72 explaining what and why.
- Small, self-contained commits. Each must build and pass the checks above.
- Fix up earlier unpushed commits with `git commit --fixup=<sha>` and autosquash later. Do not push, force-push or rewrite history unless the owner explicitly asks.

## Releases

See `RELEASING.md`. Releases are tag-driven (`vX.Y.Z`): the version comes from the tag via setuptools-scm, notes come from `CHANGELOG.md`, and `.github/workflows/release.yml` publishes to TestPyPI and, after manual approval, PyPI. Never create or push tags unless the owner asks. Keep `RELEASING.md`, the workflow and `tools/release_notes.py` consistent when changing any of them.
