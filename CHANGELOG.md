# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and versions follow
[Semantic Versioning](https://semver.org/). While the version is below 1.0,
minor releases may contain breaking changes.

Add an entry under **Unreleased** for every user-visible change. At release
time, rename that section to the new version and date (see `RELEASING.md`).

## [Unreleased]

### Added

- `tmgen.hmc.hmc_sample`, a Python-visible wrapper around the truncated
  Gaussian HMC sampler.
- Type stubs and a `py.typed` marker for the compiled modules.
- `tmgen.__version__`.
- `lognormal_tm` is exported from `tmgen.models`.

### Changed

- Requires Python 3.11 or newer.
- Package metadata and build requirements moved to `pyproject.toml`; the
  version is derived from git tags.
- `TrafficMatrix.to_csv` writes one row per epoch (`n * n` columns) for
  multi-epoch matrices, as documented, instead of one row per node pair.

### Fixed

- `modulated_gravity_tm` raised `TypeError` for epoch counts passed as floats
  (#9).
- `TrafficMatrix.mean()` returned the maximum over epochs instead of the mean.
- `TrafficMatrix.to_csv` failed for single-epoch matrices.
- The HMC sampler returned invalid samples (inverted shape check, wrong
  Cholesky factor, mutated its inputs).
- The D-ITG injector's SIGINT handler raised `NameError`.
- Importing `tmgen.plot` without seaborn raised `TypeError`.
- `InjectorBase.run` was never enforced as abstract.

### Removed

- Support for Python versions older than 3.11, and the `six` dependency.
