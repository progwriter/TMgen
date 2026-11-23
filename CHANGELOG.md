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
- Type stubs and a `py.typed` marker.
- `tmgen.__version__`.
- `lognormal_tm` is exported from `tmgen.models`.
- Keyword-only `rng` argument (seed or `numpy.random.Generator`) on every
  random model and on `hmc_sample` for reproducible, isolated randomness. Without it the models
  still draw from NumPy's global state, so `numpy.random.seed` works as before.

### Changed

- The HMC sampler is a typed Cython kernel that updates its working products
  incrementally (Gram matrix), roughly 8-25x faster. Samples are statistically
  equivalent but no longer bit-identical to the previous implementation for
  a given seed. For very large constraint sets (more than 256 MiB of Gram
  matrix) it falls back to recomputing the products.
- `gravity_tm`, `spike_tm` and `modulated_gravity_tm` are vectorized
  (`gravity_tm` is hundreds of times faster for large networks); results are
  unchanged.
- `TrafficMatrix` and the traffic models are now pure Python/NumPy instead
  of Cython, with identical results for a given NumPy seed. Only the HMC
  sampler is still compiled. `TrafficMatrix` now raises `TypeError` for
  non-array input.
- Requires Python 3.11 or newer.
- Package metadata and build requirements moved to `pyproject.toml`; the
  version is derived from git tags.
- `TrafficMatrix.to_csv` writes one row per epoch (`n * n` columns) for
  multi-epoch matrices, as documented, instead of one row per node pair.

### Fixed

- `modulated_gravity_tm` scaled the traffic by an extra `mean_traffic /
  num_nodes`, so the total of each epoch did not follow the requested
  total-traffic cycle.
- `modulated_gravity_tm` raised `TypeError` for epoch counts passed as floats
  (#9).
- `TrafficMatrix.mean()` returned the maximum over epochs instead of the mean.
- `TrafficMatrix.to_csv` failed for single-epoch matrices.
- The HMC sampler returned invalid samples (inverted shape check, wrong
  Cholesky factor, mutated its inputs).
- The D-ITG injector's SIGINT handler raised `NameError`.
- Importing `tmgen.plot` without seaborn raised `TypeError`.
- `tmgen.plot.heatmap` raised `NameError` when seaborn was not installed; it
  now falls back to plain matplotlib.
- `InjectorBase.run` was never enforced as abstract.

### Removed

- Support for Python versions older than 3.11, and the `six` dependency.
