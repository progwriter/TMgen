# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and versions follow
[Semantic Versioning](https://semver.org/). While the version is below 1.0,
minor releases may contain breaking changes.

Add an entry under **Unreleased** for every user-visible change. At release
time, rename that section to the new version and date (see `RELEASING.md`).

## [Unreleased]

### Added

- `TrafficMatrix.zero_diagonal()`, which returns a copy without self-traffic.
- `tmgen.hmc.hmc_sample`, a Python-visible wrapper around the truncated
  Gaussian HMC sampler.
- Type stubs and a `py.typed` marker.
- `tmgen.__version__`.
- `lognormal_tm` is exported from `tmgen.models`.
- Keyword-only `rng` argument (seed or `numpy.random.Generator`) on every
  random model and on `hmc_sample` for reproducible, isolated randomness. Without it the models
  still draw from NumPy's global state, so `numpy.random.seed` works as before.

### Changed

- `TrafficMatrix.between(ingress, egress, agg=None)` replaces
  `between(o, d, modestr='all')`: `agg` is `None` for the series over the
  epochs, or `'min'`, `'max'` or `'mean'` for a single number, and the return
  type follows from it. The old keywords and the old string modes (`'all'`,
  `'*'`, `'maximum'`, `'average'`, and the misspelled `'mininum'`) still work
  with a `DeprecationWarning`.
- Parameters are named consistently: `num_epochs` everywhere (was `num_tms` in
  `modulated_gravity_tm`); `total_traffic` for the total of a matrix
  (`modulated_gravity_tm`, `random_gravity_tm`, `gravity_tm`); `mean_entry` for
  the mean of an entry in `exp_tm`; `mu` in `lognormal_tm`, which was
  documented as a mean but is the mean of the underlying normal (the mean of
  the samples is `exp(mu + sigma**2 / 2)`); `value` in `exact_tm`. The old
  keyword names still work and emit a `DeprecationWarning`; positional calls
  are unaffected.
- `TrafficMatrix` is immutable: the constructor accepts any array-like, converts
  it to float64 and copies it once, and `matrix` is a read-only property (so
  `tm.matrix = ...` and `tm.matrix[...] = ...` fail, and `at_time`, `between`
  and the other accessors return read-only data; copy before modifying).
  It raises `TMgenException` for the wrong shape, NaN, infinite or negative
  volumes, or an empty matrix, and adding matrices of different shapes raises
  `TMgenException`.
- All models validate their arguments and raise `TMgenException` (a
  `ValueError`) with a message naming the argument: sizes and epoch counts must
  be positive whole numbers, volumes finite and non-negative, `sigma` and the
  variances non-negative, and populations non-negative with a positive sum.
  `uniform_tm` raises when `low > high` instead of silently swapping the
  bounds.
- `TMgenException` is now a subclass of `ValueError`. `hmc_sample` raises it
  (instead of a bare `ValueError`) and checks the shapes of its arguments and
  that `num_samples` is at least 1; `num_samples=1` returns the initial point.
- `random_gravity_tm` follows the paper and the reference the issue cites
  (#7): ingress and egress weights are independent exponential draws instead
  of one normalized uniform vector, so matrices are more skewed and no longer
  symmetric. The total is random, `mean_traffic` on average; pass
  `normalize=True` for an exact total, or `symmetric=True` to share the
  weights between rows and columns (the total is then `mean_traffic * (1 +
  1/num_nodes)` on average). The docs now say that `mean_traffic` is the
  total, not the per-pair average.
- `spike_tm` follows the paper: `num_spikes` is the total number of spikes
  over the whole `num_nodes x num_nodes x num_epochs` array, chosen uniformly,
  instead of that many per epoch. Multi-epoch results change.
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
  sampler is still compiled.
- Requires Python 3.11 or newer.
- Package metadata and build requirements moved to `pyproject.toml`; the
  version is derived from git tags.
- `TrafficMatrix.to_csv` writes one row per epoch (`n * n` columns) for
  multi-epoch matrices, as documented, instead of one row per node pair.

### Fixed

- `hmc_sample` looped forever on degenerate constraints that every candidate
  sample violates; it now raises `TMgenException` after 1000 consecutive
  rejections.
- `modulated_gravity_tm` ignored `spatial_variance` and `temporal_variance`. They
  now set the variance of the truncated-normal node shares and of the total
  traffic around its cycle, as in the paper. Both default to 0 (the previous
  effective behavior), and negative values raise `TMgenException`.
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
