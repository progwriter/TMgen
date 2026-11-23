# TMgen: traffic matrix generation

[![PyPI](https://img.shields.io/pypi/v/tmgen)](https://pypi.org/project/tmgen/)
[![Docs](https://readthedocs.org/projects/tmgen/badge/?version=latest)](https://tmgen.readthedocs.io/)

TMgen generates synthetic network **traffic matrices**: how much traffic flows
from each node of a network to each other node, optionally over time. Use it to
test routing, traffic engineering and topology design code when you do not have
(or cannot share) real measurements.

It implements the maximum entropy models of **Spatiotemporal Traffic Matrix
Synthesis** by Paul Tune and Matthew Roughan (ACM SIGCOMM 2015) next to the
simple models people usually start with, and gives you a small `TrafficMatrix`
object to work with the results.

```python
from tmgen.models import modulated_gravity_tm

# 10 nodes, one day of hourly matrices; the total follows a daily cycle around 1000
tm = modulated_gravity_tm(num_nodes=10, num_epochs=24, total_traffic=1000, rng=42)

tm.shape  # (10, 10, 24): ingress x egress x epoch
tm.at_time(12)  # the 10 x 10 matrix at noon
tm.between(0, 3, "max")  # the largest volume from node 0 to node 3
tm.zero_diagonal().save("day.npz")
```

## Models

| Model | Function | What it produces |
|---|---|---|
| Gravity | `gravity_tm` | Traffic proportional to the product of node sizes, no randomness |
| Random gravity | `random_gravity_tm` | Gravity with random node weights (the maximum entropy model for a given total) |
| Modulated gravity | `modulated_gravity_tm` | Random gravity scaled over time by a daily cycle, with optional noise |
| Spike | `spike_tm` | A few large, randomly placed flows |
| Uniform, exponential, lognormal | `uniform_tm`, `exp_tm`, `lognormal_tm` | Independent random entries |
| Exact | `exact_tm` | Every entry the same, for tests |

Any spatial model can be combined with any temporal pattern using
`TrafficMatrix.modulate` and `diurnal_profile`. The sampler behind the paper's
models, an exact Hamiltonian Monte Carlo for truncated Gaussians, is available
as `tmgen.hmc.hmc_sample`. The paper's non-stationary conditionally
independent model and the general maximum entropy model with constraints are
not exposed yet.

## Installation

```
pip install tmgen
```

Prebuilt wheels are published for common platforms. Building from source needs
Python 3.11+ and a C compiler. For the plotting helpers, install the extra:
`pip install "tmgen[plotting]"`.

## Documentation

The full documentation, with a guide to every model, file formats, sampling and
plotting, is at <https://tmgen.readthedocs.io/>. Upgrading from 0.1? See the
[changelog](CHANGELOG.md) and the upgrade notes in the docs.

## Big thanks

To Paul Tune, Matthew Roughan, and Ari Pakman for their work in this space and
for making their code available. This code is an adaptation of their Matlab
versions [MaxEnt](https://github.com/ptuls/MaxEntTM) and
[HMC](https://github.com/aripakman/hmc-tmg).

## Development

Contributions are welcome. See [DEVELOPMENT.md](DEVELOPMENT.md) for how to set
up a development environment, run the checks and submit changes.
