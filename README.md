# TMgen: traffic matrix generation

[![PyPI](https://img.shields.io/pypi/v/tmgen)](https://pypi.org/project/tmgen/)
[![Python versions](https://img.shields.io/pypi/pyversions/tmgen)](https://pypi.org/project/tmgen/)
[![CI](https://github.com/progwriter/TMgen/actions/workflows/ci.yml/badge.svg?branch=master)](https://github.com/progwriter/TMgen/actions/workflows/ci.yml)
[![Docs](https://readthedocs.org/projects/tmgen/badge/?version=latest)](https://tmgen.readthedocs.io/)
[![Coverage](https://codecov.io/gh/progwriter/TMgen/graph/badge.svg)](https://codecov.io/gh/progwriter/TMgen)
[![License](https://img.shields.io/pypi/l/tmgen)](https://github.com/progwriter/TMgen/blob/master/LICENSE)

TMgen generates synthetic network **traffic matrices**: how much traffic flows
from each node of a network to each other node, optionally over time. Use it to
test routing, traffic engineering and topology design code when you do not have
(or cannot share) real measurements.

It is a collection of synthesis techniques from the research literature, each
credited to its paper below, next to the simple models people usually start
with. They all return the same small `TrafficMatrix` object, so you can swap one
technique for another, combine them, and check the result against real traffic.

```python
from tmgen.models import modulated_gravity_tm

# 10 nodes, one day of hourly matrices; the total follows a daily cycle around 1000
tm = modulated_gravity_tm(num_nodes=10, num_epochs=24, total_traffic=1000, rng=42)

tm.shape  # (10, 10, 24): ingress x egress x epoch
tm.at_time(12)  # the 10 x 10 matrix at noon
tm.between(0, 3, "max")  # the largest volume from node 0 to node 3
tm.zero_diagonal().save("day.npz")
```

## Techniques

| Technique | What it produces | Functions | Source |
|---|---|---|---|
| Gravity models | Traffic proportional to the product of random node weights; the maximum entropy model for a given total | `gravity_tm`, `random_gravity_tm` | Roughan, [*Simplifying the synthesis of Internet traffic matrices*](https://doi.org/10.1145/1096536.1096551), CCR 2005 |
| Maximum entropy spatiotemporal models | Gravity modulated by daily and Fourier cycles; independent epochs with per-node traffic and time zones (NCIM); traffic classes; spikes; independent entries | `modulated_gravity_tm`, `diurnal_profile`, `fourier_profile`, `ncim_tm`, `time_zone_profiles`, `traffic_class_tm`, `spike_tm`, `exp_tm`, `uniform_tm` | Tune and Roughan, [*Spatiotemporal traffic matrix synthesis*](https://doi.org/10.1145/2785956.2787471), SIGCOMM 2015 |
| Exact truncated Gaussian sampler | Draws for the paper's constrained models | `tmgen.hmc.hmc_sample` | Pakman and Paninski, [*Exact Hamiltonian Monte Carlo for truncated multivariate Gaussians*](https://arxiv.org/abs/1208.4118), 2014 |
| Elephants and mice | Lognormal large flows, gamma small flows, and fluctuations that grow with the mean | `elephants_and_mice_tm`, `fluctuating_tm`, `lognormal_tm` | Nucci, Sridharan and Taft, [*The problem of synthetically generating IP traffic matrices*](https://doi.org/10.1145/1070873.1070876), CCR 2005 |
| Flow-level shot noise | Poisson flow arrivals per node pair around given mean rates, with tunable burstiness | `shot_noise_tm` | Barakat et al., [*Modeling Internet backbone traffic at the flow level*](https://doi.org/10.1109/TSP.2003.814521), IEEE TSP 2003 |
| Data centre hot nodes and racks | A fraction of hot nodes and a set share of cross-rack traffic | `datacenter_tm` | Parsonson, Benjamin and Zervas, [*Traffic generation for benchmarking data centre networks*](https://arxiv.org/abs/2107.01398), 2022 |
| Low-rank model of measurements | New series sharing the main patterns of a real one (PCA with autoregressive weights) | `tmgen.lowrank.LowRankModel` | Lakhina et al., [*Structural analysis of network traffic flows*](https://doi.org/10.1145/1012888.1005697), SIGMETRICS 2004 |
| Test fixture | Every entry the same | `exact_tm` | |

Any model of one epoch can be turned into a time series with
`TrafficMatrix.modulate` and a profile (`diurnal_profile`, `fourier_profile`),
and series add up (`large + spikes + noise`). The paper's general maximum
entropy model with arbitrary constraints is not exposed yet.

## Checking against real traffic

`tmgen.datasets.load_abilene` downloads the public
[Abilene](https://www.cs.utexas.edu/~yzhang/research/AbileneTM/) matrices
(Yin Zhang, 2004), and `tmgen.metrics` measures how close a generated series is
to a real one: the distribution of entries, how traffic spreads over node pairs,
and how it changes over time.

## Installation

TMgen needs Python 3.11 or newer.

### From PyPI

```
pip install tmgen
```

Prebuilt wheels are published for Linux, macOS and Windows, so no compiler is
needed. For the plotting helpers, install the extra:
`pip install "tmgen[plotting]"`.

### From source

Use this for the latest unreleased changes, or to try a modification. The HMC
sampler is a small Cython extension, so you need a C compiler (Xcode command
line tools on macOS, `build-essential` on Debian and Ubuntu, the Visual Studio
Build Tools on Windows). NumPy, Cython and the other build requirements are
fetched automatically.

```
git clone https://github.com/progwriter/TMgen
cd TMgen
pip install .                 # or ".[plotting]" for the plotting helpers
```

The version number comes from the git tags, so clone the repository (a plain
source archive of a release also works) rather than downloading a snapshot of a
branch. To work on TMgen itself, see [DEVELOPMENT.md](https://github.com/progwriter/TMgen/blob/master/DEVELOPMENT.md) for an
editable install with the development tools.

## Documentation

The full documentation, with a guide to every model, file formats, sampling and
plotting, is at <https://tmgen.readthedocs.io/>. Upgrading from 0.1? See the
[changelog](https://github.com/progwriter/TMgen/blob/master/CHANGELOG.md) and the upgrade notes in the docs.

## Big thanks

To Paul Tune, Matthew Roughan, and Ari Pakman for their work in this space and
for making their code available. This code is an adaptation of their Matlab
versions [MaxEnt](https://github.com/ptuls/MaxEntTM) and
[HMC](https://github.com/aripakman/hmc-tmg).

## Development

Contributions are welcome. See [DEVELOPMENT.md](https://github.com/progwriter/TMgen/blob/master/DEVELOPMENT.md) for how to set
up a development environment, run the checks and submit changes.
