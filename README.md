# TMgen: Traffic Matrix generation tool

TMgen is a tool for generating spatial, temporal, and spatio-temporal traffic
matrices. Generation is based on the Max Entropy model described in
**Spatiotemporal Traffic Matrix Synthesis** by Paul Tune and Matthew Roughan,
published in ACM SIGCOMM 2015.

Other, simple models (e.g., uniform, gravity) are also implemented for
convenience.

## Supported TM models

- [x] Random Gravity Model
- [x] Modulated Gravity Model
- [x] Spike Model
- [x] Gravity Model
- [x] Uniform, Exponential, Lognormal and Exact (constant) Models

The Max Entropy and non-stationary conditionally independent models from the
paper are not yet exposed; only the underlying truncated-Gaussian HMC sampler
(`tmgen.hmc.hmc_sample`) is implemented.

## Installation

Requires Python 3.11+ and a C compiler (numpy and cython are fetched
automatically as build dependencies).

```
pip install .
```

## Example Usage

See http://tmgen.readthedocs.io/ for the docs.

## Big Thanks
To Paul Tune, Matthew Roughan, and Ari Pakman for their work in this space and
for making their code available. This code is an adaptation of their Matlab
versions [MaxEnt](https://github.com/ptuls/MaxEntTM) and
[HMC](https://github.com/aripakman/hmc-tmg)
