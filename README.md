# TMgen: Traffic Matrix generation tool

TMgen is a tool for generating spatial, temporal, and spatio-temporal traffic
matrices. Generation is based on the Max Entropy model described in
**Spatiotemporal Traffic Matrix Synthesis** by Paul Tune and Matthew Roughan,
published in ACM SIGCOMM 2015.

Other, simple models (e.g., uniform, gravity) are also implemented for
convenience.

## Supported TM models

*This is work in progress*

- [x] Random Gravity Model
- [x] Modulated Gravity Model
- [ ] Non-stationary Conditionally Independent Model
- [x] Spike Model
- [x] Gravity Model
- [x] Uniform
- [ ] Log-normal Model
- [ ] Poisson Model

## Installation

1. Install numpy and cython. (For example using `pip install numpy cython`)
2. Run ``pip install .``

## Example Usage

TMgen defines a ``TrafficMatrix`` object that is returned by all generator
functions. Internally it contains a 3-d numpy array which contains volume of
traffic between origin-destination pairs for different time epochs. For TM models
that do not have a time component, the third dimension is of size 1.

**More on model generation coming soon**

## Big Thanks
To Paul Tune, Matthew Roughan, and Ari Pakman for their work in this space and
for making their code available. This code is an adaptation of their Matlab
versions [MaxEnt](https://github.com/ptuls/MaxEntTM) and
[HMC](https://github.com/aripakman/hmc-tmg)
