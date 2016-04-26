# TMgen: Traffic Matrix generation tool

TMgen is a tool for generating spatial, temporal, and spatio-temporal traffic matrices.
Generation is based on the Max Entropy model described in
**Spatiotemporal Traffic Matrix Synthesis** by Paul Tune and Matthew Roughan, published in ACM SIGCOMM 2015.

Other, simple models (e.g., uniform, gravity) are also implemented for convenience.

## Supported TM models

*This is work in progress*

#### Max Entropy Models
-[x] Random Gravity Model (RGM)
-[x] Modulated Gravity Model (MGM)
-[ ] Non-stationary Conditionally Independent Model (NCIM)
-[ ] Spike Model
#### Other models/distributions
-[x] Gravity Model
-[x] Uniform
-[ ] Log-normal Model
-[ ] Poisson Model

## Example Usage

Coming soon

<!-- ## Full documentation -->

## Big Thanks
To Paul Tune, Matthew Roughan, and Ari Pakman for their work in this space and for making their code available.
This code is an adaptation of their Matlab versions [MaxEnt](https://github.com/ptuls/MaxEntTM) and
[HMC](https://github.com/aripakman/hmc-tmg)
