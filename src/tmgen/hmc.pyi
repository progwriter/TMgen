import numpy
from numpy.typing import ArrayLike, NDArray

def hmc_sample(
    f: ArrayLike,
    g: ArrayLike,
    m: ArrayLike,
    mu_r: ArrayLike,
    initial_x: ArrayLike,
    num_samples: int,
    covariance: bool = True,
    *,
    rng: numpy.random.Generator | int | None = None,
) -> NDArray[numpy.float64]: ...
