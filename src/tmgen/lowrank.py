"""
A data-driven generator: learn the main patterns of a measured series of
traffic matrices and sample new series that share them.

Real traffic matrices are close to low rank: a few spatial patterns, each
following a cycle over time, explain most of the variation (Lakhina et al.,
SIGCOMM 2004; the same dimension reduction precedes the diffusion model of
Zhang et al., *Traffic matrix estimation with denoising diffusion
probabilistic models*, 2024). :py:class:`LowRankModel` is the simplest such
generator: principal component analysis (PCA) of the matrices, with a
first-order autoregressive process for the strength of each pattern over time.
"""

from dataclasses import dataclass

import numpy
from numpy.typing import NDArray

from tmgen._validation import count, number, resolve_rng
from tmgen.exceptions import TMgenException
from tmgen.models import RngLike
from tmgen.tm import TrafficMatrix

_MAX_AUTOCORRELATION = 0.99


@dataclass(frozen=True, eq=False)
class LowRankModel:
    """
    A fitted low-rank model of a series of traffic matrices. Create one with
    :py:meth:`fit` and draw new series with :py:meth:`sample`.

    A matrix is the *mean* plus a weighted sum of *components*. The weight of
    component :math:`k` is a Gaussian process with standard deviation
    ``scale[k]`` and lag-1 autocorrelation ``autocorrelation[k]``, so slowly
    changing patterns (such as a daily cycle) change slowly in the samples
    too. Samples are clipped at zero, since traffic cannot be negative.
    """

    mean: NDArray[numpy.float64]
    """The mean matrix, shape ``(num_nodes, num_nodes)``."""
    components: NDArray[numpy.float64]
    """The principal components, shape ``(rank, num_nodes, num_nodes)``."""
    scale: NDArray[numpy.float64]
    """The standard deviation of each component's weight, shape ``(rank,)``."""
    autocorrelation: NDArray[numpy.float64]
    """The lag-1 autocorrelation of each component's weight, shape ``(rank,)``."""
    explained_variance: float
    """The fraction of the variance of the fitted series the components explain."""

    @property
    def num_nodes(self) -> int:
        return int(self.mean.shape[0])

    @property
    def rank(self) -> int:
        return int(self.scale.size)

    @classmethod
    def fit(
        cls,
        tm: TrafficMatrix,
        rank: int | None = None,
        *,
        explained_variance: float = 0.95,
    ) -> "LowRankModel":
        """
        Fit the model to a series of matrices.

        :param tm: the series to learn from, with at least three epochs
        :param rank: the number of components to keep. By default the fewest
            that explain *explained_variance* of the variance
        :param explained_variance: the fraction of the variance to explain when
            *rank* is not given, in ``(0, 1]``
        :return: the fitted model
        """
        if tm.num_epochs < 3:
            raise TMgenException("fit needs a series of at least 3 epochs")
        num_nodes = tm.num_nodes
        series = tm.matrix.reshape(num_nodes * num_nodes, tm.num_epochs).T
        mean = series.mean(axis=0)
        _, singular, components = numpy.linalg.svd(series - mean, full_matrices=False)
        variance = singular**2
        total = variance.sum()
        if total == 0:
            raise TMgenException(
                "the matrices never change, so there is nothing to fit"
            )
        explained = numpy.cumsum(variance) / total
        if rank is None:
            target = number(
                explained_variance, "explained_variance", positive=True, maximum=1
            )
            rank = int(numpy.searchsorted(explained, target - 1e-12)) + 1
        else:
            rank = count(rank, "rank")
            if rank > singular.size:
                raise TMgenException(
                    f"rank can be at most {singular.size} for {tm.num_epochs} epochs"
                )
        components = components[:rank]
        weights = (series - mean) @ components.T
        scale = weights.std(axis=0, ddof=1)
        previous, current = weights[:-1], weights[1:]
        energy = (previous**2).sum(axis=0)
        correlation = numpy.divide(
            (previous * current).sum(axis=0),
            energy,
            out=numpy.zeros(rank),
            where=energy > 0,
        )
        return cls(
            mean=mean.reshape(num_nodes, num_nodes),
            components=components.reshape(rank, num_nodes, num_nodes),
            scale=scale,
            autocorrelation=numpy.clip(
                correlation, -_MAX_AUTOCORRELATION, _MAX_AUTOCORRELATION
            ),
            explained_variance=float(explained[rank - 1]),
        )

    def sample(self, num_epochs: int, *, rng: RngLike = None) -> TrafficMatrix:
        """
        Draw a new series of matrices.

        :param num_epochs: number of epochs to generate
        :param rng: seed or :py:class:`numpy.random.Generator`; see
            :py:data:`~tmgen.models.RngLike`
        :return: a new :py:class:`~tmgen.TrafficMatrix`
        """
        num_epochs = count(num_epochs, "num_epochs")
        gen = resolve_rng(rng)
        noise = gen.standard_normal((num_epochs, self.rank))
        innovation = numpy.sqrt(1 - self.autocorrelation**2)
        weights = numpy.empty_like(noise)
        weights[0] = noise[0]
        for epoch in range(1, num_epochs):
            weights[epoch] = (
                self.autocorrelation * weights[epoch - 1] + innovation * noise[epoch]
            )
        flat = self.components.reshape(self.rank, -1)
        series = self.mean.reshape(1, -1) + (weights * self.scale) @ flat
        matrix = numpy.clip(series, 0, None).T.reshape(
            self.num_nodes, self.num_nodes, num_epochs
        )
        return TrafficMatrix._from_owned(numpy.ascontiguousarray(matrix))
