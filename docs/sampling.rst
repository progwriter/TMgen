Sampling truncated Gaussians
============================

The maximum entropy models of the paper that go beyond independent exponentials
are truncated multivariate Gaussians: a Gaussian distribution restricted to the
points that satisfy linear constraints, such as non-negative traffic. TMgen
includes the sampler for them, :py:func:`tmgen.hmc.hmc_sample`, an exact
Hamiltonian Monte Carlo method by Pakman and Paninski [#pakman2012]_. It
returns *exact* draws, with no tuning and no burn-in, because the particle
follows the Gaussian dynamics exactly and only bounces off the constraint walls.

The distribution is a Gaussian with mean ``mu_r`` and covariance ``m``,
restricted to ``f @ x + g >= 0``. For example, a standard 2-dimensional
Gaussian restricted to the positive quadrant (``f`` is the identity and ``g``
is zero):

>>> import numpy
>>> from tmgen.hmc import hmc_sample
>>> samples = hmc_sample(
...     f=numpy.eye(2),
...     g=numpy.zeros(2),
...     m=numpy.eye(2),
...     mu_r=numpy.zeros(2),
...     initial_x=numpy.ones(2),
...     num_samples=2000,
...     rng=0,
... )
>>> samples.shape            # one sample per column
(2, 2000)
>>> bool((samples > 0).all())
True

The mean of a half-normal variable is ``sqrt(2 / pi)``, about 0.80:

>>> bool(abs(samples.mean(axis=1) - numpy.sqrt(2 / numpy.pi)).max() < 0.1)
True

Arguments
---------

``f``, ``g``
   The constraints ``f @ x + g >= 0``, as an ``m x d`` matrix and a vector of
   length ``m``.
``m``, ``mu_r``
   A symmetric positive definite ``d x d`` matrix and a vector of length ``d``.
   By default ``m`` is the covariance and ``mu_r`` the mean. With
   ``covariance=False``, ``m`` is the precision matrix and ``mu_r`` the
   linear term ``r`` of the log-density ``-x'mx/2 + r'x``.
``initial_x``
   A starting point that satisfies the constraints. The first sample returned
   is this point; subsequent samples are the draws.
``num_samples``
   The number of columns of the result, at least 1.
``rng``
   A seed, a :py:class:`numpy.random.Generator`, or ``None`` for the global
   NumPy state, as in :ref:`reproducibility`.

Invalid arguments raise :py:class:`~tmgen.TMgenException`: inconsistent shapes,
an initial point that violates the constraints, or constraints that no
candidate can satisfy (the sampler gives up after 1000 consecutive rejections
instead of looping forever).

Performance notes
-----------------

After every bounce the sampler updates two products with the Gram matrix
``f @ f.T`` in ``O(m)`` instead of recomputing them in ``O(m d)``, which makes it
several times faster than a straightforward implementation. The Gram matrix
needs ``m * m`` numbers; above 256 MiB the sampler falls back to recomputing the
products, which is slower but needs no extra memory.

.. [#pakman2012] Ari Pakman and Liam Paninski. *Exact Hamiltonian Monte Carlo for
   Truncated Multivariate Gaussians*. arXiv:1208.4118.
   https://arxiv.org/abs/1208.4118
