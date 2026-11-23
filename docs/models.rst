Models
======

All models live in :py:mod:`tmgen.models` and return a
:py:class:`~tmgen.TrafficMatrix`. This page explains what each one does and when
to use it. The reference for every parameter is in :doc:`api`.

Choosing a model
----------------

.. list-table::
   :header-rows: 1
   :widths: 28 22 14 36

   * - Model
     - Structure
     - Time
     - Use it for
   * - :py:func:`~tmgen.models.gravity_tm`
     - Product of node sizes
     - One epoch
     - A deterministic baseline from known node sizes
   * - :py:func:`~tmgen.models.random_gravity_tm`
     - Product of random weights
     - One epoch
     - A realistic, rank-1 matrix with a given total
   * - :py:func:`~tmgen.models.modulated_gravity_tm`
     - Random gravity times a cycle
     - Daily cycle, optional noise
     - A time series of gravity-shaped matrices
   * - :py:func:`~tmgen.models.spike_tm`
     - A few large flows
     - Any
     - Anomalies, flash crowds, peak load
   * - :py:func:`~tmgen.models.uniform_tm`
     - Independent entries
     - Any
     - Unstructured load with bounds
   * - :py:func:`~tmgen.models.exp_tm`
     - Independent entries
     - Any
     - Unstructured load with a given mean
   * - :py:func:`~tmgen.models.lognormal_tm`
     - Independent entries
     - Any
     - Heavy-tailed unstructured load
   * - :py:func:`~tmgen.models.exact_tm`
     - Constant entries
     - Any
     - Tests with exactly known values

Most of these come from the maximum entropy view in Tune and Roughan
[#tune2015]_: a model should assume nothing beyond the constraints you give it.
With only a mean per entry the maximum entropy distribution is the exponential
distribution; with a mean and a variance it is a normal distribution truncated
at zero; with only bounds it is uniform. Constraining the total traffic of each
node (rather than each entry) leads to the *gravity* form, in which an entry is
the product of an ingress weight and an egress weight.

Independent entries
-------------------

:py:func:`~tmgen.models.uniform_tm`, :py:func:`~tmgen.models.exp_tm` and
:py:func:`~tmgen.models.lognormal_tm` draw every entry independently. They have
no spatial structure, which makes them useful as a worst case or a noise
floor.

>>> from tmgen.models import uniform_tm, exp_tm, lognormal_tm, exact_tm
>>> tm = uniform_tm(num_nodes=3, low=10, high=20, rng=0)
>>> bool((tm.matrix >= 10).all() and (tm.matrix <= 20).all())
True
>>> tm = exp_tm(num_nodes=3, mean_entry=100, num_epochs=5, rng=0)
>>> tm.shape
(3, 3, 5)

The lognormal model is parametrized by the ``mu`` and ``sigma`` of the
*underlying* normal distribution, not by the mean of the samples, which is
``exp(mu + sigma**2 / 2)``:

>>> tm = lognormal_tm(num_nodes=3, mu=1.0, sigma=0.5, rng=0)

:py:func:`~tmgen.models.exact_tm` fills every entry with the same value, for
tests where you need to know the answer:

>>> exact_tm(num_nodes=2, value=5).matrix[:, :, 0].tolist()
[[5.0, 5.0], [5.0, 5.0]]

Gravity
-------

In a gravity model the traffic from ingress ``i`` to egress ``j`` is
proportional to the product of the sizes of the two nodes. Give
:py:func:`~tmgen.models.gravity_tm` the sizes ("populations") and the total
traffic of the network; the result has no randomness and one epoch:

>>> from tmgen.models import gravity_tm
>>> tm = gravity_tm(populations=[10, 20, 30], total_traffic=600)
>>> float(tm.matrix.sum())
600.0
>>> tm.matrix[:, :, 0].round(1).tolist()
[[16.7, 33.3, 50.0], [33.3, 66.7, 100.0], [50.0, 100.0, 150.0]]

Random gravity
--------------

:py:func:`~tmgen.models.random_gravity_tm` draws the node weights at random, so
you get a gravity-shaped matrix without having to know node sizes. Following the
paper, ingress and egress weights are independent exponential random variables
(the maximum entropy choice, and the model of [#roughan2005]_), so the matrix is
not symmetric and has rank 1.

``total_traffic`` is the total *in expectation*: any one draw is a little more or
less. Pass ``normalize=True`` for an exact total:

>>> from tmgen.models import random_gravity_tm
>>> tm = random_gravity_tm(num_nodes=5, total_traffic=1000, normalize=True, rng=1)
>>> round(float(tm.matrix.sum()), 6)
1000.0

``symmetric=True`` uses the same weights for ingress and egress, which makes the
matrix symmetric (and its expected total ``total_traffic * (1 + 1/num_nodes)``).

Daily cycles: modulated gravity
-------------------------------

Real traffic rises and falls over the day.
:py:func:`~tmgen.models.modulated_gravity_tm` scales a random gravity matrix by a
daily cycle, one matrix per epoch:

>>> from tmgen.models import modulated_gravity_tm
>>> tm = modulated_gravity_tm(num_nodes=8, num_epochs=48, total_traffic=1000, rng=3)
>>> tm.shape
(8, 8, 48)

With the defaults the total traffic of the epochs swings between 0.5 and 1.5
times ``total_traffic`` once per 24 epochs (``peak_to_mean=1.5``, a symmetric
cycle, as in Fig. 2 of the paper). The cycle starts at ``total_traffic`` and
rises. Change its shape with ``peak_to_mean`` (at most 2 for a symmetric cycle),
``trough_to_mean`` (the lowest value divided by the mean, for a cycle that dips
less than it peaks) and ``diurnal_freq`` (cycles per epoch).

By default the total of every epoch follows the cycle exactly and the share of
each node is fixed. Two variances add the randomness of the paper's model. Both
are in squared units of traffic:

>>> noisy = modulated_gravity_tm(
...     num_nodes=8, num_epochs=48, total_traffic=1000,
...     spatial_variance=100, temporal_variance=400, rng=3)

``temporal_variance`` makes the total traffic wobble around the cycle (a
standard deviation of ``sqrt(temporal_variance)`` traffic units), and
``spatial_variance`` varies each node's share. Both draw from normal
distributions truncated at zero. The useful values depend on your scale: a
spatial variance that is large compared to ``total_traffic / num_nodes``
inflates the total.

Spikes
------

:py:func:`~tmgen.models.spike_tm` creates a sparse matrix: exactly
``num_spikes`` entries of the *whole* array (all ingress, egress and epoch
combinations) are non-zero, chosen uniformly at random, each with an
exponentially distributed volume of mean ``mean_spike``. Spikes can cluster in
some epochs and be absent from others. Use it for anomalies and flash crowds,
or add it to a smooth model:

>>> from tmgen.models import spike_tm
>>> tm = spike_tm(num_nodes=5, num_spikes=3, mean_spike=200, num_epochs=4, rng=0)
>>> int((tm.matrix > 0).sum())
3

Matrices of the same shape can be added with ``+``, so a background model plus
spikes is one line:

>>> background = exp_tm(num_nodes=5, mean_entry=10, num_epochs=4, rng=1)
>>> combined = background + tm

Combining a spatial model with a temporal pattern
-------------------------------------------------

:py:meth:`TrafficMatrix.modulate <tmgen.TrafficMatrix.modulate>` turns any
single-epoch matrix into a time series by scaling it with a profile, one
multiplier per epoch. :py:func:`~tmgen.models.diurnal_profile` generates a daily
cycle; any array of non-negative numbers works as well.

>>> from tmgen.models import diurnal_profile
>>> spatial = gravity_tm(populations=[1, 2, 3], total_traffic=6)
>>> profile = diurnal_profile(num_epochs=24, mean=1.0)
>>> day = spatial.modulate(profile)
>>> day.shape
(3, 3, 24)

To follow a measured series of total traffic, divide it by the total of your
spatial matrix:

>>> import numpy
>>> measured_totals = numpy.array([40.0, 55.0, 70.0, 60.0])
>>> series = spatial.modulate(measured_totals / spatial.matrix.sum())
>>> numpy.allclose(series.matrix.sum(axis=(0, 1)), measured_totals)
True

Not available yet
-----------------

The paper also describes models that TMgen does not expose yet: the
non-stationary conditionally independent model (NCIM), shifted time zones, a
Fourier-constrained temporal model, traffic classes, and the general maximum
entropy model with arbitrary constraints. The sampler they rely on is
available as :py:func:`~tmgen.hmc.hmc_sample`, see :doc:`sampling`.

References
----------

.. [#tune2015] Paul Tune and Matthew Roughan. *Spatiotemporal Traffic Matrix
   Synthesis*. ACM SIGCOMM 2015, pages 579-592.
   https://doi.org/10.1145/2785956.2787471

.. [#roughan2005] Matthew Roughan. *Simplifying the Synthesis of Internet
   Traffic Matrices*. ACM SIGCOMM Computer Communication Review, October 2005.
   http://ccr.sigcomm.org/archive/2005/october/p93-roughan.pdf
