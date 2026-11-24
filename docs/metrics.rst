Judging realism
===============

How do you know whether a generated series resembles real traffic?
:py:mod:`tmgen.metrics` measures it in three separate ways, because a generator
can match one and miss another: the distribution of entry values (*marginal*),
how traffic spreads over node pairs (*spatial*) and how it changes over time
(*temporal*).

Every comparison is a distance, so 0 is a perfect match. Here a low-rank model
(:doc:`lowrank`) is fitted to a day of traffic, and a fresh sample is compared
with it, and with unstructured noise of the same size:

>>> from tmgen.lowrank import LowRankModel
>>> from tmgen.metrics import spatial_comparison, temporal_comparison
>>> from tmgen.models import exp_tm, modulated_gravity_tm
>>> real = modulated_gravity_tm(num_nodes=6, num_epochs=288, total_traffic=100,
...                             diurnal_freq=1 / 144, rng=1)
>>> sample = LowRankModel.fit(real).sample(num_epochs=288, rng=2)
>>> noise = exp_tm(num_nodes=6, mean_entry=100 / 36, num_epochs=288, rng=3)
>>> sorted(spatial_comparison(real, sample))
['gravity', 'heavy_hitters', 'ks', 'spectrum', 'wasserstein']
>>> spatial_comparison(real, sample)["spectrum"] < spatial_comparison(real, noise)["spectrum"]
True
>>> sorted(temporal_comparison(real, sample))
['autocorrelation', 'variability']
>>> temporal_comparison(real, sample)["autocorrelation"] < temporal_comparison(real, noise)["autocorrelation"]
True

The functions behind the comparisons can be used on their own:

.. list-table::
   :header-rows: 1

   * - Function
     - What it measures
   * - :py:func:`~tmgen.metrics.ks_distance`
     - largest gap between the distributions of entry values
   * - :py:func:`~tmgen.metrics.wasserstein_distance`
     - average distance one value moves to match the other
   * - :py:func:`~tmgen.metrics.singular_spectrum`
     - share of the variation each spatial pattern explains
   * - :py:func:`~tmgen.metrics.gravity_error`
     - distance of the mean matrix from a gravity model
   * - :py:func:`~tmgen.metrics.heavy_hitter_share`
     - share of traffic on the busiest node pairs
   * - :py:func:`~tmgen.metrics.autocorrelation`
     - how entries correlate with themselves over time

With the real Abilene measurements (:doc:`datasets`) as the reference, this
tells you how closely each model of :doc:`models` comes to a real network.
