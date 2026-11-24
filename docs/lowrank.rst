Learning from measurements
==========================

The other models create traffic from parameters. :py:class:`tmgen.lowrank.LowRankModel`
learns from a series of real matrices instead, and samples new series that
share their main patterns [#lakhina2004]_. It is principal component analysis of
the matrices: a mean matrix plus a few spatial patterns (components), each with
a weight that varies over time as a first-order autoregressive process.
Slowly changing patterns, such as a daily cycle, then change slowly in the
samples too.

Here a day of traffic stands in for a measurement; use your own data in its
place (for example with :py:meth:`~tmgen.TrafficMatrix.from_csv`):

>>> from tmgen.lowrank import LowRankModel
>>> from tmgen.models import modulated_gravity_tm
>>> measured = modulated_gravity_tm(num_nodes=6, num_epochs=96, total_traffic=100, rng=0)
>>> model = LowRankModel.fit(measured)
>>> model.rank, round(model.explained_variance, 3)
(1, 1.0)
>>> model.sample(num_epochs=48, rng=1).shape
(6, 6, 48)

By default the fewest components that explain 95% of the variance are kept;
pass ``rank`` or ``explained_variance`` to choose. Samples are clipped at zero.
The model captures linear structure and short-range correlation only: it does
not reproduce sharp spikes or a precise cycle, and a series with few epochs
gives a rough fit.

.. [#lakhina2004] Anukool Lakhina, Konstantina Papagiannaki, Mark Crovella,
   Christophe Diot, Eric Kolaczyk and Nina Taft. *Structural analysis of
   network traffic flows*. ACM SIGMETRICS 2004.
