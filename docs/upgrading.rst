Upgrading from 0.1
==================

This page lists what you may need to change. The complete list is in the
`changelog <https://github.com/progwriter/TMgen/blob/master/CHANGELOG.md>`_.

Old keyword names keep working for one release but emit a
``DeprecationWarning``; run your tests with ``-W error::DeprecationWarning`` to
find them.

Renamed
-------

================================================  ====================================
Old                                               New
================================================  ====================================
``modulated_gravity_tm(num_tms=...)``             ``num_epochs``
``modulated_gravity_tm(mean_traffic=...)``        ``total_traffic``
``random_gravity_tm(mean_traffic=...)``           ``total_traffic``
``exp_tm(mean_traffic=...)``                      ``mean_entry``
``lognormal_tm(mean_traffic=...)``                ``mu``
``exact_tm(val=...)``                             ``value``
``modulated_gravity_tm(pm_ratio=...)``            ``peak_to_mean``
``modulated_gravity_tm(t_ratio=...)``             ``trough_to_mean``
``tm.between(o, d, modestr)``                     ``tm.between(ingress, egress, agg)``
``tm.num_nodes()``, ``tm.num_epochs()``           ``tm.num_nodes``, ``tm.num_epochs``
``to_pickle``, ``from_pickle``                    ``save``, ``load`` (``.npz``)
================================================  ====================================

``between`` takes ``agg=None`` (the series), ``"min"``, ``"max"`` or ``"mean"``.
The old string modes, including ``"all"``, ``"*"`` and the misspelled
``"mininum"``, still work with a warning.

Behavior changes
----------------

These change results, so check your code and your expectations:

* **The matrix is read-only and validated.** ``tm.matrix[...] = value`` and
  ``tm.matrix = ...`` fail; build a new ``TrafficMatrix`` from a copy. Negative,
  NaN or infinite volumes and wrongly shaped arrays are rejected.
* **Random draws changed.** The models use NumPy's ``Generator`` machinery and
  the ``rng`` argument. The same seed no longer gives the same matrix as 0.1.
* **Invalid arguments raise** :py:class:`~tmgen.TMgenException` (a
  ``ValueError``). ``uniform_tm`` raises when ``low > high`` instead of swapping
  the bounds.
* **``modulated_gravity_tm`` was fixed.** The total of each epoch used to be off
  by a factor of ``mean_traffic / num_nodes``, the "trough" of the daily cycle
  was above the mean, and ``spatial_variance`` and ``temporal_variance`` were
  ignored. The default cycle is now symmetric (0.5 to 1.5 times the total for
  ``peak_to_mean=1.5``), ingress and egress shares are independent, and both
  variances (default 0) work as in the paper.
* **``random_gravity_tm`` follows the paper.** Weights are independent
  exponentials (they used to be normalized uniforms shared by rows and columns),
  so matrices are more skewed and not symmetric, and the total is
  ``total_traffic`` in expectation. Use ``normalize=True`` for an exact total
  and ``symmetric=True`` for symmetric matrices.
* **``spike_tm`` follows the paper.** ``num_spikes`` is the total over the whole
  array, not per epoch.
* **``to_csv`` writes a long format** with a header, and ``from_csv`` reads it.
* **``tmditg`` reads ``.npz`` files** instead of pickles.
* **No ``.pxd`` files** are shipped, so ``cimport tmgen...`` no longer works.
