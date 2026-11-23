Concepts
========

What a traffic matrix is
------------------------

A traffic matrix records how much traffic flows between the nodes of a network.
TMgen stores it as a 3-dimensional array with the shape ``(n, n, T)``:

* the first axis is the **ingress** node (where traffic enters the network),
* the second axis is the **egress** node (where it leaves),
* the third axis is the **epoch**, a measurement interval such as five minutes
  or an hour.

An **entry** is the volume from one ingress to one egress node during one
epoch. TMgen does not fix the unit: the numbers you pass in (bytes, packets per
second, Gbps) are the unit of the numbers you get out. Models with no notion of
time produce a single epoch.

Vocabulary
----------

====================  ======================================================
Term                  Meaning
====================  ======================================================
node                  A point of presence or router; ``num_nodes`` is ``n``
ingress, egress       The source and the destination of a flow
epoch                 One time interval; ``num_epochs`` is ``T``
entry                 The volume from one ingress to one egress in one epoch
``total_traffic``     The sum of all entries of a matrix
``mean_entry``        The mean of a single entry (so about
                      ``total_traffic / num_nodes**2``)
====================  ======================================================

The same words are used for every function and parameter.

The TrafficMatrix object
------------------------

:py:class:`~tmgen.TrafficMatrix` wraps the array and keeps it valid:

>>> import numpy
>>> from tmgen import TrafficMatrix
>>> tm = TrafficMatrix(numpy.ones((2, 2, 1)))
>>> tm.shape
(2, 2, 1)

* **It is checked.** The array must have the shape ``(n, n, T)`` with at least
  one node and one epoch, and every volume must be finite and non-negative.
  Otherwise a :py:class:`~tmgen.TMgenException` is raised (it is a
  ``ValueError``).
* **It is copied once** when the matrix is created, and converted to float64,
  so changing your array afterwards does not affect the matrix.
* **It is read-only.** Accessors return read-only data, so a matrix cannot
  change under you:

  >>> tm.matrix[0, 0, 0] = 5
  Traceback (most recent call last):
    ...
  ValueError: assignment destination is read-only

  To modify values, make a copy and build a new matrix:

  >>> values = numpy.array(tm)            # a writable copy
  >>> values[0, 1, 0] = 7
  >>> changed = TrafficMatrix(values)

Self-traffic
------------

Every model generates traffic from a node to itself (the diagonal). If your
network does not have it, remove it; the result is a new matrix, and the totals
drop by the volume that was removed:

>>> tm.zero_diagonal().matrix[:, :, 0].tolist()
[[0.0, 1.0], [1.0, 0.0]]

.. _reproducibility:

Reproducibility
---------------

Every random model, and :py:func:`~tmgen.hmc.hmc_sample`, takes a keyword-only
``rng`` argument:

* an ``int`` seeds a fresh generator,
* a :py:class:`numpy.random.Generator` is used as it is, which is handy for
  drawing many matrices from one stream,
* ``None`` (the default) draws from NumPy's global random state, so
  ``numpy.random.seed`` still works.

>>> from tmgen.models import exp_tm
>>> exp_tm(3, 10, rng=7) == exp_tm(3, 10, rng=7)
True
>>> exp_tm(3, 10, rng=7) == exp_tm(3, 10, rng=8)
False

The exact numbers for a seed are not guaranteed to stay the same between
releases of TMgen or NumPy; pin versions if you need to reproduce a result
exactly.

Errors
------

Invalid input raises :py:class:`~tmgen.TMgenException`, a subclass of
``ValueError``, with a message that names the argument:

>>> exp_tm(0, 10)
Traceback (most recent call last):
  ...
tmgen.exceptions.TMgenException: num_nodes must be a positive integer, got 0
