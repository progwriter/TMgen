Quickstart
==========

Installation
------------

::

    pip install tmgen

TMgen needs Python 3.11 or newer. Prebuilt wheels are published for common
platforms; building from source needs a C compiler (the HMC sampler is a small
Cython extension). The plotting helpers need matplotlib, which you can get with
``pip install "tmgen[plotting]"``.

To work on TMgen itself, see ``DEVELOPMENT.md`` in the repository.

Your first traffic matrix
-------------------------

Every model is a function in :py:mod:`tmgen.models` that returns a
:py:class:`~tmgen.TrafficMatrix`. Here is a network of 4 nodes where each entry
is drawn from an exponential distribution with a mean of 100, for 3 time
epochs:

>>> from tmgen.models import exp_tm
>>> tm = exp_tm(num_nodes=4, mean_entry=100, num_epochs=3, rng=42)
>>> tm.shape
(4, 4, 3)

The shape is ``(ingress, egress, epoch)``: entry ``tm[i, j, t]`` is the volume
from node ``i`` to node ``j`` during epoch ``t``. Models without a time
component, such as :py:func:`~tmgen.models.gravity_tm`, have a single epoch.

The ``rng`` argument makes the result reproducible: the same seed gives the same
matrix (see :ref:`reproducibility`).

Reading the data
----------------

>>> tm.num_nodes, tm.num_epochs
(4, 3)
>>> tm.at_time(1).shape             # the 4 x 4 matrix of epoch 1
(4, 4)
>>> tm.between(0, 2).shape          # volumes from node 0 to node 2, per epoch
(3,)
>>> bool(tm.between(0, 2, "max") >= tm.between(0, 2, "mean"))
True

:py:attr:`~tmgen.TrafficMatrix.matrix` is the underlying NumPy array, and
``numpy.asarray(tm)`` works too. Matrices are read-only: see :doc:`concepts`
for why, and :doc:`io` for everything you can do with the data.

A realistic example
-------------------

Real traffic follows a daily cycle. The modulated gravity model produces a
matrix per hour whose total follows a cycle, with traffic spread over the
nodes in a gravity pattern:

>>> from tmgen.models import modulated_gravity_tm
>>> day = modulated_gravity_tm(num_nodes=6, num_epochs=24, total_traffic=1000, rng=1)
>>> totals = day.matrix.sum(axis=(0, 1))     # total traffic of each hour
>>> round(float(totals.min())), round(float(totals.max()))  # 0.5x to 1.5x
(500, 1500)

Saving and loading
------------------

A matrix is saved as a compressed NumPy ``.npz`` file. Loading never runs code
from the file, so it is safe for files you did not create:

>>> from tmgen import TrafficMatrix
>>> day.save("day.npz")
>>> TrafficMatrix.load("day.npz") == day
True

Next steps
----------

* :doc:`models` helps you pick a model.
* :doc:`concepts` explains the vocabulary and the guarantees of a
  ``TrafficMatrix``.
