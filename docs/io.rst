Working with matrices
=====================

Reading entries
---------------

>>> import numpy
>>> from tmgen import TrafficMatrix
>>> tm = TrafficMatrix(numpy.arange(8.0).reshape((2, 2, 2)))
>>> tm.num_nodes, tm.num_epochs, tm.shape
(2, 2, (2, 2, 2))
>>> tm.at_time(1)                    # the matrix of one epoch (ingress x egress)
array([[1., 3.],
       [5., 7.]])
>>> tm.between(0, 1)                 # the series from ingress 0 to egress 1
array([2., 3.])
>>> tm.between(0, 1, agg="max")      # or one number: "min", "max" or "mean"
3.0

:py:meth:`~tmgen.TrafficMatrix.mean` averages over the epochs and
:py:meth:`~tmgen.TrafficMatrix.worst_case` takes the maximum per
ingress-egress pair; both return a new single-epoch matrix.

>>> tm.mean().matrix[:, :, 0]
array([[0.5, 2.5],
       [4.5, 6.5]])

NumPy and Python interoperability
---------------------------------

A traffic matrix behaves like a read-only array in the places where that makes
sense:

>>> numpy.asarray(tm).shape          # also numpy.sum(tm), numpy.array(tm), ...
(2, 2, 2)
>>> float(tm[0, 1, 1])               # indexing as tm[ingress, egress, epoch]
3.0
>>> [epoch.shape for epoch in tm]    # iteration is over the epochs, like len(tm)
[(2, 2), (2, 2)]
>>> len(tm)
2

Matrices compare with ``==``, can be scaled by a non-negative number, and
matrices of the same shape can be added:

>>> float((2 * tm).matrix.max())
14.0
>>> float((tm / 2 + tm).matrix.max())
10.5
>>> tm == TrafficMatrix(tm.matrix)
True

Results that would violate the rules of a matrix (a negative factor, adding
matrices of different shapes) raise :py:class:`~tmgen.TMgenException`.

Changing a matrix
-----------------

A matrix is read-only, so each operation returns a new one:

>>> tm.zero_diagonal().matrix[:, :, 0].tolist()
[[0.0, 2.0], [4.0, 0.0]]
>>> single = TrafficMatrix(numpy.ones((2, 2, 1)))
>>> single.modulate([1.0, 2.0, 0.5]).shape   # one epoch per multiplier
(2, 2, 3)

Files
-----

**NumPy ``.npz``** is the native format. It stores the array and a format
version, compresses well, and loading never unpickles anything, so it is safe
for files from untrusted sources. The file name is used as given:

>>> tm.save("matrix.npz")
>>> TrafficMatrix.load("matrix.npz") == tm
True

:py:meth:`~tmgen.TrafficMatrix.load` raises
:py:class:`~tmgen.TMgenException` if the file is not a TMgen file, was written by
a newer version, or contains invalid data.

**CSV** is a long format that other tools can read: a header
``epoch,ingress,egress,volume`` and one line per entry.

>>> tm.to_csv("matrix.csv")
>>> print(open("matrix.csv").read().splitlines()[:3])
['epoch,ingress,egress,volume', '0,0,0,0', '0,0,1,2']
>>> TrafficMatrix.from_csv("matrix.csv") == tm
True

``from_csv`` accepts the lines in any order, and requires every combination of
epoch, ingress and egress exactly once.

**Pickle** (``to_pickle`` and ``from_pickle``) is deprecated: unpickling a file
can run arbitrary code. Convert old pickle files that you trust once:

.. code-block:: python

   import warnings

   with warnings.catch_warnings():
       warnings.simplefilter("ignore", DeprecationWarning)
       TrafficMatrix.from_pickle("old.pickle").save("old.npz")
