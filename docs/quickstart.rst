Quickstart
==========

This document oulitnes how to easily start using the TMgen tool.

Download and installation
-------------------------

Using pip
^^^^^^^^^

::

    pip install tmgen

From source
^^^^^^^^^^^

1. Clone ::

    git clone https://github.com/progwriter/tmgen

2. Install using pip in development mode ::

    cd tmgen
    pip install -e .

Example usage
-------------

TMgen defines a ``TrafficMatrix`` object that is returned by all generator
functions. Internally it contains a 3-d numpy array which contains volume of
traffic between origin-destination pairs for different time epochs. For TM models
that do not have a time component, the third dimension is of size 1.

Quick overview on how to use them is given here,
for full details please see the :ref:`API`

Generating a traffic matrix
^^^^^^^^^^^^^^^^^^^^^^^^^^^

Lets generate a uniform traffic matrix for a network with 3 nodes:

>>> from tmgen.models import uniform_tm
>>> tm = uniform_tm(3, 100, 300)
>>> print(tm) # doctest: +ELLIPSIS
array([[[...
<BLANKLINE>
...
<BLANKLINE>
...]]])

This gives us a 3x3x1 array with values between 100 and 300 --- volume for each
node pair, with only one time epoch.

Accessing tm entries
^^^^^^^^^^^^^^^^^^^^

TMgen gives you a number of ways to access the TM values. Lets generate an
exponential TM with the mean volume of 500 and 2 time epochs.

>>> from tmgen.models import exp_tm
>>> tm = exp_tm(3, 500, 2)

Accessing the *matrix* attrbute gives us the underlying Numpy array:

>>> tm.matrix # doctest: +ELLIPSIS
array([[[...
<BLANKLINE>
...
<BLANKLINE>
...]]])

Also we can request a traffic matrix at a specific epoch (0-indexed):

>>> tm.at_time(1) # doctest: +ELLIPSIS
array([[...]])

Or the values between any node pair, with a particular aggregation metric:

>>> tm.between(0,2,'all') # doctest: +ELLIPSIS
array([...])
>>> tm.between(0,2,'max') # doctest: +SKIP
>>> tm.between(0,2,'mean') # doctest: +SKIP

See :ref:`API` for all supported functions.

Saving/Loading a traffic matrix
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

TMs can be easily loaded using the python-native pickle format:

>>> from tmgen import TrafficMatrix
>>> tm.to_pickle('mytm')
>>> tm = TrafficMatrix.from_pickle('mytm')
