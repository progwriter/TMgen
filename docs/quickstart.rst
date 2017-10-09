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

  >>> from tmgen import uniform_tm
  >>> tm = uniform_tm(3, 100, 300)
  >>> print tm
  array([[[ 219.73016387],
        [ 161.41332385],
        [ 120.68272977]],
       [[ 246.9339771 ],
        [ 189.98348848],
        [ 290.19555679]],
       [[ 119.98074672],
        [ 150.05173824],
        [ 255.83845338]]])

This gives us a 3x3x1 array with values between 100 and 300 --- volume for each
node pair, with only one time epoch.

Accessing tm entries
^^^^^^^^^^^^^^^^^^^^

TMgen gives you a number of ways to access the TM values. Lets generate an
exponential TM with the mean volume of 500 and 2 time epochs.

  >>> from tmgen import exp_tm
  >>> tm = exp_tm(3, 500, 2)

Accessing the *matrix* attrbute gives us the underlying Numpy array:

  >>> tm.matrix
  array([[[  6.37997965e+02,   1.09182535e+02],
        [  3.14477723e+02,   8.80934257e+02],
        [  3.48359849e+02,   1.00303448e+03]],
       [[  6.51216211e+02,   1.16041768e+03],
        [  3.16695016e+02,   6.97480254e-01],
        [  3.00624933e+02,   1.26349570e+02]],
       [[  1.43754204e+03,   6.61064394e+00],
        [  3.31300472e+02,   1.12039376e+02],
        [  5.79562994e+02,   4.57798655e+01]]])

Also we can request a tm at a specific epoch (0-indexed):
  >>> tm.at_time(1)
  array([[  1.09182535e+02,   8.80934257e+02,   1.00303448e+03],
       [  1.16041768e+03,   6.97480254e-01,   1.26349570e+02],
       [  6.61064394e+00,   1.12039376e+02,   4.57798655e+01]])

Or the values between any node pair:

  >>> tm.between(0,2,'all')
  array([  348.35984873,  1003.03447668])
  >>> tm.between(0,2,'max')
  1003.0344766776753
  >>> tm.between(0,2,'mean')
  675.69716270379729

See :ref:`API` for all supported functions.

Saving/Loading a traffic matrix
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

TMs can be easily loaded using the python-native pickle format:

  >>> tm.to_pickle('mytm')
  >>> tm = TrafficMatrix.from_pickle('mytm')
