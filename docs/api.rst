.. _API:

API reference
=============

Models
------

.. automodule:: tmgen.models
  :members:

Learning from measurements
--------------------------

.. autoclass:: tmgen.lowrank.LowRankModel
  :members: fit, sample, num_nodes, rank

TrafficMatrix
-------------

.. autoclass:: tmgen.TrafficMatrix
  :members:

Errors
------

.. autoexception:: tmgen.TMgenException

Sampling
--------

.. autofunction:: tmgen.hmc.hmc_sample

Plotting
--------

.. automodule:: tmgen.plot
  :members:

Traffic injection
-----------------

.. autoclass:: tmgen.inject.InjectorBase
  :members:
