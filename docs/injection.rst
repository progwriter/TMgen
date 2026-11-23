Injecting traffic with D-ITG
============================

TMgen can replay a traffic matrix as real network traffic with `D-ITG
<http://www.grid.unina.it/software/ITG/>`_, a traffic generator. The
``tmditg`` command starts the D-ITG receiver and senders on one node of your
testbed according to a matrix. Entries are interpreted as packets per second,
and every epoch of the matrix lasts ``--epoch-length`` seconds.

The traffic matrix must be a file written by
:py:meth:`TrafficMatrix.save <tmgen.TrafficMatrix.save>`:

.. code-block:: python

   from tmgen.models import modulated_gravity_tm

   tm = modulated_gravity_tm(4, 24, total_traffic=2000, rng=0)
   tm.save("tm.npz")

Then, on the machine that plays node ``0``:

.. code-block:: console

   tmditg --tm tm.npz --epoch-length 60 --id 0 \
          --destinations destinations.json --port 9000

``--destinations`` is a JSON file that maps node IDs to lists of IP addresses.
``--scale`` multiplies all entries, which is handy to fit the traffic to the
capacity of your testbed. Run ``tmditg --help`` for all options.

.. note::

   Older versions read pickle files here. Loading a pickle can run arbitrary
   code, so ``tmditg`` now reads only the ``.npz`` format. Convert trusted
   pickles once with ``TrafficMatrix.from_pickle(...).save(...)``.

For your own injector, subclass :py:class:`tmgen.inject.InjectorBase`.
