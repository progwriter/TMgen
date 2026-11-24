TMgen: traffic matrix generation
================================

TMgen generates synthetic network traffic matrices: how much traffic flows from
each node of a network to each other node, optionally over time. Use it to test
routing, traffic engineering and topology design code when you do not have (or
cannot share) real measurements.

It is a collection of synthesis techniques from the research literature, each
credited to its paper in :doc:`models`, next to the simple models people usually
start with. They all return the same small :py:class:`~tmgen.TrafficMatrix`
object, with NumPy interoperability, safe file formats and basic plots, so you
can swap one technique for another, combine them, and check the result against
real traffic (:doc:`metrics`, :doc:`datasets`).

.. code-block:: python

   from tmgen.models import modulated_gravity_tm

   # 10 nodes, a day of hourly matrices, about 1000 units of traffic per hour
   tm = modulated_gravity_tm(num_nodes=10, num_epochs=24, total_traffic=1000, rng=42)

Where to go next
----------------

:doc:`quickstart`
   Install TMgen, generate your first matrix, read it, and save it.

:doc:`concepts`
   What a traffic matrix is here, the vocabulary used throughout, and the rules
   every matrix follows.

:doc:`models`
   The models, when to use each, and how to combine them.

:doc:`lowrank`, :doc:`datasets`, :doc:`metrics`
   Learn a model from measured matrices, load real Abilene traffic, and compare
   synthetic with real series.

:doc:`io`
   Reading entries, NumPy interoperability, and the ``.npz`` and CSV formats.

:doc:`sampling`
   The truncated Gaussian sampler behind the paper's models.

:doc:`plotting`, :doc:`injection`
   Visualizing matrices and replaying them as network traffic with D-ITG.

:doc:`upgrading`
   What changed since 0.1 and how to update your code.

:doc:`api`
   The reference for every public function and class.

.. toctree::
   :hidden:
   :maxdepth: 2

   quickstart
   concepts
   models
   lowrank
   datasets
   metrics
   io
   sampling
   plotting
   injection
   upgrading
   api
