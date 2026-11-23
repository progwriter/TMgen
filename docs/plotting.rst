Plotting
========

:py:mod:`tmgen.plot` has two basic plots. They need matplotlib (``pip install
"tmgen[plotting]"``); seaborn is used for prettier heatmaps when it is
installed. Both functions draw on a matplotlib ``Axes`` that you can pass with
``ax`` (the current axes by default) and return it, so they fit in subplots and
you can customize the result. Creating, showing and saving figures is up to you.

Heatmap
-------

A single epoch, or one chosen epoch of a longer matrix, is shown as an ingress
by egress heatmap:

.. code-block:: python

   from matplotlib import pyplot as plt

   from tmgen import models, plot

   tm = models.modulated_gravity_tm(10, 24, total_traffic=1000, rng=0)

   fig, (left, right) = plt.subplots(1, 2, figsize=(10, 4))
   plot.heatmap(tm, epoch=12, ax=left)   # noon
   plot.heatmap(tm, ax=right)            # all epochs: pairs on y, epochs on x
   plt.show()

Without ``epoch`` a multi-epoch matrix is drawn with the ingress-egress pairs
on the y axis and the epochs on the x axis.

Time series
-----------

:py:func:`~tmgen.plot.timeseries` draws one line per ingress-egress pair, with
the epochs on the x axis:

.. code-block:: python

   fig, ax = plt.subplots()
   plot.timeseries(tm, ax=ax)
   ax.set_title("Traffic per ingress-egress pair")
   fig.savefig("timeseries.png")
