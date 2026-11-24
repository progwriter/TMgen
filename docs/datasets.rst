Real datasets
=============

:py:mod:`tmgen.datasets` loads public measurements, to learn from
(:doc:`lowrank`), to compare generators against, or to test code on real
traffic. Nothing is bundled: a dataset is downloaded from its original host the
first time you ask for it, checked against a checksum and cached in
``~/.cache/tmgen`` (set ``TMGEN_DATA_DIR`` to change that).

Abilene
-------

:py:func:`~tmgen.datasets.load_abilene` returns real traffic matrices of the
Abilene research backbone (Internet2) in 2004 [#zhang2004]_: 12 nodes, one
matrix every 5 minutes, in bytes per second. The data come in 24 weeks of 2016
matrices (about 7 MB each to download); ask for one week, or several, which are
joined in the order you give.

.. code-block:: python

    from tmgen.datasets import ABILENE_NODES, ABILENE_WEEKS, load_abilene

    week = load_abilene(1)          # 2004-03-01, 12 x 12 x 2016
    two_weeks = load_abilene([1, 2])
    week.between(0, 5, "mean")      # ATLA-M5 to IPLSng, mean bytes per second

``ABILENE_NODES`` lists the router names in row and column order and
``ABILENE_WEEKS`` the date each week starts. The measurements do not include
traffic to the Abilene network itself, so the matrices are slightly smaller than
the true totals.

If the download fails with a certificate error, your Python cannot find the
system's certificate authorities; point ``SSL_CERT_FILE`` at a CA bundle (for
example the one from the ``certifi`` package).

.. [#zhang2004] Yin Zhang. *Abilene traffic matrices*, built from Internet2
   NetFlow data (2004). https://www.cs.utexas.edu/~yzhang/research/AbileneTM/
