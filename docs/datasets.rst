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

GEANT
-----

:py:func:`~tmgen.datasets.load_geant` returns real traffic matrices of the
GEANT European research backbone in 2005 [#uhlig2006]_: 23 nodes, one matrix
every 15 minutes from 2005-01-01 to 2005-04-29, in bytes per second (the files
give kilobits per second). The whole archive is one 18 MB download, from which
you pick weeks 1 to 17, joined in the order you give.

.. code-block:: python

    from tmgen.datasets import GEANT_NODES, load_geant

    week = load_geant(1)         # about 670 matrices of 23 x 23
    week.save("geant-week1.npz") # to use the data without TMgen's loader

``GEANT_NODES`` lists the anonymised router ids in row and column order (node
``i`` of a matrix is router ``i + 1``). The dates are anonymised too, so only
the differences between them mean something. The original data lack a few
matrices (some hours here and there, and almost five days in March); they are
skipped rather than filled in, so consecutive epochs are not always 15 minutes
apart. Entries absent from a file are taken as no traffic. Reading the archive
takes a few seconds whichever week you ask for.

The dataset's page states no licence, only a request to cite Uhlig et al. in
any publication that uses the matrices.

.. [#zhang2004] Yin Zhang. *Abilene traffic matrices*, built from Internet2
   NetFlow data (2004). https://www.cs.utexas.edu/~yzhang/research/AbileneTM/

.. [#uhlig2006] Steve Uhlig, Bruno Quoitin, Simon Balon and Jean Lepropre.
   *Providing public intradomain traffic matrices to the research community*.
   ACM SIGCOMM Computer Communication Review 36(1), 2006.
   https://totem.info.ucl.ac.be/dataset.html
