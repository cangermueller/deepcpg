.. _data:

=============
Data creation
=============


This tutorial describes how to create and analyze the input data of DeepCpG.

Creating data
=============

``dcpg_data.py`` creates the data for model training and evaluation, given multiple methylation profiles:

.. code:: bash

  dcpg_data.py
      --cpg_profiles ./cpg/*.tsv
      --dna_files mm10/*.dna.*.fa.gz
      --cpg_wlen 50
      --dna_wlen 1001
      --out_dir ./data

``--cpg_profiles`` specifies a list of files that store the observed CpG methylation states for either a bulk- or single-cell methylation profile. Supports bedGraph and TSV files.

BedGraph files must start with ``track type=bedGraph``. Each following line represents the methylation state of a CpG site, for example:

.. parsed-literal::

  track type=bedGraph
  chr1    3007532 3007533 1.0
  chr1    3007580 3007581 0.4
  chr1    3012096 3012097 1.0
  chr1    3017509 3017510 0.1

The columns have the following meaning:

* Column 1: the chromosome of the CpG site starting with ``chr``.
* Column 2: the location of the C of the CpG site. Positions are enumerated starting at one.
* Column 3: the position of the G of the CpG site.
* Column 4: the observed methylation value (:math:`\in (0;1)`) of the CpG site. If all values are binary, i.e. either zero or one, they will also be stored by DeepCpG as binary values, which reduces disk usage. Continuous values are required for representing hemi-methylation or bulk methylation profiles.


TSV files do not start with a track column and only contain three columns, for example:

.. parsed-literal::

  chr1    3007532 1.0
  chr1    3007580 0.4
  chr1    3012096 1.0
  chr1    3017509 0.1

``--cpg_profiles`` files can be gzip-compressed (``*.gz``) to reduce disk usage.

``--dna_files`` specifies a list of FASTA files, where each file stores the DNA sequence of a particular chromosome. Files can be downloaded from `Ensembl <http://www.ensembl.org/info/data/ftp/index.html>`_, e.g. `mm10 <http://ftp.ensembl.org/pub/release-85/fasta/mus_musculus/dna/>`_ for mouse or `hg38 <http://ftp.ensembl.org/pub/release-86/fasta/homo_sapiens/dna/>`_ for human, and specified either via a glob pattern, e.g. ``--dna_files mm10/*.dna.*fa.gz`` or simply by the directory name, e.g. ``--dna_files mm10``. The argument ``--dna_files`` is not required for imputing methylation states from neighboring methylation states without using the DNA sequence.

``--cpg_wlen`` specifies the sum of CpG sites to the left and right of the target site that DeepCpG will use for making predictions. For example, DeepCpG will use 25 CpG sites to the left and right of the target CpG site using ``--cpg_wlen 50``. A value of about 50 usually covers a wide methylation context and is sufficient to achieve a good performance. If you are dealing with many cells, I recommend using a smaller value to reduce disk usage.

``--dna_wlen`` specifies the width of DNA sequence windows in base pairs that are centered on the target CpG site. Wider windows usually improve prediction accuracy but increase compute- and storage costs. I recommend ``--dna_wlen 1001``.

These are the most important arguments for imputing methylation profiles. ``dcpg_data.py`` provides additional arguments for debugging and predicting statistics across profiles, e.g. the mean methylation rate or cell-to-cell variance.


Debugging
---------

For debugging, testing, or reducing compute costs, ``--chromos`` can be used the select certain chromosomes. ``--nb_sample_chromo`` randomly samples a certain number of CpG sites from each chromosome, and ``--nb_sample`` specifies the maximum number of CpG sites in total.


Predicting statistics
---------------------

For predicting statistics across methylation profiles, ``--cpg_stats`` and ``--win_stats`` can be used. These arguments specify a list of statistics that are computed across profiles for either a single CpG site or in windows of length ``--win_stats_wlen`` that are centered on a CpG site. Following statistics are supported:

* ``mean``: the mean methylation rate.
* ``mode``: the mode of methylation rates.
* ``var``: the cell-to-cell variance.
* ``cat_var``: three categories of cell-to-cell variance, i.e. low, medium, or high variance.
* ``cat2_var``: two categories of cell-to-cell variance, i.e. low or high variance.
* ``entropy``: the entropy across cells.
* ``diff``: if a CpG site is differentially methylated, i.e. methylated in one profile but zero in others.
* ``cov``: the CpG coverage, i.e. the number of profiles for which the methylation state of the target CpG site is observed.

Per-CpG statistics specified by ``--cpg_stats`` are computed only for CpG sites that are covered by at least ``--cpg_stats_cov`` (default 3) cells. Increasing ``--cpg_stats_cov`` will lead to more robust estimates.


Common issues
-------------

**Why am I getting warnings 'No CpG site at position X!' when using ``dcpg_data.py``?**

This means that some sites in ``--cpg_profile`` files are not CpG sites, i.e. there is no CG dinucleotide at the given position in the DNA sequence. Make sure that ``--dna_files`` point to the correct genome and CpG sites are correctly aligned. Since DeepCpG currently does not support allele-specific methylation, data from different alleles must be merged (recommended) or only one allele be used.


Computing data statistics
=========================

``dcpg_data_stats.py`` enables to compute statistics for a list of DeepCpG input files:

.. code:: bash

  dcpg_data_stats.py ./data/c1_000000-001000.h5 ./data/c13_000000-001000.h5

.. parsed-literal::

             output  nb_tot  nb_obs  frac_obs      mean       var
  0  cpg/BS27_1_SER    2000     391    0.1955  0.790281  0.165737
  1  cpg/BS27_3_SER    2000     408    0.2040  0.740196  0.192306
  2  cpg/BS27_5_SER    2000     393    0.1965  0.692112  0.213093
  3  cpg/BS27_6_SER    2000     402    0.2010  0.666667  0.222222
  4  cpg/BS27_8_SER    2000     408    0.2040  0.776961  0.173293

The columns have the following meaning:

* ``output``: The name of the target cell.
* ``nb_tot``: The total number of CpG sites.
* ``nb_obs``: The number of CpG sites for which the true label of ``output`` is observed.
* ``frac_obs``: The fraction ``nb_obs/nb_tot`` of observed CpG sites.
* ``mean``: The mean of ``output``, e.g. the mean methylation rate.
* ``var``: The variance of ``output``, e.g. the variance in CpG methylation levels.

``--nb_tot`` and ``--nb_obs`` are particularly useful for deciding how to split the data into a training, test, validation set as explained in the :ref:`training tutorial <train>`. Statistics can be written to a TSV file using ``--out_tsv`` and be visualized using ``--out_plot``.


Printing data
=============

``dcpg_data_show.py`` enables to selectively print the content of a list of DeepCpG data files. Using ``--outputs`` prints all DeepCpG model outputs in a selected region:

.. code:: bash

  dcpg_data_show.py ./data/c1_000000-001000.h5 --chromo 1 --start  189118909 --end 189867450 --outputs

.. parsed-literal::

        loc                   outputs
      chromo        pos cpg/BS27_1_SER cpg/BS27_3_SER cpg/BS27_5_SER cpg/BS27_6_SER cpg/BS27_8_SER
  950      1  189118909             -1             -1              1             -1             -1
  951      1  189314906             -1             -1              1             -1             -1
  952      1  189506185              1             -1             -1             -1             -1
  953      1  189688256             -1              0             -1             -1             -1
  954      1  189688274             -1             -1             -1             -1              0
  955      1  189699529             -1             -1             -1              1             -1
  956      1  189728263             -1             -1              0             -1             -1
  957      1  189741539             -1              1             -1             -1             -1
  958      1  189850865             -1             -1             -1              1             -1
  959      1  189867450             -1              1             -1             -1             -1


``-1`` indicates unobserved methylation states. If ``--outputs`` is followed by a list of output names, only they will be printed. The arguments ``--cpg``, ``--cpg_wlen``, and ``--cpg_dist`` control how many (``--cpg_wlen``) neighboring methylation states (``--cpg``) and corresponding distances (``--cpg_dist``) are printed. For example, the following commands prints the state and distance of four neighboring CpG sites of cell *BS27_1_SER*:

.. code:: bash

  dcpg_data_show.py ./data/c1_000000-001000.h5 --chromo 1 --start  189118909 --end 189867450 --outputs cpg/BS27_1_SER --cpg BS27_1_SER --cpg_wlen 4 --cpg_dist

.. parsed-literal::

        loc                   outputs BS27_1_SER/state          BS27_1_SER/dist
      chromo        pos cpg/BS27_1_SER               -2 -1 +1 +2              -2        -1        +1        +2
  950      1  189118909             -1                1  1  1  1         84023.0   65557.0  114153.0  373437.0
  951      1  189314906             -1                1  1  1  1        261554.0   81844.0  177440.0  191279.0
  952      1  189506185              1                1  1  1  0        273123.0   13839.0  162360.0  662239.0
  953      1  189688256             -1                1  1  0  1        182071.0   19711.0  480168.0  705968.0
  954      1  189688274             -1                1  1  0  1        182089.0   19729.0  480150.0  705950.0
  955      1  189699529             -1                1  1  0  1        193344.0   30984.0  468895.0  694695.0
  956      1  189728263             -1                1  1  0  1        222078.0   59718.0  440161.0  665961.0
  957      1  189741539             -1                1  1  0  1        235354.0   72994.0  426885.0  652685.0
  958      1  189850865             -1                1  1  0  1        344680.0  182320.0  317559.0  543359.0
  959      1  189867450             -1                1  1  0  1        361265.0  198905.0  300974.0  526774.0

Analogously, ``--dna_wlen`` prints the DNA sequence window of that length centered on the target CpG sites:

.. code:: bash

  dcpg_data_show.py ./data/c1_000000-001000.h5 --chromo 1 --start  189118909 --end 189867450 --outputs cpg/BS27_1_SER --dna_wlen 11

.. parsed-literal::

        loc                   outputs dna
      chromo        pos cpg/BS27_1_SER  -5 -4 -3 -2 -1  0 +1 +2 +3 +4 +5
  950      1  189118909             -1   2  1  0  0  0  3  2  2  0  0  3
  951      1  189314906             -1   3  1  3  3  2  3  2  3  0  1  3
  952      1  189506185              1   0  3  3  3  0  3  2  2  2  0  1
  953      1  189688256             -1   2  3  3  2  2  3  2  2  3  2  2
  954      1  189688274             -1   2  3  0  2  0  3  2  1  3  2  2
  955      1  189699529             -1   2  3  2  2  0  3  2  3  1  1  1
  956      1  189728263             -1   3  1  3  3  3  3  2  2  3  3  2
  957      1  189741539             -1   2  0  2  1  2  3  2  1  2  2  3
  958      1  189850865             -1   2  2  3  2  2  3  2  2  3  2  2
  959      1  189867450             -1   3  1  3  0  3  3  2  1  2  3  0

With ``--out_hdf``, the selected data can be stored as `Pandas data frame <http://pandas.pydata.org/pandas-docs/stable/io.html#io-hdf5>`_ to a HDF5 file.
