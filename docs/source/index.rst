========================================================================
DeepCpG: Deep neural networks for predicting single-cell DNA methylation
========================================================================

DeepCpG [1]_ is a deep neural network for predicting the methylation state of CpG dinucleotides in multiple cells. It allows to accurately impute incomplete DNA methylation profiles, to discover predictive sequence motifs, and to quantify the effect of sequence mutations. (`Angermueller et al, 2017 <http://dx.doi.org/10.1186/s13059-017-1189-z>`_).

.. figure:: fig1.png
   :width: 640 px
   :align: left
   :alt: DeepCpG model architecture and applications

   **DeepCpG model architecture and applications.**

   \(a\) Sparse single-cell CpG profiles as obtained from scBS-seq or scRRBS-seq. Methylated CpG sites are denoted by ones, unmethylated CpG sites by zeros, and question marks denote CpG sites with unknown methylation state (missing data). (b) DeepCpG model architecture. The DNA model consists of two convolutional and pooling layers to identify predictive motifs from the local sequence context, and one fully connected layer to model motif interactions. The CpG model scans the CpG neighborhood of multiple cells (rows in b), using a bidirectional gated recurrent network (GRU), yielding compressed features in a vector of constant size. The Joint model learns interactions between higher-level features derived from the DNA- and CpG model to predict methylation states in all cells. (c, d) The trained DeepCpG model can be used for different downstream analyses, including genome-wide imputation of missing CpG sites (c) and the discovery of DNA sequence motifs that are associated with DNA methylation levels or cell-to-cell variability (d).


.. [1] Angermueller, Christof, Heather J. Lee, Wolf Reik, and Oliver Stegle. *DeepCpG: Accurate Prediction of Single-Cell DNA Methylation States Using Deep Learning.* Genome Biology 18 (April 11, 2017): 67. doi:10.1186/s13059-017-1189-z.


Installation
============

The easiest way to install DeepCpG is to use ``PyPI``:

.. code:: bash

  pip install deepcpg

Alternatively, you can checkout the repository

.. code:: bash

  git clone https://github.com/cangermueller/deepcpg.git


and then install DeepCpG using ``setup.py``:

.. code:: bash

  python setup.py install


Examples
========

Interactive examples on how to use DeepCpG can be found `here <https://github.com/cangermueller/deepcpg/tree/master/examples>`_.


Documentation
=============

* :ref:`data` -- Creating and analyzing data.
* :ref:`train` -- Training DeepCpG models.
* :ref:`models` -- Description of DeepCpG model architectures.
* :ref:`scripts` -- Documentation of DeepCpG scripts.
* :ref:`library` -- Documentation of DeepCpG library.


Indices and tables
==================
* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`


.. toctree::
  :maxdepth: 1
  :hidden:

  data
  train
  models
  scripts/index
  lib/index
