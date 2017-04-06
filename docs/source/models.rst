.. _models:

===================
Model architectures
===================

DeepCpG consists of a DNA model to recognize features in the DNA
sequence, a CpG model to recognize features in the methylation
neighborhood of multiple cells, and a Joint model to combine the features
from the DNA and CpG model.

DeepCpG provides different architectures for the DNA, CpG, and joint
model. Architectures differ in the number of layers and neurons, and
are hence more or less complex. More complex models are usually more
accurate, but more expensive to train. You can select a certain
architecture using the ``--dna_model``, ``--cpg_model``, and
``--joint_model`` argument of ``dcpg_train.py``, for example:

.. code:: bash

    dcpg_train.py
        --dna_model CnnL2h128
        --cpg_model RnnL1
        --joint_model JointL2h512

In the following, the following layer specifications will be used:

+-----------------+----------------------------------------------------------------------------+
| Specification   | Description                                                                |
+=================+============================================================================+
| conv[x\@y]      | Convolutional layer with x filters of size y                               |
+-----------------+----------------------------------------------------------------------------+
| mp[x]           | Max-pooling layer with size x                                              |
+-----------------+----------------------------------------------------------------------------+
| fc[x]           | Full-connected layer with x units                                          |
+-----------------+----------------------------------------------------------------------------+
| do              | Dropout layer                                                              |
+-----------------+----------------------------------------------------------------------------+
| bgru[x]         | Bidirectional GRU with x units                                             |
+-----------------+----------------------------------------------------------------------------+
| gap             | Global average pooling layer                                               |
+-----------------+----------------------------------------------------------------------------+
| resb[x,y,z]     | Residual network with three bottleneck residual units of size x, y, z      |
+-----------------+----------------------------------------------------------------------------+
| resc[x,y,z]     | Residual network with three convolutional residual units of size x, y, z   |
+-----------------+----------------------------------------------------------------------------+
| resa[x,y,z]     | Residual network with three Atrous residual units of size x, y, z          |
+-----------------+----------------------------------------------------------------------------+

DNA model architectures
=======================

+-------------+--------------+-----------------------------------------------------------------------------+
| Name        | Parameters   | Specification                                                               |
+=============+==============+=============================================================================+
| CnnL1h128   | 4,100,000    | conv[128\@11]\_mp[4]\_fc[128]\_do                                           |
+-------------+--------------+-----------------------------------------------------------------------------+
| CnnL1h256   | 8,100,000    | conv[128\@11]\_mp[4]\_fc[256]\_do                                           |
+-------------+--------------+-----------------------------------------------------------------------------+
| CnnL2h128   | 4,100,000    | conv[128\@11]\_mp[4]\_conv[256\@3]\_mp[2]\_fc[128]\_do                      |
+-------------+--------------+-----------------------------------------------------------------------------+
| CnnL2h256   | 8,100,000    | conv[128\@11]\_mp[4]\_conv[256\@3]\_mp[2]\_fc[256]\_do                      |
+-------------+--------------+-----------------------------------------------------------------------------+
| CnnL3h128   | 4,400,000    | conv[128\@11]\_mp[4]\_conv[256\@3]\_mp[2]\_conv[512\@3]\_mp[2]\_fc[128]\_do |
+-------------+--------------+-----------------------------------------------------------------------------+
| CnnL3h256   | 8,300,000    | conv[128\@11]\_mp[4]\_conv[256\@3]\_mp[2]\_conv[512\@3]\_mp[2]\_fc[128]\_do |
+-------------+--------------+-----------------------------------------------------------------------------+
| CnnRnn01    | 1,100,000    | conv[128\@11]\_pool[4]\_conv[256\@7]\_pool[4]\_bgru[256]\_do                |
+-------------+--------------+-----------------------------------------------------------------------------+
| ResNet01    | 1,700,000    | conv[128\@11]\_mp[2]\_resb[2x128\|2x256\|2x512\|1x1024]\_gap\_do            |
+-------------+--------------+-----------------------------------------------------------------------------+
| ResNet02    | 2,000,000    | conv[128\@11]\_mp[2]\_resb[3x128\|3x256\|3x512\|1x1024]\_gap\_do            |
+-------------+--------------+-----------------------------------------------------------------------------+
| ResConv01   | 2,800,000    | conv[128\@11]\_mp[2]\_resc[2x128\|1x256\|1x256\|1x512]\_gap\_do             |
+-------------+--------------+-----------------------------------------------------------------------------+
| ResAtrous01 | 2,000,000    | conv[128\@11]\_mp[2]\_resa[3x128\|3x256\|3x512\|1x1024]\_gap\_do            |
+-------------+--------------+-----------------------------------------------------------------------------+

Th prefixes ``Cnn``, ``CnnRnn``, ``ResNet``, ``ResConv``, and
``ResAtrous`` denote the class of the DNA model.

Models starting with ``Cnn`` are convolutional neural networks (CNNs).
DeepCpG CNN architectures consist of a series of convolutional and
max-pooling layers, which are followed by one fully-connected layer.
Model ``CnnLxhy`` has ``x`` convolutional-pooling layers, and one
fully-connected layer with ``y`` units. For example, ``CnnL2h128`` has
two convolutional layers, and one fully-connected layer with 128 units.
``CnnL3h256`` has three convolutional layers and one fully-connected
layer with 256 units. ``CnnL1h128`` is the fastest model, but models
with more layers and neurons usually perform better. In my experiments,
``CnnL2h128`` provided a good trade-off between performance and runtime,
which I recommend as default.

``CnnRnn01`` is a `convolutional-recurrent neural
network <http://nar.oxfordjournals.org/content/44/11/e107>`__. It
consists of two convolutional-pooling layers, which are followed by a
bidirectional recurrent neural network (RNN) with one layer and gated
recurrent units (GRUs). ``CnnRnn01`` is slower than ``Cnn``
architectures and did not perform better in my experiments.

Models starting with ``ResNet`` are `residual neural
networks <https://arxiv.org/abs/1603.05027>`__. ResNets are very deep
networks with skip connections to improve the gradient flow and to allow
learning how many layers to use. A residual network consists of multiple
residual blocks, and each residual block consists of multiple residual
units. Residual units have a bottleneck architecture with three
convolutional layers to speed up computations. ``ResNet01`` and
``ResNet02`` have three residual blocks with two and three residual
units, respectively. ResNets are slower than CNNs, but can perform
better on large datasets.

Models starting with ``ResConv`` are ResNets with modified residual
units that have two convolutional layers instead of a bottleneck
architecture. ``ResConv`` models performed worse than ``ResNet``
models in my experiments.

Models starting with ``ResAtrous`` are ResNets with modified residual
units that use `Atrous convolutional
layers <http://arxiv.org/abs/1511.07122>`__ instead of normal
convolutional layers. Atrous convolutional layers have dilated filters,
i.e. filters with 'holes', which allow scanning wider regions in the
inputs sequence and thereby better capturing distant patters in the DNA
sequence. However, ``ResAtrous`` models performed worse than ``ResNet``
models in my experiments

CpG model architectures
=======================

+---------+--------------+-----------------------------------+
| Name    | Parameters   | Specification                     |
+=========+==============+===================================+
| FcAvg   | 54,000       | fc[512]\_gap                      |
+---------+--------------+-----------------------------------+
| RnnL1   | 810,000      | fc[256]\_bgru[256]\_do            |
+---------+--------------+-----------------------------------+
| RnnL2   | 1,100,000    | fc[256]\_bgru[128]\_bgru[256]\_do |
+---------+--------------+-----------------------------------+

``FcAvg`` is a lightweight model with only 54000 parameters, which
first transforms observed neighboring CpG sites of all cells
independently, and than averages the transformed features across cells.
``FcAvg`` is very fast, but performs worse than RNN models.

``Rnn`` models consists of bidirectional recurrent neural networks
(RNNs) with gated recurrent units (GRUs) to summarize the methylation
neighborhood of cells in a more clever way than averaging. ``RnnL1``
consists of one fully-connected layer with 256 units to transform the
methylation neighborhood of each cell independently, and one
bidirectional GRU with 2x256 units to summarize the transformed
methylation neighborhood of cells. ``RnnL2`` has two instead of one GRU
layer. ``RnnL1`` is faster and performed as good as ``RnnL2`` in my
experiments.

Joint model architectures
=========================

+---------------+--------------+---------------------------+
| Name          | Parameters   | Specification             |
+===============+==============+===========================+
| JointL0       | 0            |                           |
+---------------+--------------+---------------------------+
| JointL1h512   | 524,000      | fc[512]                   |
+---------------+--------------+---------------------------+
| JointL2h512   | 786,000      | fc[512]\_fc[512]          |
+---------------+--------------+---------------------------+
| JointL3h512   | 1,000,000    | fc[512]\_fc[512]\_fc[512] |
+---------------+--------------+---------------------------+

Joint models join the feature from the DNA and CpG model. ``JointL0``
simply concatenates the features and has no learnable parameters (ultra
fast). ``JointLXh512`` has ``X`` fully-connect layers with 512 neurons.
Models with more layers usually perform better, at the cost of a higher
runtime. I recommend using ``JointL2h512`` or ``JointL3h12``.
