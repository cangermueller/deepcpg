.. _train:

==============
Model training
==============

Here you can find information about how to train DeepCpG models.

.. _train_split:

Splitting data into training, validation, and test set
======================================================

For comparing different models, it is necessary to train, select
hyper-parameters, and test models on distinct data. In holdout
validation, the dataset is split into a training set (~60% of the data),
validation set (~20% of the data), and test set (~20% of the data).
Models are trained on the training set, hyper-parameters selected on the
validation set, and the selected models compared on the test set. For
example, you could use chromosome 1-5, 7, 9, 11, 13 as training set,
chromosome 14-19 as validation set, and chromosome 6, 8, 10, 12, 14 as
test set:

.. code:: bash

    train_files="$data_dir/c{1,2,3,4,5,7,9,11,13}_*.h5
    val_files="$data_dir/c{14,15,16,17,18,19}_*.h5"
    test_files="$data_dir/c{6,8,10,12,14}_*.h5"

    dcpg_train.py
        $train_files
        --val_file $val_files
        ...

As you can see, DeepCpG allows to easily split the data by glob
patterns. You do not have to split the dataset by chromosomes. For
example, you could use ``train_files=$data_Dir/c*_[01].h5`` to select
all data files starting with index 0 or 1 for training, and use the
remaining files for validation.

If you are not concerned about comparing DeepCpG with other models, you
do not need a test set. In this case, you could, for example, leave out
chromosome 14-19 as validation set, and use the remaining chromosomes for
training.

If your data were generated using whole-genome scBS-seq, then the number
of CpG sites on few chromosomes is usually already sufficient for
training. For example, chromosome 1, 3, and 5 from *Smallwood et al
(2014)* cover already more than 3 million CpG sites. I found about 3
million CpG sites as sufficient for training models without overfitting.
However, if you are working with scRRBS-seq data, you probably need more
chromosomes for training. To check how many CpG sites are stored in a set
of DeepCpG data files, you can use the ``dcpg_data_stats.py``. The
following command will compute different statistics for the training
set, including the number number of CpG sites:

.. code:: bash

    dcpg_data_stats.py $data_dir/$train_files


.. parsed-literal::

    #################################
    dcpg_data_stats.py ./data/c19_000000-032768.h5 ./data/c19_032768-050000.h5
    #################################
               output  nb_tot  nb_obs  frac_obs      mean       var
    0  cpg/BS27_1_SER   50000   20621   0.41242  0.665972  0.222453
    1  cpg/BS27_3_SER   50000   13488   0.26976  0.573102  0.244656
    2  cpg/BS27_5_SER   50000   25748   0.51496  0.529633  0.249122
    3  cpg/BS27_6_SER   50000   17618   0.35236  0.508117  0.249934
    4  cpg/BS27_8_SER   50000   16998   0.33996  0.661019  0.224073


For each output cell, ``nb_tot`` is the total number of CpG sites,
``nb_obs`` the number of CpG sites with known methylation state,
``frac_obs`` the ratio between ``nb_obs`` and ``nb_tot``, ``mean`` the
mean methylation rate, and ``var`` the variance of the methylation rate.

.. _train_joint:

Training DeepCpG models jointly
================================

As described in `Angermueller et al
(2017) <http://biorxiv.org/content/early/2017/02/01/055715>`__, DeepCpG
consists of a DNA, CpG, and Joint model. The DNA model recognizes
features in the DNA sequence window that is centered on a target site,
the CpG model recognizes features in observed neighboring methylation
states of multiple cells, and the Joint model integrates features from
the DNA and CpG model and predicts the methylation state of all cells.

The easiest way is to train all models jointly:

.. code:: bash

    dcpg_train.py
        $train_files
        --val_files $val_files
        --dna_model CnnL2h128
        --cpg_model RnnL1
        --out_dir $models_dir/joint
        --nb_epoch 30

``--dna_model``, ``--cpg_model``, and ``--joint_model`` specify the
architecture of the DNA, CpG, and Joint model, respectively, which are
described in `here <./models.rst>_`.

.. _train_sep:

Training DeepCpG models separately
===================================

Although it is convenient to train all models jointly by running only a
single command as described above, I suggest to train models
separately. First, because it enables to train the DNA and CpG model in
parallel on separate machines and thereby to reduce the training time.
Second, it enables to compare how predictive the DNA model is relative
to CpG model. If you think the CpG model is already accurate enough
alone, you might not need the DNA model. Thirdly, I obtained better
results by training the models separately. However, this may not be
true for your particular dataset.

You can train the CpG model separately by only using the
``--cpg_model`` argument, but not ``--dna_model``:

.. code:: bash

    dcpg_train.py
        $train_files
        --val_files $val_files
        --dna_model CnnL2h128
        --out_dir $models_dir/dna
        --nb_epoch 30

You can train the DNA model separately by only using ``--dna_model``:

.. code:: bash

    dcpg_train.py
        $train_files
        --val_files $val_files
        --cpg_model RnnL1
        --out_dir $models_dir/cpg
        --nb_epoch 30

After training the CpG and DNA model, we are joining them by specifying
the name of the Joint model with ``--joint_model``:

.. code:: bash

    dcpg_train.py
        $train_files
        --val_files $val_files
        --dna_model $models_dir/dna
        --cpg_model $models_dir/cpg
        --joint_model JointL2h512
        --train_models joint
        --out_dir $models_dir/joint
        --nb_epoch 10

``--dna_model`` and ``--cpg_model`` point to the output training
directory of the DNA and CpG model, respectively, which contains their
specification and weights:

.. code:: bash

    ls $models_dir/dna


.. parsed-literal::

    events.out.tfevents.1488213772.lawrence model.json
    lc_train.csv                            model_weights_train.h5
    lc_val.csv                              model_weights_val.h5
    model.h5


``model.json`` is the specification of the trained model,
``model_weights_train.h5`` the weights with the best performance on the
training set, and ``model_weights_val.h5`` the weights with the best
performance on the validation set. ``--dna_model ./dna`` is equivalent
to using ``--dna_model ./dna/model.json ./dna/model_weights_val.h5``,
i.e. the validation weights will be used. The training weights can be
used by ``--dna_model ./dna/model.json ./dna/model_weights_train.h5``

In the command above, we used ``--train_models joint`` to only train the
parameters of the Joint model without training the pre-trained DNA and
CpG model. Although this reduces training time, you might obtain better results
by also fine-tuning the parameters of the DNA and CpG model without using
``--train_models``.

.. _train_monitor:

Monitoring training progress
============================

To check if your model is training correctly, you should monitor the
training and validation loss. DeepCpG prints the loss and performance
metrics for each output to the console as you can see from the previous
commands. ``loss`` is the loss on the training set, ``val_loss`` the
loss on the validation set, and ``cpg/X_acc``, is, for example, the
accuracy for output cell X. DeepCpG also stores these metrics in
``X.csv`` in the training output directory.

Both the training loss and validation loss should continually decrease
until saturation. If at some point the validation loss starts to
increase while the training loss is still decreasing, your model is
overfitting the training set and you should stop training. DeepCpG will
automatically stop training if the validation loss does not increase
over the number of epochs that is specified by ``--early_stopping`` (by
default 5). If your model is overfitting already after few epochs, your
training set might be to small, and you could try to regularize your
model model by choosing a higher value for ``--dropout`` or
``--l2_decay``.

If your training loss fluctuates or increases, then you should decrease
the learning rate. For more information on interpreting learning curves
I recommend this tutorial.

To stop training before reaching the number of epochs specified by
``--nb_epoch``, you can create a :ref:`stop file <train_time>` (default name ``STOP``) in
the training output directory with ``touch STOP``.

Watching numeric console outputs is not particular user friendly.
`TensorBoard <https://www.tensorflow.org/get_started/summaries_and_tensorboard>`__
provides a more convenient and visually appealing way to mointor
training. You can use TensorBoard provided that you are using the
:ref:`Tensorflow backend <train_backend>`. Simply go to the training output
directory and run ``tensorboard --logdir .``.

.. _train_time:

Deciding how long to train
==========================

The arguments ``--nb_epoch`` and ``--early_stopping`` control how long
models are trained.

``--nb_epoch`` defines the maximum number of training epochs (default
30). After one epoch, the model has seen the entire training set once.
The time per epoch hence depends on the size of the training set, but
also on the complexity of the model that you are training and the
hardware of your machine. On a large dataset, you have to train for
fewer epochs than on a small dataset, since your model will have seen
already a lot of training samples after one epoch. For training on about
3,000,000 samples, good default values are 20 for the DNA and CpG
model, and 10 for the Joint model.

Early stopping stops training if the loss on the validation set did not
improve after the number of epochs that is specified by
``--early_stopping`` (default 5). If you are training without specifying
a validation set with ``--val_files``, early stopping will be
deactivated.

``--max_time`` sets the maximum training time in hours. This guarantees
that training terminates after a certain amount of time regardless of
the ``--nb_epoch`` or ``--early_stopping`` argument.

``--stop_file`` defines the path of a file that, if it exists, stop
training after the end of the current epoch. This is useful if you are
monitoring training and want to terminate training manually as soon as
the training loss starts to saturate regardless of ``--nb_epoch`` or
``--early_stopping``. For example, when using
``--stop_file ./train/STOP``, you can create an empty file with
``touch ./train/STOP`` to stop training at the end of the current epoch.

.. _train_hyper:

Optimizing hyper-parameters
===========================

DeepCpG has different hyper-parameters, such as the learning rate,
dropout rate, or model architectures. Although the performance of
DeepCpG is relatively robust to different hyper-parameters, you can
tweak performances by trying out different parameter combinations. For
example, you could train different models with different parameters on a
subset of your data, select the parameters with the highest performance
on the validation set, and then train the full model.

The following hyper-parameters are most important (default values shown):
1. Learning rate: ``--learning_rate 0.0001``
2. Dropout rate: ``--dropout 0.0``
3. DNA model architecture: ``--dna_model CnnL2h128``
4. Joint model architecture: ``--joint_model JointL2h512``
5. CpG model architecture: ``--cpg_model RnnL1``
6. L2 weight decay: ``--l2_decay 0.0001``

The learning rate defines how aggressively model parameters are updated
during training. If the training loss :ref:`changes only slowly <train_monitor>`,
you could try increasing the learning rate. If your model is overfitting
of if the training loss fluctuates, you should decrease the learning
rate. Reasonable values are 0.001, 0.0005, 0.0001, 0.00001, or values in
between.

The dropout rate defines how strongly your model is regularized. If you
have only few data and your model is overfitting, then you should
increase the dropout rate. Reasonable values are, e.g., 0.0, 0.2, 0.4.

DeepCpG provides different architectures for the DNA, CpG, and joint
model. Architectures are more or less complex, depending on how many
layers and neurons say have. More complex model might yield better
performances, but take longer to train and might overfit your data. You can find
more information about available model architecture :doc:`here <models>`.

L2 weight decay is an alternative to dropout for regularizing model
training. If your model is overfitting, you might try 0.001, or 0.005.

.. _train_test:

Testing training
================

``dcpg_train.py`` provides different arguments that allow to briefly
test training before training the full model for a about a day.

``--nb_train_sample`` and ``--nb_val_sample`` specify the number of
training and validation samples. When using ``--nb_train_sample 500``,
the training loss should briefly decay and your model should start
overfitting.

``--nb_output`` and ``--output_names`` define the maximum number and the
name of model outputs. For example, ``--nb_output 3`` will train only on
the first three outputs, and ``--output_names cpg/.*SER.*`` only on
outputs that include 'SER' in their name.

Analogously, ``--nb_replicate`` and ``--replicate_name`` define the
number and name of cells that are used as input to the CpG model.
``--nb_replicate 3`` will only use observed methylation states from the
first three cells, and allows to briefly test the CpG model.

``--dna_wlen`` specifies the size of DNA sequence windows that will be
used as input to the DNA model. For example, ``--dna_wlen 101`` will
train only on windows of size 101, instead of using the full window
length that was specified when creating data files with
``dcpg_data.py``.

Analogously, ``--cpg_wlen`` specifies the sum of the number of observed
CpG sites to the left and the right of the target CpG site for training
the CpG model. For example, ``--cpg_wlen 10`` will use 5 observed CpG
sites to the left and to the right.

.. _train_tune:

Fine-tuning and training selected components
============================================

``dcpg_train.py`` provides different arguments that allow to selectively
train only some components of a model.

With ``--fine_tune``, only the output layer will be trained. As the name
implies, this argument is useful for fine-tuning a pre-trained model.

``--train_models`` specifies which models are trained. For example,
``--train_models joint`` will train the Joint model, but not the DNA
and CpG model. ``--train_models cpg joint`` will train the CpG and
Joint model, but not the DNA model.

``--trainable`` and ``--not_trainable`` allow including and excluding
certain layers. For example,
``--not_trainable '.*' --trainable 'dna/.*_2'`` will only train the
second layers of the DNA model.

``--freeze_filter`` excludes the first convolutional layer of the DNA
model from training.

.. _train_backend:

Configuring the Keras backend
=============================

DeepCpG use the `Keras <https://keras.io>`__ deep learning library,
which supports `Theano <http://deeplearning.net/software/theano/>`__ or
`Tensorflow <https://www.tensorflow.org/>`__ as backend. While Theano
has long been the dominant deep learning library, Tensorflow is more
suited for parallelizing computations on multiple GPUs and CPUs, and
provides
`TensorBoard <https://www.tensorflow.org/get_started/summaries_and_tensorboard>`__
to interactively monitor training.

You can configure the backend by setting the ``backend`` attribute in
``~/.keras/keras.json`` to ``tensorflow`` or ``theano``. Alternatively
you can set the environemnt variable ``KERAS_BACKEND='tensorflow'`` to
use Tensorflow, or ``KERAS_BACKEND='theano'`` to use Theano.

You can find more information about Keras backends
`here <https://keras.io/backend/>`__.
