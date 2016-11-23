DeepCpG
=======

Python package for predicting single-cell CpG methylation states from DNA
sequence and neighboring CpG sites using multi-task convolutional neural
networks.

***This package is still under heavy development and will be documented soon.***

Installation
------------

The DeepCpG git repository can be checked out into the current directory by
executing the command

``git clone https://github.com/cangermueller/deepcpg2.git``.

To install DeepCpG, execute `python setup.py install` in the root directory.

Usage
-----
Use `dcpg_data.py` to generate the input data, `dcpg_train.py` to train models,
and `dcpg_eval.py` to evaluate trained models.

Content
-------
* `/deepcpg/`: DeepCpG package
* `/script/`: Scripts for data generation, model training, and imputation

Contact
-------
* Christof Angermueller
* cangermueller@gmail.com
* https://cangermueller.com
