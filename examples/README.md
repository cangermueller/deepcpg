# DeepCpG examples

Here you can find IPython notebooks and shell scripts that illustrate how to use DeepCpG.

To setup the required dependencies, execute `setup.sh`:

```shell
bash setup.sh
```

## Notebooks
`./notebooks` contains [IPython notebooks](https://ipython.org/notebook.html), which can be executed interactively. They can be exported to shell scripts and executed in the terminal by selecting `File -> Download as -> Bash (.sh)`. This is recommended for large-scale experiments, e.g. training models on the entire data set.

* [DeepCpG basics](./notebooks/basics/index.ipynb): Pre-processing data, training models, and evaluating models.
* [Fine-tuning](./notebooks/fine_tune/index.ipynb): Fine-tuning a pre-trained model to speed-up training.
* [Motif analysis](./notebooks/motifs/index.ipynb): Visualizing and analyzing learned motifs.
* [Mutations effects](./notebooks/snp/index.ipynb): Computing and visualizing mutations effects.
* [Predicting statistics](./notebooks/stats/index.ipynb): Predicting statistics such as cell-to-cell variance.

## Shell scripts
`./scripts` contains shell scrips with recommended default parameters. They may help you to easily build a DeepCpG pipeline for creating data, training models, and evaluating models. Set `test_mode` variable in scripts to `1` for testing, and `0` otherwise.

* [lib.sh](./scripts/lib.sh): Global variable definitions and functions.
* [data.sh](./scripts/data.sh): Creates DeepCpG data files.
* [train.sh](./scripts/train.sh): Trains DeepCpG DNA, CpG, and Joint model separately.
* [eval.sh](./scripts/eval.sh): Evaluates prediction performances and imputes methylation profiles.
