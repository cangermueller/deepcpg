# DeepCpG examples

The following tutorials illustrate how to use DeepCpG for imputing and analysing single-cell methylation data. Tutorials are implemented as [IPython notebooks](https://ipython.org/notebook.html) and can be executed interactively. They can be exported to shell scripts and executed in the terminal by selecting `File -> Download as -> Bash (.sh)`. This is recommended for large-scale experiments, e.g. training models on the entire data set.

`setup.sh` must be execute once to setup the required dependencies:

```shell
bash setup.sh
```

## Tutorials

* [DeepCpG basics](./src/basics/index.ipynb): Pre-processing data, training models, and evaluating models.
* [Fine-tuning](./src/fine_tune/index.ipynb): Fine-tuning a pre-trained model to speed-up training.
* [Motif analysis](./src/motifs/index.ipynb): Visualizing and analyzing learned motifs.

... more tutorials to come!
