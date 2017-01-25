# DeepCpG

Python package for predicting single-cell CpG methylation states from DNA sequence and neighboring CpG sites using deep neural networks ([Angermueller et al., 2016](http://biorxiv.org/content/early/2016/05/27/055715)).

```
Angermueller, Christof, Heather Lee, Wolf Reik, and Oliver Stegle. “Accurate Prediction of Single-Cell DNA Methylation States Using Deep Learning.” bioRxiv, May 27, 2016, 055715. doi:10.1101/055715.
```


## Installation

Clone the DeepCpG repository into you current directory:

```
  git clone https://github.com/cangermueller/deepcpg.git
```

Install DeepCpG and its dependencies:

```
python setup.py install
```


## Getting started with DeepCpG in 30 seconds

1. Store known CpG methylation states of each cell into a tab-delimted file with the following columns:
   * Chromosome (without chr)
   * Position of the CpG site on the chromosome
   * Binary methylation state of the CpG sites (0=unmethylation, 1=methylated)

  Example:

  ```
  1   3000827   1.0
  1   3001007   0.0
  1   3001018   1.0
  ...
  Y   90829839  1.0
  Y   90829899  1.0
  Y   90829918  0.0
  ```


2. Run `dcpg_data.py` to create the input data for DeepCpG:

```
  dcpg_data.py
  --cpg_profiles ./cpg/cell1.tsv ./cpg/cell2.tsv ./cpg/cell3.tsv
  --dna_files ./dna/*.dna.chromosome.*.fa*
  --out_dir ./data
```

`./cpg/cell[123].tsv` store the methylation data from step 1., `dna` contains the DNA database, e.g. [mm10](http://ftp.ensembl.org/pub/release-85/fasta/mus_musculus/dna/) for mouse or [hg38](http://ftp.ensembl.org/pub/release-86/fasta/homo_sapiens/dna/) for human, and output data files will be stored in `./data`.

3. Fine-tune a pre-trained model or train your own model from scratch with `dcpg_train.py`:

```
  dcpg_train.py
    ./data/c{1,2,3}_*.h5
    --val_data ./data/c{10,11,13}_*.h5
    --dna_model CnnL2h128
    --cpg_model RnnL1
    --joint_model JointL2h512
    --nb_epoch 30
    --out_dir ./model
```

This command uses chromosomes 1-3 for training and 10-13 for validation. `dna_model`, `cpg_model`, and `joint_model` specify the architecture of the CpG, DNA, and joint model, respectively. Training will stop after at most 30 epochs and model files will be stored in `./model`.

4. Use `dcpg_eval.py` to predict missing methylation states and evaluate prediction performances:

```
  dcpg_eval.py
    ./data/c*.h5
    --model_files ./model/model.json ./model/model_weights_val.h5
    --out_data ./eval/data.h5
    --out_report ./eval/report.tsv
```

This command predicts missing methylation states of all cells and chromosomes and evaluates prediction performances using known methylation states. Predicted states will be stored in `./eval/data.h5` and performance metrics in `./eval/report.tsv`.



## Examples

Interactive examples on how to use DeepCpG can be found [here](./examples/README.md).



## Models

Pre-trained models can be downloaded from the [DeepCpG model zoo](docs/models.md).



## Content
* `/deepcpg/`: Source code
* `/docs`: Documentation
* `/examples/`: Examples for using DeepCpG
* `/script/`: Executable scripts for data creation, model training, and interpretation
* `/tests`: Test files



## Contact
* Christof Angermueller
* cangermueller@gmail.com
* https://cangermueller.com
