#!/usr/bin/env bash

# Trains DeepCpG models separately.

# Source dependencies.
source "./lib.sh"

# Change the following glob patterns depending on how to you want to split the
# data into training, validation, and optionally test set. The training set
# should contain at least 3 million CpG sites. Use `dcpg_data_stats.py` to count
# the number of CpG sites (see `data.sh` script).

# Training set
train_files="$data_dir/c{1,3,5,7,9}*.h5"
# Validation set
val_files="$data_dir/c{13,14,15,16}*.h5"

# Set to 1 for testing and 0 for real run.
test_mode=1

if [[ $test_mode -eq 1 ]]; then
  val_files="$train_files"
fi


# Train DNA model.
cmd="dcpg_train.py
  $train_files
  --val_files $val_files
  --out_dir $models_dir/dna
  --dna_model CnnL2h128
  "
if [[ $test_mode -eq 1 ]]; then
  cmd="$cmd
    --val_files $train_files
    --nb_train_sample 500
    --nb_val_sample 500
    --nb_epoch 1
    "
else
  cmd="$cmd
    --nb_epoch 30
    "
fi
run $cmd


# Train CpG model.
cmd="dcpg_train.py
  $train_files
  --val_files $val_files
  --out_dir $models_dir/cpg
  --cpg_model RnnL1
  "
if [[ $test_mode -eq 1 ]]; then
  cmd="$cmd
    --val_files $train_files
    --nb_train_sample 500
    --nb_val_sample 500
    --nb_epoch 1
    "
else
  cmd="$cmd
    --nb_epoch 20
    "
fi
run $cmd


# Train Joint model.
cmd="dcpg_train.py
  $train_files
  --val_files $val_files
  --out_dir ./models/joint
  --dna_model ./models/dna
  --cpg_model ./models/cpg
  --joint_model JointL2h512
  --train_models joint
  "
if [[ $test_mode -eq 1 ]]; then
  cmd="$cmd
    --val_files $train_files
    --nb_train_sample 500
    --nb_val_sample 500
    --nb_epoch 1
    "
else
  cmd="$cmd
    --nb_epoch 10
    "
fi
run $cmd
