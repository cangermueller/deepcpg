#!/usr/bin/env bash

# Evaluates model prediction performances and imputes methylation profiles.


# Source dependencies.
source "./lib.sh"

# Set to 1 for testing and 0 for real run.
test_mode=1

run "rm -rf $eval_dir"
run "mkdir -p $eval_dir"

# Evaluate model and impute methylation profiles. You can change the input model
# to `dna` or `cpg` for evaluation the DNA or CpG model, respectively.
cmd="dcpg_eval.py
  $data_dir/*.h5
  --model $models_dir/joint
  --out_data $eval_dir/data.h5
  --out_report $eval_dir/report.csv
  "
if [[ $test_mode -eq 1 ]]; then
  cmd="$cmd --nb_sample 1000"
fi
run $cmd


# Export imputed methylation profile. You can use `-f bedGraph` to export to
# gzip-compressed bedGraph files, which, however, is slower.
cmd="dcpg_eval_export.py
  $eval_dir/data.h5
  -o $eval_dir/hdf
  -f hdf
  "
if [[ $test_mode -eq 1 ]]; then
  cmd="$cmd --nb_sample 1000"
fi
run $cmd
