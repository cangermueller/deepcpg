#!/usr/bin/env bash

model_dir="../train"
out_dir="."
mkdir -p $out_dir

if [ -z "$VIRTUAL_ENV" ]; then
  source virtualenvwrapper.sh
  workon predict
fi

fheck=1
function run {
  cmd=$@
  echo
  echo "#################################"
  echo $cmd
  echo "#################################"
  eval $cmd
  if [ $check -ne 0 -a $? -ne 0 ]; then
    1>&2 echo "Command failed!"
    exit 1
  fi
}


model_file="$model_dir/model.pkl"
if [ ! -e $model_file ]; then
  cmd="$src_dir/pkl.py
       $model_dir/model.json $model_dir/model_weights.h5
       -o $model_file"
  run $cmd
fi

cmd="$Pd/test.py
  $Ev/data/2iser_w501_test.h5
  --model $model_file
  -o $out_dir/test.h5
  "
run $cmd

cmd="rm -f test_sliced.h5 &&
  $Pd/pred_slice.py
  $out_dir/test.h5
  -o $out_dir/test_sliced.h5
  --annos_file $Cannos
  --annos '^loc_' '^licr_' 'H3'
  --nb_sample 100000
  "
run $cmd

model=$(basename $(dirname $(readlink -f .)))
cmd="$Pd/eval_var.py
  $out_dir/test.h5
  --sql_file $out_dir/test.sql
  --sql_meta model=$model
  --annos_file $Cannos
  --annos '^loc_' '^licr_' 'H3'
  "
run $cmd

cmd="rmd.py
  $Pd/R/eval_var.Rmd
  --copy eval_var.Rmd
  "
run $cmd
