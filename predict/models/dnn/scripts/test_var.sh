#!/usr/bin/env bash

out_dir="."
mkdir -p $out_dir

if [ -z "$VIRTUAL_ENV" ]; then
  source virtualenvwrapper.sh
  workon predict
fi

check=1
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


train_dir=$(find ../ -type 'd' -name 'train*' | sort | tail -n 1)
if [ -e /dev/nvidea0 ]; then
  model_file="$train_dir/model.pkl"
else
  model_file="$train_dir/model_cpu.pkl"
fi

if [ ! -e $model_file ]; then
  cmd="$Pd/convert.py
       $train_dir/model.json $train_dir/model_weights.h5
       -p $model_file"
  run $cmd
fi

cmd="python -u $Pd/test.py
  $Ev/data/2iser_w501_test.h5
  --model $model_file
  -o $out_dir/test.h5
  --batch_size 512
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
