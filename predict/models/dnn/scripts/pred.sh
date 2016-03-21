#!/usr/bin/env bash

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


train_dir=$(find ../ -type 'd' -name 'train*' | sort | tail -n 1)
if [ -n "$(ls /dev/nvidia* 2> /dev/null)" ]; then
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

cmd="$Pv/pred.py
  $Ev/data/2iser_w501_val.h5
  --model $train_dir/model_cpu.pkl
  -o $out_dir/pred.h5
  --nb_sample 100000
  "
run $cmd

cmd="$Pv/pred_slice.py
  $out_dir/pred.h5
  -o $out_dir/pred_sliced.h5
  --annos_file $Cannos
  --annos '^loc_.*' '^licr_.*' '.*H3.*'
  --sample 10000
  --targets ser_w3000_var 2i_w3000_var
  "
run $cmd

model=$(basename $(dirname $(readlink -f .)))
echo $model
cmd="$Pv/eval_pred.py
  $out_dir/pred.h5
  --sql_file $out_dir/pred.sql
  --sql_meta model=$model
  --annos_file $Cannos
  --annos '^loc_.*' '^licr_.*' '.*H3.*'
  "
run $cmd
