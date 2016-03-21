#!/usr/bin/env bash

out_dir="."
mkdir -p $out_dir

train_dir=$(find ../ -type 'd' -name 'train*' | sort | tail -n 1)
data_file="$E2d/w501_train.h5"

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

cmd="python -u $Pd/filter_act.py
  $data_file
  --model $model_file
  --conv_node s_h1a
  --nb_sample 50000
  --outputs act
  -o $out_dir/filter_act.h5
"
run $cmd

method='tsne'
cmd="$Pd/tsne.py
  $out_dir/filter_act.h5
  --annos_file $Cannos
  --stats_file $Cstats
  --nb_sample 30000
  --method $method
  -o $out_dir/tsne_$method.h5
  "
run $cmd
