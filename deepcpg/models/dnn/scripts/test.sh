#!/usr/bin/env bash

train_dir=$(find ../ -type 'd' -name 'train*' | sort | tail -n 1)
out_dir="."
src_dir="$Pd"
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


if [ -n "$(ls /dev/nvidia* 2> /dev/null)" ]; then
  model_file="$train_dir/model.pkl"
else
  model_file="$train_dir/model_cpu.pkl"
fi

if [ ! -e $model_file ]; then
  cmd="$src_dir/convert.py
       $train_dir/model.json $train_dir/model_weights.h5
       -p $model_file"
  run $cmd
fi

cmd="python -u $Pd/test.py
  $E2/data/w501_test.h5
  --model $model_file
  -o $out_dir/test.h5
  --batch_size 128
  "
run $cmd

model=$(basename $(dirname $(readlink -f $out_dir)))
cmd="eval.py
  ./test.h5
  --sql_meta model=$model eval=$Elabel trial=1
  --chromos $Etest_chromos
  --annos_file $Cannos
  --annos '^loc_' 'H3' 'dnase'
  --stats_file $Cstats
  --sql_file $out_dir/test.sql
  "
run $cmd
