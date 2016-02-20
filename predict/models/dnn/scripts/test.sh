#!/usr/bin/env bash

model_dir="../train"
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


model_file="$model_dir/model.pkl"
if [ ! -e $model_file ]; then
  cmd="$src_dir/convert.py
       $model_dir/model.json $model_dir/model_weights.h5
       -p $model_file"
  run $cmd
fi

cmd="$src_dir/test.py
  ../data/w201_k20_val.h5
  --model $model_file
  -o $out_dir/test.h5
  --nb_sample 1000
  "
run $cmd
