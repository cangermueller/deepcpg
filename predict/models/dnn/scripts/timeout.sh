#!/usr/bin/env bash

model_dirs=$@
for model_dir in $model_dirs; do
  log_file="$model_dir/train/train.sh.out"
  if [[ ! -e $log_file ]]; then
    continue
  fi
  h=$(grep '^Stop training' $log_file)
  if [ -n "$h" ]; then
    echo $model_dir
  fi
done
