#!/usr/bin/env bash

model_files=$@

if [ -z "$model_files" ]; then
  >&2 echo "No models provided!"
  exit 1
fi

parallel=1
data_base="$Ev/data/2iser_w401"
run=$(basename $(pwd))

if [ -z "$VIRTUAL_ENV" ]; then
  source virtualenvwrapper.sh
  workon predict
fi

function train_model {
  model_file=$1
  model=$(basename $model_file)
  model=${model%.*}
  data=$(basename $data_base)
  out_dir="train/${data}_${model}"
  rm -rf $out_dir
  mkdir -p $out_dir
  max_time=6
  cmd="python -u $Pv/train.py
    ${data_base}_train.h5
    --val_file ${data_base}_val.h5
    --params $model_file
    --targets
      ser_w1000_var ser_w2000_var ser_w3000_var ser_w4000_var ser_w5000_var
      2i_w1000_var 2i_w2000_var 2i_w3000_var 2i_w4000_var 2i_w5000_var
    --nb_sample 100000
    --nb_val_sample 100000
    --nb_epoch 20
    --early_stop 3
    --lr_schedule 1
    --max_time $max_time
    --batch_size 128
    --out_dir $out_dir"
  if [ $parallel -eq 1 ]; then
    job_name=$(basename $out_dir)
    cmd="sbatch --mem=32000 -J $job_name --time=$max_time:00:00
    -A STEGLE-SL3-GPU -o $out_dir/train.out -e $out_dir/train.err $sgpu $cmd"
  fi
  eval $cmd
}

for model_file in $model_files; do
  train_model $model_file
done
