#!/usr/bin/env bash

out_dir="."
mkdir -p $out_dir

train_dir=$(find ../ -type 'd' -name 'train*' | sort | tail -n 1)
data_file="$E2d/w501_train.h5"
motif_dbs="$Pmotifs/CIS-BP/Mus_musculus.meme
  $Pmotifs/JASPAR/JASPAR_CORE_2016_vertebrates.meme
  $Pmotifs/MOUSE/uniprobe_mouse.meme
  $Pmotifs/MOUSE/chen2008.meme"

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

cmd="$Pd/filter_export.py
  $train_dir/model_cpu.pkl
  -o filter_weights.h5
  "
run $cmd

cmd="$Pd/filter_viz.R
  $out_dir/filter_weights.h5
  --weights /s_c1/weights
  -o $out_dir/motifs.pdf
  "
run $cmd

cmd="python -u $Pd/filter_act.py
  $data_file
  --model $model_file
  --nb_sample 30000
  -o $out_dir/filter_act.h5
"
run $cmd

cmd="$Pd/filter_motifs.py
  $out_dir/filter_act.h5
  -o $out_dir/motifs
  "
run $cmd

cmd="tomtom -dist pearson -thresh 0.1 -oc
  $out_dir/tomtom
  $out_dir/motifs/filters_meme.txt
  $motif_dbs"
check=0
run $cmd
check=1

cmd="tomtom_format.py $out_dir/tomtom/tomtom.txt
  -m $motif_dbs
  -o $out_dir/tomtom/tomtom.csv"
run $cmd

cmd="rmd.py
  $Pd/R/filter_motifs.Rmd
  --copy filter_motifs.Rmd
  "
run $cmd

fun="mean"
cmd="python -u $Pd/filter_act.py
  $data_file
  --model $model_file
  --outputs act z
  --fun $fun
  -o $out_dir/filter_act_$fun.h5
"
run $cmd

cmd="$Pd/eval_filter_act.py
  $out_dir/filter_act_$fun.h5
  --sql_file $out_dir/filter_act_$fun.sql
  --annos_file $Cannos
  --annos 'loc_' 'licr_' 'H3' 'dnase'
"
run $cmd

cmd="rmd.py
  $Pd/R/filter_act.Rmd
  --copy filter_act.Rmd
  "
run $cmd
