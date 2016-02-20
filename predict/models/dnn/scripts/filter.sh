#!/usr/bin/env bash

model_dir="../train"
out_dir="."
mkdir -p $out_dir

src_dir=$Pd
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


model_file="$model_dir/model_cpu.pkl"
if [ ! -e $model_file ]; then
  cmd="$src_dir/convert.py
       $model_dir/model.json $model_dir/model_weights.h5
       -p $model_file"
  run $cmd
fi

cmd="$src_dir/filter_export.py
  $model_dir/model_cpu.pkl
  -o filter_weights.h5
  "
run $cmd

cmd="$src_dir/filter_viz.R
  $out_dir/filter_weights.h5
  --weights /s_c1/weights
  -o $out_dir/motifs.pdf
  "
run $cmd

cmd="python -u $src_dir/filter_act.py
  $data_file
  --model $model_file
  --nb_sample 20000
  -o $out_dir/filter_act.h5
"
run $cmd

cmd="$src_dir/filter_motifs.py
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

cmd="$src_dir/eval_filter_act.py
  $out_dir/filter_act.h5
  --sql_file $out_dir/filter_act.sql
  --annos_file $Cannos
  --annos '^loc_.*' '^licr_.*' '.*H3.*'
  --stats_file $Cstats
"
run $cmd

cmd="$Pv/R/run_filter_act.R
  $out_dir/filter_act.sql
  -o $out_dir/filter_act.html
  "
run $cmd

rmd_file="$out_dir/filter_act.Rmd"
cp $Pvr/filter_act.Rmd $rmd_file &&
Rscript -e "library(methods); rmarkdown::run('$rmd_file'$args)"
