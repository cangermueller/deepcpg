#!/usr/bin/env bash


if [ -z "$VIRTUAL_ENV" ]; then
  source virtualenvwrapper.sh
  workon predict
fi

out_file="data_shuffled.h5"
rm -rf $out_file

cmd="python $Pv/data.py
  $Cseq
  $Cdata/stats/ser.h5
  --stats w3000_var w1000_var
  --out_file $out_file
  --annos_file $Cannos
  --annos loc_H3K4me1
  --chromos 14 5
  --nb_sample 1000
  --seq_len 25
  --chunk_out 500
  --shuffle
  "
eval $cmd
