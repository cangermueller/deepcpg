#!/usr/bin/env bash


if [ -z "$VIRTUAL_ENV" ]; then
  source virtualenvwrapper.sh
  workon predict
fi

targets="$Ctargets/ESC_2i_10_RSC25_8.h5
  $Ctargets/ESC_2i_11_RSC26_1.h5
  $Ctargets/ESC_2i_12_RSC26_2.h5"

python ../data.py $targets \
  --chromos 5 \
  --max_samples 1000 \
  --out_file data.h5 \
  --chunk_size 333 \
  --seq_file $Cseq
