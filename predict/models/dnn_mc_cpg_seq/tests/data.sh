#!/usr/bin/env bash


# source virtualenvwrapper.sh
# workon predict

  #../../../data/CSC4_8G.h5 \
python ../../src/data.py \
  ../../../../data/ESC_2i_10_RSC25_8.h5 \
  ../../../../data/ESC_2i_11_RSC26_1.h5 \
  ../../../../data/ESC_2i_12_RSC26_2.h5 \
  --max_samples 1000 \
  --out_file data.h5 \
  --seq_file ../../../../data/seq_w1001.h5 \
  --chromos 5
