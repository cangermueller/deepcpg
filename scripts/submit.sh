#!/usr/bin/env bash

script=$1

job=$(basename $script)
job=${job%\.*}
cmd="sbatch -J $job -o $script.out -e $script.err -A STEGLE-SL3-GPU
  --time 3:00:00 $sgpu $script"
eval $cmd
