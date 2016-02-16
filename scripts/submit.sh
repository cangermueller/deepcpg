#!/usr/bin/env bash

script=$1

job=$(basename $script)
job=${job%\.*}
cmd="sbatch -J $job -o $job.out -e $job.err -A STEGLE-SL3-GPU
  --time 4:00:00 $sgpu $script"
eval $cmd
