#!/usr/bin/env bash

script=$1

job=$(basename $script)
job=${job%\.*}
cmd="sbatch -J $job -o $job.out -e $job.err -A STEGLE-SL3
  --time 2:00:00 $scpu $script"
eval $cmd
