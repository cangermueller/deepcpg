#!/usr/bin/env bash

script=$1
hours=${2:-2}

job=$(basename $script)
job=${job%\.*}
cmd="sbatch -J $job -o $script.out -e $script.err -A STEGLE-SL3
  --time $hours:00:00 $scpu $script"
eval $cmd
