#!/usr/bin/env bash

target_dirs=$@
if [ -z "$target_dirs" ]; then
  1>&2 echo No target directories given!
  exit 1
fi

out_file='test.h5'
rm -f $out_file*

for target_dir in $target_dirs; do
  target=$(basename $target_dir)
  in_file="$target_dir/test/test.h5"
  groups=$(h5ls $in_file | cut -f 1 -d ' ')
  for group in $groups; do
    cmd="h5copy -i $in_file -o $out_file -s /$group -d /$target/$group -p"
    echo $cmd
    eval $cmd
  done
done
