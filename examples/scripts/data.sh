#!/usr/bin/env bash

# Creates DeepCpG data files.


# Source dependencies.
source "./lib.sh"

# Set to 1 for testing and 0 for real run.
test_mode=1
# Directory with CpG profiles.
cpg_dir="../data/cpg"
# Directory with DNA sequences.
dna_dir="../data/dna/mm10"

# Create data files.
cmd="dcpg_data.py
  --cpg_profiles $cpg_dir/*.tsv
  --dna_files $dna_dir
  --out_dir $data_dir
  --dna_wlen 1001
  --cpg_wlen 50
  "
if [[ $test_mode -eq 1 ]]; then
  cmd="$cmd --nb_sample 1000"
fi
run $cmd

# Compute statistics, e.g. the total number of CpG sites and the mean
# methylation rate of each cell. Change the input `./data/*` to
# `./data/c{1,3,5}*.h5` to compute statistics for a subset of the data, which is
# useful for deciding how to split the data into training, validation, and test
# set.
cmd="dcpg_data_stats.py $data_dir/* | tee $data_dir.txt"
run $cmd
