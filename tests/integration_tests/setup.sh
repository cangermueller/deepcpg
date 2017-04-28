#!/usr/bin/env bash

function run {
  cmd=$@
  echo
  echo "#################################"
  echo $cmd
  echo "#################################"
  eval $cmd
  if [[ $check -ne 0 && $? -ne 0 ]]; then
    1>&2 echo "Command failed!"
    exit 1
  fi
}

data_dir="./data"
run "mkdir -p $data_dir"

dna_db="$data_dir/dna_db"
if [[ ! -e $dna_db ]]; then
  run "mkdir -p $dna_db"
  cmd="wget
    -P $dna_db
    ftp://ftp.ensembl.org/pub/release-85/fasta/mus_musculus/dna/Mus_musculus.GRCm38.dna.chromosome.{18,19}.fa.gz
    "
  run $cmd
fi


out_dir="$data_dir/data"
run "rm -rf $out_dir"
cmd="dcpg_data.py
  --dna_files $dna_db
  --cpg_profiles $(ls $data_dir/cpg_profiles/*)
  --anno_files $(ls $data_dir/annos/*)
  --cpg_cov 1
  --cpg_stats mean mode var cat_var cat2_var diff
  --cpg_stats_cov 1
  --dna_wlen 501
  --cpg_wlen 50
  --chromos 18 19
  --chunk_size 5000
  --out_dir $data_dir/data"
run $cmd
