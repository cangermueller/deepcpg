#!/usr/bin/env bash

set -e
shopt -s extglob

check=1
function run {
  cmd=$@
  echo
  echo "#################################"
  echo $cmd
  echo "#################################"
  eval $cmd
  if [ $check -ne 0 -a $? -ne 0 ]; then
    1>&2 echo "Command failed!"
    exit 1
  fi
}

data_dir="./data"
data_url="http://www.ebi.ac.uk/~angermue/deepcpg"

function download_genome {
  name=$1
  url=$2

  out_dir="$data_dir/dna/$name"
  if [[ -d $out_dir ]]; then
    return
  fi

  run "mkdir -p $out_dir"
  run "wget $url -P $out_dir"
}

function download_zip {
  url=$1
  out_file=$2

  run "wget $url -O $out_file"
  run "unzip -o $out_file -d $(dirname $out_file)"
  run "rm $out_file"
}


download_genome "mm10" "ftp://ftp.ensembl.org/pub/release-85/fasta/mus_musculus/dna/Mus_musculus.GRCm38.dna.chromosome.*.fa.gz"

download_zip "${data_url}/examples/data/cpg.zip" "$data_dir/cpg.zip"
run "gunzip $data_dir/cpg/*gz"
