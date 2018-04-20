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
data_host="http://www.ebi.ac.uk/~angermue/deepcpg/alias"

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
  key=$1
  out_dir=$2

  if [[ -e $out_dir ]]; then
    return
  fi

  run "wget $data_host/$key -O $out_dir.zip"
  run "unzip -o $out_dir.zip -d $out_dir"
  run "rm $out_dir.zip"
}


# Genome
download_genome "mm10" "ftp://ftp.ensembl.org/pub/release-85/fasta/mus_musculus/dna/Mus_musculus.GRCm38.dna.chromosome.*.fa.gz"

# CpG profiles
if [[ ! -e "$data_dir/cpg" ]]; then
  download_zip "b3afd7f831dec739d20843a3ef2dbeff" "$data_dir/cpg"
  run "gunzip $data_dir/cpg/*gz"
fi

# Motif database
motif_file="motif_databases.12.15.tgz"
if [[ ! -e $data_dir/motif_databases ]]; then
  run "wget http://meme-suite.org/meme-software/Databases/motifs/$motif_file -O $data_dir/$motif_file"
  run "tar xf $data_dir/$motif_file -C $data_dir"
  run "rm $data_dir/$motif_file"
fi


# Annotations
if [[ ! -e "$data_dir/anno" ]]; then
  echo "If the following command fails, download 'anno.zip' manually from the following link and extract to './data/anno':"
  echo "https://drive.google.com/open?id=1rjQLshQZi1KdGSs-HUIB8vyPHOIYprkL"
  download_zip 8c336f759e7010fa7a8287576281110e "$data_dir/anno"
fi
