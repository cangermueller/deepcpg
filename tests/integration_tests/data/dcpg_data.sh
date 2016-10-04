#/usr/bin/env bash


out_dir="./data"
cpg_files=$(ls ./cpg_files/*bed)

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

cmd="rm -rf $out_dir && mkdir -p $out_dir"
cmd="$cmd && dcpg_data.py
  --dna_db ./dna_db
  --cpg_files $(ls ./cpg_files/BS27_4_SER.bed ./cpg_files/BS28_2_SER.bed)
  --dna_wlen 501
  --cpg_wlen 50
  --chunk_size 5000
  --out_dir $out_dir"
run $cmd
