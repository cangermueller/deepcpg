#/usr/bin/env bash


out_dir="./data"
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

cmd="rm -rf $out_dir && mkdir -p $out_dir"
cmd="$cmd && dcpg_data.py
  --dna_db ./dna_db
  --cpg_profiles $(ls ./cpg_files/BS27_4_SER.bed ./cpg_files/BS28_2_SER.bed)
  --bulk_profiles $(ls ./cpg_files/BS9N_2I.bed ./cpg_files/BS9N_SER.bed)
  --anno_files $(ls ./annos/*bed)
  --cpg_cov 1
  --cpg_stats mean mode var cat_var cat2_var diff
  --cpg_stats_cov 1
  --dna_wlen 501
  --cpg_wlen 50
  --chunk_size 5000
  --chromos 18 19
  --out_dir $out_dir"
run $cmd
