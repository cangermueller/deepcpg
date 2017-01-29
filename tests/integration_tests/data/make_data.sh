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

dna_db="./dna_db"
if [[ ! -e $dna_db ]]; then
  run "mkdir -p $dna_db"
  cmd="wget
    -P $dna_db
    ftp://ftp.ensembl.org/pub/release-85/fasta/mus_musculus/dna/Mus_musculus.GRCm38.dna.chromosome.{18,19}.fa.gz
    "
  run $cmd
fi

cmd="rm -rf $out_dir && mkdir -p $out_dir"
cmd="$cmd && dcpg_data.py
  --dna_files ./dna_db
  --cpg_profiles $(ls ./cpg_files/BS27_4_SER.bed.gz ./cpg_files/BS28_2_SER.bed.gz)
  --bulk_profiles $(ls ./cpg_files/BS9N_2I.bed.gz ./cpg_files/BS9N_SER.bed.gz)
  --anno_files $(ls ./annos/*)
  --cpg_cov 1
  --stats mean mode var cat_var cat2_var diff
  --stats_cov 1
  --dna_wlen 501
  --cpg_wlen 50
  --chunk_size 5000
  --chromos 18 19
  --out_dir $out_dir"
run $cmd
