# Stop execution if one command fails.
set -e

# Where data files are stored.
data_dir="./data"
# Where models are stored.
models_dir="./models"
# Where imputed methylation profiles are stored.
eval_dir="./eval"

# Helper function for running commands
function run {
  cmd=$@
  echo
  echo "#################################"
  echo $cmd
  echo "#################################"
  # Uncomment next line to only print commands without executing
  eval $cmd
  if [[ $? -ne 0 ]]; then
    1>&2 echo "Command failed!"
    exit 1
  fi
}
