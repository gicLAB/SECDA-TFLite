#!/bin/bash

# Requires jq to be installed
board_user=$(jq -r '.board_user' ../../config.json)
board_hostname=$(jq -r '.board_hostname' ../../config.json)
board_dir=$(jq -r '.board_dir' ../../config.json)
board_port=$(jq -r '.board_port' ../../config.json)
conda_path=$(jq -r '.conda_path' ../../config.json)
host_data_dir=$(jq -r '.data_dir' ../../config.json)

# usually board_eval_dir should be ${board_dir}/apps_eval_suite
# board_eval_dir=${board_dir}/apps_eval_suite
# since we want to create evaluation suite seperate form benchmark-suite and
# board directory currently contains "Workspace/secda_benchmark_suite"
board_eval_dir=$(dirname "${board_dir}")/secda_apps_evaluation_suite

# board_data_dir will be place where model and data will be copied from the host
# it should be ${board_dir}/data
# but currently data is within the /secda_benchmark_suite
# folder name is same as host_data_dir
board_data_dir=$(dirname "${board_dir}")/data



helpFunction() {
  echo ""
  echo "Usage: $0 -j aec_path -i -b -c -l -p -n name"
  echo -e "\t-j apps evaluation config.json file"
  echo -e "\t-i Initialize the board"
  echo -e "\t-b Generate binaries"
  echo -e "\t-c Copy BitStreams to Boards"
  echo -e "\t-l Do not Load BitStreams to FPGA"
  echo -e "\t-p Power collection"
  echo -e "\t-n Name of the experiment"
  exit 1 # Exit script after printing help
}

# Optional arguments
# bin_gen: Generate binaries
aec_path="" ## apps evaluation config.json file , give the default path
init=0
name=""
bin_gen=0
cpy_bst=0
ld_bst=1
collect_power=0
now=$(date +"%Y_%m_%d_%H_%M")

while getopts "hj:in:bclp" flag; do
  case $flag in
    h)
      helpFunction
      exit
      ;;
    j) aec_path=$OPTARG ;;
    i) init=1 ;;
    n) name=$OPTARG ;;
    b) bin_gen=1 ;;
    c) cpy_bst=1 ;;
    l) ld_bst=0 ;;
    p) collect_power=1 ;;
    :)
      echo "Missing argument for option -$OPTARG"
      exit 1
      ;;
    \?)
      helpFunction
      exit 1
      ;;
  esac
done
shift $((OPTIND - 1))

if [ "$aec_path" == "" ]; then
  aec_path="configs/default_exp.json"
fi

if [ "$name" == "" ]; then
  name="evaluation_${now}"
else
  name="${name}_${now}"
fi

function ctrl_c() {
  echo "Exiting"
  exit 1
}


trap ctrl_c INT
echo "-----------------------------------------------------------"
echo "-- SECDA-TFLite Evaluation Suite --"
echo "-----------------------------------------------------------"
echo "Configurations"
echo "--------------"
echo "  board_user: $board_user"
echo "  board_hostname: $board_hostname"
echo "  board_dir: $board_dir"
echo "  board_port: $board_port"
echo "  conda_path: $conda_path"
echo "  host_data_dir: $host_data_dir"
echo "  board_eval_dir: $board_eval_dir"
echo "  board_data_dir: $board_data_dir"
echo "  Name: ${name}"
echo "  AEC Path: ${aec_path}"
echo "  Init: ${init}"
echo "  Bin Gen: ${bin_gen}"
echo "  Copy BitStreams: ${cpy_bst}"
echo "  Load BitStreams: ${ld_bst}"
echo "  Collect Power: ${collect_power}"
echo "  Time: ${now}"
echo "-----------------------------------------------------------"

# create a directory in the host in the path of the script
mkdir -p ./results
# create subdirectory in the name of ${name}
if [ -d "./results/${name}" ]; then
  rm -rf "./results/${name}"
fi
mkdir -p "./results/${name}"
#copy the aec_path file to the results directory
cp "$aec_path" "./results/${name}/"

# define function to which create secda_benchmark_suite directory on the board at board_dir
function create_dir() {
  ssh -o LogLevel=QUIET -t -p $board_port $board_user@$board_hostname "mkdir -p $board_eval_dir  && mkdir -p $board_eval_dir/scripts && mkdir -p $board_eval_dir/bitstreams && mkdir -p $board_eval_dir/bins"
  rsync -q -r -avz -e 'ssh -p '${board_port} ./scripts/fpga_scripts/ $board_user@$board_hostname:$board_eval_dir/scripts/
  echo "Transferring Data to Target Device ..."
  # analyze the host_data_dir and return the first level subdirectories to give the user option to select which subdirectory to sync
  subdirs=($(find "${host_data_dir}" -mindepth 1 -maxdepth 1 -type d))
  echo "Select subdirectories to sync (separate numbers by space, or type 'all' for all):"
  for i in "${!subdirs[@]}"; do
    echo "$((i+1))) ${subdirs[$i]}"
  done
  read -p "Enter selection: " selection

  if [[ "$selection" == "all" ]]; then
    selected_subdirs=("${subdirs[@]}")
  else
    selected_subdirs=()
    for idx in $selection; do
      if [[ "$idx" =~ ^[0-9]+$ ]] && (( idx >= 1 && idx <= ${#subdirs[@]} )); then
        selected_subdirs+=("${subdirs[$((idx-1))]}")
      fi
    done
  fi

  for subdir in "${selected_subdirs[@]}"; do
    echo "Syncing $subdir ..."
    rsync -r -avz -e 'ssh -p '${board_port} "$subdir" $board_user@$board_hostname:$board_data_dir/
  done
  echo "Initialization Done"
}

echo "-----------------------------------------------------------"
echo "Initializing SECDA-TFLite Evaluation Suite"
echo "-----------------------------------------------------------"

## need to check board is connected or not
## run a py file for that.

if [ $init -eq 1 ]; then
  create_dir
fi
echo "-----------------------------------------------------------"

source ${conda_path}/activate secda-tflite
# Generate binaries and experiment configurations
echo "-----------------------------------------------------------"
echo "Process Apps Configurations"
echo "-----------------------------------------------------------"
python3 scripts/process_config.py $aec_path $bin_gen $board_eval_dir $cpy_bst $ld_bst

#copy generated config files to ./results
cp -r "./generated" "./results/${name}/"
# exit 0

if [ $bin_gen -eq 1 ]; then
  echo "-----------------------------------------------------------"
  echo "Generating Binaries"
  source ./generated/gen_bins.sh
  echo "-----------------------------------------------------------"
fi


source ./generated/configs.sh
length=${#hw_array[@]}

echo "-----------------------------------------------------------"
echo "Transferring Experiment Configurations to Target Device"
rsync -q -r -avz -e 'ssh -p '$board_port ./generated/configs.sh $board_user@$board_hostname:$board_eval_dir/
rsync -q -r -avz -e 'ssh -p '$board_port ./generated/run_collect.sh $board_user@$board_hostname:$board_eval_dir/
ssh -o LogLevel=QUIET -t -p $board_port $board_user@$board_hostname "cd $board_eval_dir/ && chmod +x ./*.sh"


echo "-----------------------------------------------------------"
echo "Running Experiments"
echo "-----------------------------------------------------------"
ssh -o LogLevel=QUIET -t -p $board_port $board_user@$board_hostname "cd $board_eval_dir/ && ./run_collect.sh $collect_power"

echo "-----------------------------------------------------------"
echo "Transferring Results to Host"
echo "-----------------------------------------------------------"

# rsync -q -r -av -e 'ssh -p '$board_port $board_user@$board_hostname:$board_eval_dir/tmp ./
rsync -q -av -e 'ssh -p '$board_port $board_user@$board_hostname:$board_eval_dir/tmp/* ./results/${name}/
echo "-----------------------------------------------------------"

echo "Analyzing Results to Host"
python3 scripts/result_analysis.py \
  --results-dir "./results/${name}" \
  --config ./results/${name}/generated/configs.sh

python3 scripts/process_latency.py \
--results-dir "./results/${name}" \
--index "./results/${name}/index.csv"

python3 scripts/process_power.py \
  --results-dir "./results/${name}" \
  --index "./results/${name}/index.csv"


echo "-----------------------------------------------------------"
echo "Exiting SECDA-TFLite Evaluation Suite"
echo "-----------------------------------------------------------"