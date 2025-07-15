#!/bin/bash

# Requires jq to be installed
board_user=$(jq -r '.board_user' ../../config.json)
board_hostname=$(jq -r '.board_hostname' ../../config.json)
board_dir=$(jq -r '.board_dir' ../../config.json)
board_port=$(jq -r '.board_port' ../../config.json)
conda_path=$(jq -r '.conda_path' ../../config.json)
data_dir=$(jq -r '.data_dir' ../../config.json)
bitstream_dir=$(jq -r '.bitstream_dir' ../../config.json)
eval_dir=${board_dir}/apps_eval_suite

helpFunction() {
  echo ""
  echo "Usage: $0 -j aec_path -i -n name"
  echo -e "\t-j apps evaluation config.json file"
  echo -e "\t-i Initialize the board"
  echo -e "\t-n Name of the experiment"
  echo -e "\t-b Generate binaries"
  exit 1 # Exit script after printing help
}

# Optional arguments
# bin_gen: Generate binaries
aec_path="" ## apps evaluation config.json file , give the default path
init=0
name=""
bin_gen=0
now=$(date +"%Y_%m_%d_%H_%M")

while getopts hj:in:b flag; do
  case $flag in
  h)
    helpFunction
    exit
    ;;
  j) aec_path=$OPTARG ;;
  i) init=1 ;;
  n) name=$OPTARG ;;

  b) bin_gen=1 ;;
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
  name="run_${now}"
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
echo "Evaluation Config File: ${aec_path}"
echo "Board User: ${board_user}"
echo "Board Hostname: ${board_hostname}"
echo "Board Evaluation Dir: ${eval_dir}"
echo "Name: ${name}"
echo "Bin Gen: ${bin_gen}"
echo "-----------------------------------------------------------"

# define function to which create secda_benchmark_suite directory on the board at board_dir
function create_dir() {
  ssh -o LogLevel=QUIET -t -p $board_port $board_user@$board_hostname "mkdir -p $eval_dir  && mkdir -p $board_dir/scripts && mkdir -p $board_dir/bitstreams && mkdir -p $eval_dir/bins"
  rsync -q -r -avz -e 'ssh -p '${board_port} ./scripts/fpga_scripts/ $board_user@$board_hostname:$board_dir/scripts/
  rsync -r -avz -e 'ssh -p '${board_port} ${data_dir}  $board_user@$board_hostname:$board_dir/
  rsync -r -avz -e 'ssh -p '${board_port} ${bitstream_dir}  $board_user@$board_hostname:$board_dir/
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

# Generate binaries and experiment configurations
echo "-----------------------------------------------------------"
echo "Process Apps Configurations"
echo "-----------------------------------------------------------"
python3 scripts/process_config.py $aec_path $bin_gen


if [ $bin_gen -eq 1 ]; then
  echo "-----------------------------------------------------------"
  echo "Generating Binaries"
  source ./generated/gen_bins.sh
  echo "-----------------------------------------------------------"
fi


source ./generated/configs.sh
length=${#hw_array[@]}
source ${conda_path}/activate tf # current the tf environment can#t be used for bazel build (glibc version issue)

echo "-----------------------------------------------------------"
echo "Transferring Experiment Configurations to Target Device"
rsync -q -r -avz -e 'ssh -p '$board_port ./generated/configs.sh $board_user@$board_hostname:$eval_dir/
rsync -q -r -avz -e 'ssh -p '$board_port ./generated/run_collect.sh $board_user@$board_hostname:$eval_dir/
ssh -o LogLevel=QUIET -t -p $board_port $board_user@$board_hostname "cd $eval_dir/ && chmod +x ./*.sh"


echo "-----------------------------------------------------------"
echo "Running Experiments"
echo "-----------------------------------------------------------"
ssh -o LogLevel=QUIET -t -p $board_port $board_user@$board_hostname "cd $eval_dir/ && ./run_collect.sh $process_on_fpga $skip_inf_diff $collect_power"

echo "Transferring Results to Host"
rsync -q -r -av -e 'ssh -p '$board_port $board_user@$board_hostname:$eval_dir/tmp ./
echo "-----------------------------------------------------------"


echo "-----------------------------------------------------------"
echo "Exiting SECDA-TFLite Evaluation Suite"
echo "-----------------------------------------------------------"