#!/bin/bash

# Requires jq to be installed
board_user=$(jq -r '.board_user' ../../config.json)
board_hostname=$(jq -r '.board_hostname' ../../config.json)
board_dir=$(jq -r '.board_dir' ../../config.json)
board_port=$(jq -r '.board_port' ../../config.json)
conda_path=$(jq -r '.conda_path' ../../config.json)
bench_dir=${board_dir}/apps_eval_suite

helpFunction() {
  echo ""
  echo "Usage: $0 -s -b -c -p -t -i -n name"
  echo -e "\t-s Skip running experiment"
  echo -e "\t-b Generate binaries"
  echo -e "\t-c Skip inference difference checks"
  echo -e "\t-p Power collection"
  echo -e "\t-t Test run"
  echo -e "\t-i Initialize the board"
  echo -e "\t-n Name of the experiment"
  exit 1 # Exit script after printing help
}

# Optional arguments
# 1: skip_bench: Skip running benchmarks on the target board
# 2: bin_gen: Generate binaries
# 3: skip_inf_diff: Skip inference difference checks on target board and fpga
# 4: collect_power: Collect power data // Needs to be used in conjunction with collect_power.sh
skip_bench=0
bin_gen=0
skip_inf_diff=0
process_on_fpga=0
collect_power=0
test_run=0
init=0
name=""
now=$(date +"%Y_%m_%d_%H_%M")

while getopts hsbcptin: flag; do
  case $flag in
  h)
    helpFunction
    exit
    ;;
  s) skip_bench=1 ;;
  b) bin_gen=1 ;;
  c) skip_inf_diff=1 ;;
  p) collect_power=1 ;;
  t) test_run=1 ;;
  i) init=1 ;;
  n) name=$OPTARG ;;
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
echo "-- SECDA-TFLite Benchmark Suite --"
echo "-----------------------------------------------------------"
echo "Configurations"
echo "--------------"
echo "Board User: ${board_user}"
echo "Board Hostname: ${board_hostname}"
echo "Board Dir: ${board_dir}"
echo "Skip Bench: ${skip_bench}"
echo "Bin Gen: ${bin_gen}"
echo "Skip Inf Diff: ${skip_inf_diff}"
echo "Collect Power: ${collect_power}"
echo "Test Run: ${test_run}"
echo "Name: ${name}"
echo "-----------------------------------------------------------"

# define function to which create secda_benchmark_suite directory on the board at board_dir
function create_dir() {
  ssh -o LogLevel=QUIET -t -p $board_port $board_user@$board_hostname "mkdir -p $board_dir  && mkdir -p $board_dir/bitstreams && mkdir -p $bench_dir/bins && mkdir -p $bench_dir/models"
  rsync -q -r -avz -e 'ssh -p '${board_port} ./scripts/check_valid.py $board_user@$board_hostname:$bench_dir/
  rsync -q -r -avz -e 'ssh -p '${board_port} ./scripts/load_bitstream.py $board_user@$board_hostname:~/
  rsync -r -avz -e 'ssh -p '${board_port} ./model_gen/models  $board_user@$board_hostname:$bench_dir/

  rsync -q -r -avz -e 'ssh -p '${board_port} ./scripts/fpga_scripts/ $board_user@$board_hostname:$board_dir/scripts/
  rsync -r -avz -e 'ssh -p '${board_port} ${data_dir}  $board_user@$board_hostname:$board_dir/
  rsync -r -avz -e 'ssh -p '${board_port} ${bitstream_dir}  $board_user@$board_hostname:$board_dir/
  echo "Initialization Done"
}

echo "-----------------------------------------------------------"
echo "Initializing SECDA-TFLite Benchmark Suite"
echo "-----------------------------------------------------------"
if [ $skip_bench -eq 0 ]; then
echo "Clearing cache"
rm -rf ./tmp
fi

if [ $init -eq 1 ]; then
  create_dir
fi
echo "-----------------------------------------------------------"

# Generate binaries and experiment configurations
echo "-----------------------------------------------------------"
echo "Configuring Benchmark"
echo "-----------------------------------------------------------"
python3 scripts/configure_benchmark.py $bin_gen

if [ $bin_gen -eq 1 ]; then
  echo "-----------------------------------------------------------"
  echo "Generating Binaries"
  source ./generated/gen_bins.sh
  echo "-----------------------------------------------------------"
fi

source ./generated/configs.sh
length=${#hw_array[@]}
source ${conda_path}/activate tf # current the tf environment can#t be used for bazel build (glibc version issue)

if [ $skip_bench -eq 0 ]; then
  echo "-----------------------------------------------------------"
  echo "Transferring Experiment Configurations to Target Device"
  rsync -q -r -avz -e 'ssh -p '$board_port ./generated/configs.sh $board_user@$board_hostname:$bench_dir/
  rsync -q -r -avz -e 'ssh -p '$board_port ./generated/run_collect.sh $board_user@$board_hostname:$bench_dir/
  ssh -o LogLevel=QUIET -t -p $board_port $board_user@$board_hostname "cd $bench_dir/ && chmod +x ./*.sh"

  if [ $collect_power -eq 1 ]; then
    echo "-----------------------------------------------------------"
    echo "Initializing Power Capture"
    python3 scripts/record_power.py $name &
    echo $! >/tmp/record_power.py.pid
    echo "-----------------------------------------------------------"
  fi

  echo "-----------------------------------------------------------"
  echo "Running Experiments"
  echo "-----------------------------------------------------------"
  ssh -o LogLevel=QUIET -t -p $board_port $board_user@$board_hostname "cd $bench_dir/ && ./run_collect.sh $process_on_fpga $skip_inf_diff $collect_power $test_run"
  # if not test run, then collect results
  if [ $test_run -eq 0 ]; then
    echo "Transferring Results to Host"
    rsync -q -r -av -e 'ssh -p '$board_port $board_user@$board_hostname:$bench_dir/tmp ./
  fi
  echo "-----------------------------------------------------------"

  # echo "Runs can be found in:" ./tmp/runs.csv
  if [ $collect_power -eq 1 ]; then
    if [[ -e /tmp/record_power.py.pid ]]; then
      kill $(cat /tmp/record_power.py.pid)
      echo "-----------------------------------------------------------"
      echo "Power Capture Done"
      echo "-----------------------------------------------------------"
      echo "Processing Power Data"
      echo "-----------------------------------------------------------"

      python3 scripts/process_power.py $name $length
      echo "Power Processing Done"
      echo "-----------------------------------------------------------"
    else
      echo $(cat /tmp/record_power.py.pid) "not found"
    fi
    echo "Simple csv:" ./files/${name}_clean.csv
    echo "-----------------------------------------------------------"
  fi
fi

# Post processing on host
if [ $test_run -eq 0 ]; then
  echo "-----------------------------------------------------------"
  echo "Post Processing"
  prev_failed=0
  prev_hw=""
  length=${#hw_array[@]}
  for ((i = 0; i < length; i++)); do
    index=$((i + 1))
    HW=${hw_array[$i]}
    MODEL=${model_array[$i]}
    THREAD=${thread_array[$i]}
    NUM_RUN=${num_run_array[$i]}
    VERSION=${version_array[$i]}
    DEL_VERSION=${del_version_array[$i]}
    DEL=${del_array[$i]}
    prev_failed=0
    prev_hw=${HW}
    runname=${HW}_${VERSION}_${DEL}_${DEL_VERSION}_${MODEL}_${THREAD}_${NUM_RUN}
    if [ $((${i} % 1)) -eq 0 ]; then
      echo "========================================================================"
      echo "${runname} ${index}/${length}"
      echo "========================================================================"
    fi

    # Check verify correctness of accelerator
    valid=1
    if [ ${HW} != "CPU" ] && [ "${skip_inf_diff}" -eq 0 ]; then {
      if [ ! -f tmp/${runname}_id.txt ]; then
        valid=0 && echo "Correct Check File Missing for: ${runname}"
      else
        python3 scripts/check_valid.py tmp/${runname}_id.txt
        if [ $? -ne 0 ]; then valid=0 && echo "Correctness Check Failed ${runname}"; fi
      fi
    }; fi
    # echo "Processing run"
    python3 scripts/process_run.py ${MODEL} ${THREAD} ${NUM_RUN} ${HW} ${VERSION} ${DEL} ${DEL_VERSION} ${valid} ${name}
    if [ $? -ne 0 ]; then prev_failed=1 && echo "Process Run Failed" && continue; fi
  done # HW

  python3 scripts/process_all_runs.py ${name}
  echo "-----------------------------------------------------------"
  echo "Post Processing Done"
  echo "-----------------------------------------------------------"
fi

echo "-----------------------------------------------------------"
echo "Exiting SECDA-TFLite Benchmark Suite"
echo "-----------------------------------------------------------"
