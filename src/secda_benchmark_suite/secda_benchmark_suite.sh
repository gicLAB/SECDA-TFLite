#!/bin/bash
skip_bench=0
bin_gen=0
process_local=0
skip_inf_diff=0

if [ $# -eq 1 ]; then
  skip_bench=$1
fi

if [ $# -eq 2 ]; then
  skip_bench=$1
  bin_gen=$2
fi

if [ $# -eq 3 ]; then
  skip_bench=$1
  bin_gen=$2
  skip_inf_diff=$3
fi

if [ $bin_gen -eq 1 ]; then
  source ./bin_gen.sh
fi

board_user=xilinx
board_hostname=jharis.ddns.net
arm_dir=/home/xilinx/Workspace/secda_benchmark_suite/


python3 scripts/configure_benchmark.py
if [ $skip_bench -eq 0 ]; then
  rsync -r -avz -e 'ssh -p 2202' ./generated/configs.sh $board_user@$board_hostname:$arm_dir/
  rsync -r -avz -e 'ssh -p 2202' ./generated/run_collect.sh $board_user@$board_hostname:$arm_dir/
  ssh -t -p 2202 $board_user@$board_hostname "cd $arm_dir/ && chmod +x ./*.sh"
  ssh -t -p 2202 $board_user@$board_hostname "cd $arm_dir/ && ./run_collect.sh $process_local $skip_inf_diff"
fi

now=$(date +"%Y_%m_%d_%H_%M")

# Post processing on host
if [ $process_local -eq 0 ]; then
  rsync -r -av -e 'ssh -p 2202' $board_user@$board_hostname:$arm_dir/tmp ./
  source ./generated/configs.sh
  prev_failed=0
  prev_hw=""
  length=${#hw_array[@]}
  for ((i = 0; i < length; i++)); do
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
    if [ $((${i} % 50)) -eq 0 ]; then
      echo "========================================================================"
      echo "${runname} ${i}/${length}"
      echo "========================================================================"
    fi

    # Check verify correctness of accelerator
    valid=1
    if [ ${HW} != "CPU" ] && [ "${skip_inf_diff}" -eq 0 ]; then {
      python3 scripts/check_valid.py tmp/${runname}_id.txt
      if [ $? -ne 0 ]; then valid=0 && echo "Correctness Check Failed ${runname}"; fi
    }; fi
    # echo "Processing run"
    python3 scripts/process_run.py ${MODEL} ${THREAD} ${NUM_RUN} ${HW} ${VERSION} ${DEL} ${DEL_VERSION} ${valid} ${now}
    if [ $? -ne 0 ]; then prev_failed=1 && echo "Process Run Failed" && continue; fi
  done # HW

  python3 scripts/process_all_runs.py ${now}
fi
