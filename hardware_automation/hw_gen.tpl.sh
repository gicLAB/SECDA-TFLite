#!/bin/bash
# set -e
pushd "$(dirname "$0")"

run_hls=1
run_hlx=1
run_board_script=0

# Default above, but we can change it based on the first argument
if [ $# -eq 1 ]; then
  run_hls=$1
fi

# if there are two arguments, then the second one is the debug flag
if [ $# -eq 2 ]; then
  run_hls=$1
  run_hlx=$2
fi

# if there are two arguments, then the second one is the debug flag
if [ $# -eq 3 ]; then
  run_hls=$1
  run_hlx=$2
  run_board_script=$3
fi

replace_string_in_files() {
  local directory=$1
  local search_string=$2
  local replace_string=$3
  local file_extension=$4

  find "$directory" -type f -name "*.${file_extension}" -exec sed -i "s/${search_string}/${replace_string}/g" {} +

  if [ $? -eq 0 ]; then
    echo "Replacement completed successfully."
  else
    echo "An error occurred during the replacement process."
  fi
}


if [ $run_hls == 1 ]; then 
  rm -rf ./vivado_hls*
  rm -rf ./&{acc_tag}
  cp &{acc_link_folder}/*.h ./src
  cp &{acc_link_folder}/*.cc ./src
  start_hls=`date +%s`
  &{vp_hls}/vivado_hls -f hls_script.tcl
  hls_exit=$?
  end_hls=`date +%s`
  if [ $hls_exit -ne 0 ]; then
      echo "--------------HLS FAILED--------------"
      curl --header 'Access-Token: o.eIEuBUZBIooNKzofTc6WATcyobjqK4TD' \
      --header 'Content-Type: application/json' \
      --data-binary '{"device_iden": "ujDnqxJ2S2Csjx4TEjgAtE","body":"HLS Error","title":"&{acc_tag}","type":"note"}' \
      --request POST \
      https://api.pushbullet.com/v2/pushes
      exit 1
  fi
  curl --header 'Access-Token: o.eIEuBUZBIooNKzofTc6WATcyobjqK4TD' \
  --header 'Content-Type: application/json' \
  --data-binary '{"device_iden": "ujDnqxJ2S2Csjx4TEjgAtE","body":"HLS Done","title":"&{acc_tag}","type":"note"}' \
  --request POST \
  https://api.pushbullet.com/v2/pushes
  
fi


if [ $run_hlx == 1 ]; then 
  start_hlx=`date +%s`
  rm -rf ./&{acc_tag}_hlx
  rm -rf ./Xil
  rm -rf ./generated_files
  rm -rf ./NA
  rm -rf ./vivado.*
  &{vp}/vivado -mode batch -source hlx_script.tcl -tclargs --origin_dir . --project_name &{acc_tag}_hlx --ip_repo ./&{acc_tag}/&{acc_tag}
  hlx_exit=$?
  end_hlx=`date +%s`
  if [ $hlx_exit -ne 0 ]; then
      echo "--------------HLX FAILED--------------"
      curl --header 'Access-Token: o.eIEuBUZBIooNKzofTc6WATcyobjqK4TD' \
      --header 'Content-Type: application/json' \
      --data-binary '{"device_iden": "ujDnqxJ2S2Csjx4TEjgAtE","body":"HLX Error","title":"&{acc_tag}","type":"note"}' \
      --request POST \
      https://api.pushbullet.com/v2/pushes
      exit 1
  fi
  curl --header 'Access-Token: o.eIEuBUZBIooNKzofTc6WATcyobjqK4TD' \
  --header 'Content-Type: application/json' \
  --data-binary '{"device_iden": "ujDnqxJ2S2Csjx4TEjgAtE","body":"HLX Done","title":"&{acc_tag}","type":"note"}' \
  --request POST \
  https://api.pushbullet.com/v2/pushes

  mkdir -p ./generated_files
  if [ "&{hlx_version}" = "2024" ] ; then
    cp ./&{acc_tag}_hlx/&{acc_tag}_hlx.gen/sources_1/bd/design_1/hw_handoff/design_1.hwh ./generated_files/&{acc_tag}.hwh
  else
    cp ./&{acc_tag}_hlx/&{acc_tag}_hlx.srcs/sources_1/bd/design_1/hw_handoff/design_1.hwh ./generated_files/&{acc_tag}.hwh
  fi
  cp ./&{acc_tag}_hlx/&{acc_tag}_hlx.runs/impl_1/design_1_wrapper.bit ./generated_files/&{acc_tag}.bit
  cp ./&{acc_tag}_hlx/utilization_report_impl_full.txt ./generated_files/
  # cp ./&{acc_tag}_hlx/utilization_report_impl_ip.txt ./generated_files/
  cp ./&{acc_tag}_hlx/timing_report_impl_full.txt ./generated_files/
  # cp ./&{acc_tag}_hlx/timing_report_impl_ip.txt ./generated_files/
  cp ./generated_files/&{acc_tag}.hwh ../../../src/secda_benchmark_suite/bitstreams/&{board}/
  cp ./generated_files/&{acc_tag}.bit ../../../src/secda_benchmark_suite/bitstreams/&{board}/

  # create the directory on the board
  ssh -q -t -p &{board_port} &{board_user}@&{board_hostname} "mkdir -p &{board_dir}"
  rsync -r -av -e 'ssh -p &{board_port}' ./generated_files/&{bitstream}.bit &{board_user}@&{board_hostname}:&{board_dir}/
  rsync -r -av -e 'ssh -p &{board_port}' ./generated_files/&{bitstream}.hwh &{board_user}@&{board_hostname}:&{board_dir}/
fi

# if [ $run_board_script == 1 ]; then 
#   # load the bitstream
#   ssh -t -p &{board_port} &{board_user}@&{board_hostname} "sudo python3 ~/load_bitstream.py &{board_dir}/&{bitstream}.bit"
#   start_run=`date +%N`
#   ssh -t -p &{board_port} &{board_user}@&{board_hostname} "chmod +x &{board_script}; sudo &{board_script}"
#   end_run=`date +%N`
# fi

hls_runtime=$((end_hls-start_hls))
hlx_runtime=$((end_hlx-start_hlx))
run_runtime=$((end_run-start_run))
echo ""
echo "HLS runtime: $(($hls_runtime / 60)):$(($hls_runtime % 60)) mins"
echo "HLX runtime: $(($hlx_runtime / 60)):$(($hlx_runtime % 60)) mins"
echo "Run runtime: $(($run_runtime/1000)) ms"

popd