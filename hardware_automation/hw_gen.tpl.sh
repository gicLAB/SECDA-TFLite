#!/bin/bash
# set -e
pushd "$(dirname "$0")"

run_hls=1
run_hlx=1
offload_hls_hlx=0

# Default above, but we can change it based on the first argument
if [ $# -eq 1 ]; then
  run_hls=$1
fi

# if there are two arguments, then the second one is the debug flag
if [ $# -eq 2 ]; then
  run_hls=$1
  run_hlx=$2
fi

# if there are three arguments, then the second one is the offload flag
if [ $# -eq 3 ]; then
  run_hls=$1
  run_hlx=$2
  offload_hls_hlx=$3
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

send_pushbullet_notification() {
  local message="$1"
  curl -s -o /dev/null --header 'Access-Token: &{pb_token}' \
    --header 'Content-Type: application/json' \
    --data-binary "{\"device_iden\": \"ujDnqxJ2S2Csjx4TEjgAtE\",\"body\":\"${message}\",\"title\":\"&{acc_tag}\",\"type\":\"note\"}" \
    --request POST \
    https://api.pushbullet.com/v2/pushes
  push=$?
  echo "Pushbullet response: $push"
}

if [ $run_hls == 1 ]; then 
  rm -rf ./vivado_hls*
  rm -rf ./&{acc_tag}
  cp &{acc_link_folder}/*.h ./src
  cp &{acc_link_folder}/*.cc ./src
  start_hls=`date +%s`
  
  if [ $offload_hls_hlx == 1 ]; then
    # check server directory is available in the server
    ssh -q -t -p &{server_port} &{gateway} &{server_user}@&{server_hostname} "mkdir -p &{server_dir}"
    rsync -r -av -e 'ssh -p &{server_port} &{gateway}' ./src  &{server_user}@&{server_hostname}:&{server_dir}/&{acc_tag}/
    rsync -r -av -e 'ssh -p &{server_port} &{gateway}' hls_script.tcl &{server_user}@&{server_hostname}:&{server_dir}/&{acc_tag}/
    rsync -r -av -e 'ssh -p &{server_port} &{gateway}' hlx_script.tcl &{server_user}@&{server_hostname}:&{server_dir}/&{acc_tag}/

    ## run the script on the remote server after moving to the server directory
    echo "Running HLS at server..."
    ssh -t -p "&{server_port}" "&{gateway}" "&{server_user}@&{server_hostname}" "
      cd \"&{server_dir}/&{acc_tag}\" &&&& \
      rm -rf ./vivado_hls* &&&& \
      rm -rf ./&{acc_tag} &&&& \
      &{svp_hls}/vivado_hls -f hls_script.tcl; \
      hls_exit=\$? &&&& \
      echo \"HLS started at: \$start_hls\" &&&& \
      echo \"HLS ended at: \$end_hls\" &&&& \
      echo \"HLS exit status: \$hls_exit\" &&&& \
      exit \$hls_exit
    " &{stderr} | tee outputHLS.log
    hls_exit=$(grep -q "HLS exit status: 1" outputHLS.log &&&& echo 1 || echo 0)
    if [ $hls_exit == 0 ]; then
      ## copy the generated HLS project files back to the local machine
      rsync -r -av -e 'ssh -p &{server_port} &{gateway}' &{server_user}@&{server_hostname}:&{server_dir}/&{acc_tag}/&{acc_tag} ./      
    fi
  else
    # run HLS locally
    echo "Running HLS locally..."
    &{vp_hls}/vivado_hls -f hls_script.tcl
    hls_exit=$?
  fi
  end_hls=`date +%s`
  if [ $hls_exit -ne 0 ]; then
      echo "--------------HLS FAILED--------------"
      send_pushbullet_notification "HLS Error"
      exit 1
  fi
  echo "--------------HLS PASSED--------------"
  send_pushbullet_notification "HLS Done"

  # Fix for the floating point IP version needed for 2019.2 HLS to 2024.1 HLX
  if [ "&{board}" = "KRIA" ] ; then
    echo "Kria Copy And Replace"
    replace_string_in_files "./&{acc_tag}/&{acc_tag}/impl/ip/" "floating_point_v7_1_9" "floating_point_v7_1_18" "vhd"
  fi
  
fi


if [ $run_hlx == 1 ]; then 
  start_hlx=`date +%s`
  rm -rf ./&{acc_tag}_hlx
  rm -rf ./Xil
  rm -rf ./generated_files
  rm -rf ./NA
  rm -rf ./vivado.*

  if [ $offload_hls_hlx == 1 ]; then
    # check server directory is available in the server
    ssh -q -t -p &{server_port} &{gateway} &{server_user}@&{server_hostname} "mkdir -p &{server_dir}"
    rsync -r -av -e 'ssh -p &{server_port} &{gateway}' ./src &{server_user}@&{server_hostname}:&{server_dir}/&{acc_tag}/
    rsync -r -av -e 'ssh -p &{server_port} &{gateway}' hls_script.tcl &{server_user}@&{server_hostname}:&{server_dir}/&{acc_tag}/
    rsync -r -av -e 'ssh -p &{server_port} &{gateway}' hlx_script.tcl &{server_user}@&{server_hostname}:&{server_dir}/&{acc_tag}/

    ## run the script on the remote server after moving to the server directory
    echo "Running HLX at server..."
    ssh -t -p "&{server_port}" "&{gateway}" "&{server_user}@&{server_hostname}" "
      cd \"&{server_dir}/&{acc_tag}\" &&&& \
      rm -rf ./&{acc_tag}_hlx &&&& \
      rm -rf ./Xil &&&& \
      rm -rf ./generated_files &&&& \
      rm -rf ./NA &&&& \
      rm -rf ./vivado.* &&&& \
      &{svp_hlx}/vivado -mode batch -source hlx_script.tcl -tclargs --origin_dir . --project_name &{acc_tag}_hlx --ip_repo ./&{acc_tag}/&{acc_tag}; \
      hlx_exit=\$? &&&& \
      echo \"HLX started at: \$start_hlx\" &&&& \
      echo \"HLX ended at: \$end_hlx\" &&&& \
      echo \"HLX exit status: \$hlx_exit\" &&&& \
      exit \$hlx_exit" &{stderr} | tee outputHLX.log
    hlx_exit=$(grep -q "HLX exit status: 1" outputHLX.log &&&& echo 1 || echo 0)
    end_hlx=`date +%s`
    if [ $hlx_exit == 0 ]; then
      echo "HLX completed successfully."
      ## copy the generated HLX project files back to the local machine
      rsync -r -av -e 'ssh -p &{server_port} &{gateway}' &{server_user}@&{server_hostname}:&{server_dir}/&{acc_tag}/&{acc_tag}_hlx ./
    fi
  else
    # run HLX locally
    echo "Running HLX locally..."
    &{vp}/vivado -mode batch -source hlx_script.tcl -tclargs --origin_dir . --project_name &{acc_tag}_hlx --ip_repo ./&{acc_tag}/&{acc_tag}
    hlx_exit=$?
  fi
  end_hlx=`date +%s`
  if [ $hlx_exit -ne 0 ]; then
      echo "--------------HLX FAILED--------------"
      send_pushbullet_notification "HLX Error"
      exit 1
  fi
  echo "--------------HLX PASSED--------------"
  send_pushbullet_notification "HLX Done"

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
  cp ./generated_files/&{acc_tag}.hwh ../../../src/benchmark_suite/bitstreams/&{board}/
  cp ./generated_files/&{acc_tag}.bit ../../../src/benchmark_suite/bitstreams/&{board}/

 # check board is connected to the network
  timeout --foreground 60 ssh -q -t -p &{board_port} &{board_user}@&{board_hostname} "echo 2>&&1" &&&& board_connected=1 || board_connected=0
  if [ $board_connected == 1 ]; then
    echo "Board connected: $board_connected"
    echo "Transferring bitstream and HWH to the board"
    # create the directory on the board
    ssh -q -t -p &{board_port} &{board_user}@&{board_hostname} "mkdir -p &{board_dir}"
    rsync -r -av -e 'ssh -p &{board_port}' ./generated_files/&{bitstream}.bit &{board_user}@&{board_hostname}:&{board_dir}/
    rsync -r -av -e 'ssh -p &{board_port}' ./generated_files/&{bitstream}.hwh &{board_user}@&{board_hostname}:&{board_dir}/
  else
    echo "Board is not connected to the network. Skipping bitstream transfer."
  fi
fi


hls_runtime=$((end_hls-start_hls))
hlx_runtime=$((end_hlx-start_hlx))
run_runtime=$((end_run-start_run))
echo ""
echo "HLS runtime: $(($hls_runtime / 60)):$(($hls_runtime % 60)) mins"
echo "HLX runtime: $(($hlx_runtime / 60)):$(($hlx_runtime % 60)) mins"
echo "Run runtime: $(($run_runtime/1000)) ms"

popd