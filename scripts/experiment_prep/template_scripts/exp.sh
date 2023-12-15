#!/bin/bash

pushd ${path_to_tf}
bazel build --config=elinux_armhf -c opt //${del_path}:${bazel_name} --copt="-DACC_PROFILE" --copt="-DTFLITE_ENABLE_XNNPACK=OFF" --copt="-DTFLITE_WITHOUT_XNNPACK" --copt="-DACC_NEON"
popd

rm -rf ${output_path}
mkdir -p ${output_path}
cp ${path_to_tf}/bazel-out/armhf-opt/bin/${del_path}/${bazel_name} ${output_path}/${app_short}_${trial_name}

# copy hw files to output folder
cp ${hw_path}/${acc_tag}.bit ${output_path}/
cp ${hw_path}/${acc_tag}.hwh ${output_path}/

# send experiment to board
rsync -r -av -e 'ssh -p ${board_port}' ${output_path} ${board_user}@${board_hostname}:${pynq_dir}/

# load the bitstream
ssh -t -p ${board_port} ${board_user}@${board_hostname} "cd ${pynq_dir}/${exp_folder} && sudo python3 ~/load_bitstream.py ${acc_tag}.bit"

# run experiment on board
ssh -t -p ${board_port} ${board_user}@${board_hostname} "cd ${pynq_dir}/${exp_folder} && chmod +x ./${app_short}_${trial_name} && sudo ${run_cmd}"
