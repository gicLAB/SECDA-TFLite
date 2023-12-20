#!/bin/bash
load_bitstream=1
copy_hw=1
run_exp=1

# Default above, but we can change it based on the first argument
if [ $# -eq 1 ]; then
  load_bitstream=$1
fi

# if there are two arguments, then the second one is the debug flag
if [ $# -eq 2 ]; then
  load_bitstream=$1
  copy_hw=$2
fi

# if there are two arguments, then the second one is the debug flag
if [ $# -eq 3 ]; then
  load_bitstream=$1
  copy_hw=$2
  run_exp=$3
fi

set -e -o pipefail

pushd ^{path_to_tf}
bazel build --config=elinux_armhf -c opt //^{del_path}:^{bazel_name} --copt="-DACC_PROFILE" --copt="-DTFLITE_ENABLE_XNNPACK=OFF" --copt="-DTFLITE_WITHOUT_XNNPACK" --copt="-DACC_NEON"
popd

rm -rf ^{output_path}
mkdir -p ^{output_path}
cp ^{path_to_tf}/bazel-out/armhf-opt/bin/^{del_path}/^{bazel_name} ^{output_path}/^{app_short}_^{trial_name}

# copy hw files to output folder
if [ $copy_hw == 1 ]; then
cp ^{hw_path}/^{acc_tag}.bit ^{output_path}/
cp ^{hw_path}/^{acc_tag}.hwh ^{output_path}/
fi

# send experiment to board
rsync -r -av -e 'ssh -p ^{board_port}' ^{output_path} ^{board_user}@^{board_hostname}:^{pynq_dir}/

# load the bitstream
if [ $load_bitstream == 1 ]; then
ssh -t -p ^{board_port} ^{board_user}@^{board_hostname} "cd ^{pynq_dir}/^{exp_folder} && sudo python3 ~/load_bitstream.py ^{acc_tag}.bit"
fi


# run experiment on board
if [ $run_exp == 1 ]; then
ssh -t -p ^{board_port} ^{board_user}@^{board_hostname} "cd ^{pynq_dir}/^{exp_folder} && chmod +x ./^{app_short}_^{trial_name} && sudo ^{run_cmd}"
fi
