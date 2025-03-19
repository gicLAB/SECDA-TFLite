import sys

sys.dont_write_bytecode = True

import os
from benchmark_utils import *


supported_tools = {
    "benchmark_model": "bm",
    "inference_diff": "id",
    # "eval_model": "em",
}

cpu_paths = {
    "benchmark_model": ["tensorflow/lite/tools/benchmark", "benchmark_model"],
    "inference_diff": [
        "tensorflow/lite/tools/evaluation/tasks/inference_diff",
        "run_eval",
    ],
    "eval_model": ["tensorflow/lite/examples/secda_apps/eval_model", "eval_model"],
}


# bb_po = "--copt='-DSECDA_LOGGING_DISABLED' --cxxopt='-march=armv7-a' --cxxopt='-mfpu=neon' --cxxopt='-funsafe-math-optimizations' --cxxopt='-ftree-vectorize' --copt='-DACC_PROFILE' --define tflite_with_xnnpack=false --copt='-DTFLITE_ENABLE_XNNPACK=OFF' --copt='-DTFLITE_WITHOUT_XNNPACK' --copt='-DACC_NEON'"
# bb_po = "--copt='-DSECDA_LOGGING_DISABLED' --copt='-DACC_PROFILE' --define tflite_with_xnnpack=false --copt='-DTFLITE_ENABLE_XNNPACK=OFF' --copt='-DTFLITE_WITHOUT_XNNPACK'"

# elinux_armhf
bb_pr = "bazel6 build --config=elinux_armhf -c opt //"
bb_po = "--copt='-DSECDA_LOGGING_DISABLED' --copt='-DACC_PROFILE' --define tflite_with_xnnpack=false --copt='-DTFLITE_ENABLE_XNNPACK=OFF' --copt='-DTFLITE_WITHOUT_XNNPACK' --copt='-DACC_NEON' --@secda_tools//:config=fpga"

# elinux_aarch64
# bb_pr = "bazel6 build --config=elinux_aarch64 -c opt //"
# bb_po = "--copt='-DSECDA_LOGGING_DISABLED' --copt='-DACC_PROFILE' --define tflite_with_xnnpack=false --copt='-DTFLITE_ENABLE_XNNPACK=OFF' --copt='-DTFLITE_WITHOUT_XNNPACK' --copt='-DACC_NEON' --copt='-DKRIA'"


def gen_bins(sc, exp, board_config):
    output_path = f"{sc['out_dir']}/gen_bins.sh"
    board = board_config["board"]
    board_user = board_config["board_user"]
    board_hostname = board_config["board_hostname"]
    board_port = board_config["board_port"]
    board_dir = board_config["board_dir"]
    path_to_tf = sc["secda_tflite_path"] + "/tensorflow"
    rdel_path = sc["path_to_dels"]
    cpu_type = "aarch64-opt" if board == "KRIA" else "armhf-opt"

    ## JDOC: This part generate the binaries for the different boards
    bb_pr = "bazel6 build --config=elinux_armhf -c opt //"
    bb_po = "--copt='-DSECDA_LOGGING_DISABLED' --copt='-DACC_PROFILE' --define tflite_with_xnnpack=false --copt='-DTFLITE_ENABLE_XNNPACK=OFF' --copt='-DTFLITE_WITHOUT_XNNPACK' --copt='-DACC_NEON'"
    if board == "Z1":
        bb_pr = "bazel6 build --config=elinux_armhf -c opt //"
        bb_po = "--copt='-DSECDA_LOGGING_DISABLED' --copt='-DACC_PROFILE' --define tflite_with_xnnpack=false --copt='-DTFLITE_ENABLE_XNNPACK=OFF' --copt='-DTFLITE_WITHOUT_XNNPACK' --copt='-DACC_NEON' --@secda_tools//:config=fpga"
    elif board == "KRIA":
        bb_pr = "bazel6 build --config=elinux_aarch64 -c opt //"
        bb_po = "--copt='-DSECDA_LOGGING_DISABLED' --copt='-DACC_PROFILE' --define tflite_with_xnnpack=false --copt='-DTFLITE_ENABLE_XNNPACK=OFF' --copt='-DTFLITE_WITHOUT_XNNPACK' --copt='-DACC_NEON' --copt='-DKRIA'"
        # bb_po = "--copt='-DSECDA_LOGGING_DISABLED' --copt='-DACC_PROFILE' --define tflite_with_xnnpack=false --copt='-DTFLITE_ENABLE_XNNPACK=OFF' --copt='-DTFLITE_WITHOUT_XNNPACK' --copt='-DKRIA'"


    delegates_needed = {}
    for hw in exp[1]:
        hw_config_file = find_hw_config(
            f"{sc['secda_tflite_path']}/{sc['hw_configs']}", hw
        )
        hw_config = load_config(hw_config_file)
        curr_delegate = hw_config["del"]
        curr_version = hw_config["del_version"]
        if curr_delegate not in delegates_needed:
            delegates_needed[curr_delegate] = [curr_version]
        else:
            if curr_version not in delegates_needed[curr_delegate]:
                delegates_needed[curr_delegate].append(curr_version)

    script = "#!/bin/bash\n"
    script += "set -e\n"
    script += f"pushd {path_to_tf}\n"
    for delegate, vers in delegates_needed.items():
        for ver in vers:
            for tool, sn in supported_tools.items():
                del_path = f"{rdel_path}/{delegate}/v{ver}"
                # check if path exists
                if not os.path.exists(sc["secda_tflite_path"] + "/" + del_path):
                    del_path = f"{rdel_path}/{delegate}"
                del_path = del_path[del_path.index("/") + 1 :]

                name = f"{sn}_{delegate}_{ver}"
                bin_name = f"{tool}_plus_{delegate}"

                if delegate == "cpu":
                    del_path = cpu_paths[tool][0]
                    bin_name = cpu_paths[tool][1]

                script += f"{bb_pr}{del_path}:{bin_name} {bb_po} \n"
                

                script += f"rsync -r -avz -e 'ssh -p {board_port}' {path_to_tf}/bazel-out/{cpu_type}/bin/{del_path}/{bin_name} {board_user}@{board_hostname}:{board_dir}/benchmark_suite/bins/{name}\n" 

    script += f"ssh -t -p {board_port} {board_user}@{board_hostname} 'cd {board_dir}/benchmark_suite/bins/ && chmod 775 ./*'\n"
    script += "popd\n"
    # create folder to output
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w") as f:
        f.write(script)
    os.system(f"chmod +x {output_path}")
