import os
supported_delegates = {
    # "vm_delegate": "1",
    # "secda_sa_delegate": "1",
    "mm2im_delegate": "1",
    # "mm2im_fpga_delegate": "1",
    # "cpu": "1"
}
supported_tools = {
    "benchmark_model": "bm",
    "inference_diff": "id",
    "eval_model": "em",
}

cpu_paths = {
    "benchmark_model": ["tensorflow/lite/tools/benchmark", "benchmark_model"],
    "inference_diff": ["tensorflow/lite/tools/evaluation/tasks/inference_diff", "run_eval"],
    "eval_model": ["tensorflow/lite/examples/secda_apps/eval_model", "eval_model"],
}


bb_pr = "bazel build --config=elinux_armhf -c opt //"
bb_po = "--cxxopt='-mfpu=neon' --copt='-DACC_PROFILE' --copt='-DTFLITE_ENABLE_XNNPACK=OFF' --copt='-DTFLITE_WITHOUT_XNNPACK' --copt='-DACC_NEON'"
board_user = "xilinx"
board_hostname = "jharis.ddns.net"
arm_dir = "/home/xilinx/Workspace/secda_benchmark_suite"
path_to_tf = "/home/jude/Workspace/SECDA-TFLite/tensorflow"
def generate_generation_script(output_path):
    script = "#!/bin/bash\n"
    script += "set -e\n"
    script += f"arm_dir={arm_dir}/bins\n"
    script += f"pushd {path_to_tf}\n"
    for delegate, ver in supported_delegates.items():

        for tool, sn in supported_tools.items():
            del_path = f"tensorflow/lite/delegates/utils/secda_delegates/{delegate}"
            name = f"{sn}_{delegate}_{ver}"
            bin_name = f"{tool}_plus_{delegate}"
            if delegate == "cpu" :
                del_path =  cpu_paths[tool][0]
                bin_name = cpu_paths[tool][1]

            script += f"{bb_pr}{del_path}:{bin_name} {bb_po} \n"
            script += f"rsync -r -avz -e 'ssh -p 2202' {path_to_tf}/bazel-out/armhf-opt/bin/{del_path}/{bin_name} {board_user}@{board_hostname}:{arm_dir}/bins/{name}\n"
    script += f"ssh -t -p 2202 {board_user}@{board_hostname} 'cd {arm_dir}/bins/ && chmod 775 ./*'\n"
    script += "popd\n"
    with open(output_path, "w") as f:
        f.write(script)
    os.system(f"chmod +x {output_path}")

generate_generation_script("generated/gen_bins.sh")