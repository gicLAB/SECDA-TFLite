import argparse
import shutil
from string import Template
from itertools import product
# from utils import utils.load_config, utils.find_hw_config, utils.declare_array,utils.find_the_boards
import utils
import os
import json
import sys
sys.dont_write_bytecode = True
import subprocess

model_alts = ["model", "tflite_model", "graph", "tflite_graph", "model_file"]

def log_out(string):
    log.write(string + "\n")
    print(string)

def log_subprocess_run(cmd, **kwargs):
    """Log subprocess.run command and output to log file"""
    log_out(f"[CMD] {cmd}")
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True, **kwargs)
    if result.stdout:
        log_out(f"[STDOUT] {result.stdout}")
    if result.stderr:
        log_out(f"[STDERR] {result.stderr}")
    return result


def main(raw_args=None):
    global log
    
    parser = argparse.ArgumentParser(description="Process Flags and Apps Configurations")
    parser.add_argument("sc_config_file", type=str, help="System config file")
    parser.add_argument("aec_config_file", type=str, help="AEC config file")
    parser.add_argument("init", type=int, help="whether to initialize the board")
    parser.add_argument("gen_bin", type=int,
                        help="whether to generate binaries")
    parser.add_argument("copy_bitstreams", type=int,
                        help="whether to copy bitstreams")
    parser.add_argument("load_bitstreams", type=int,
                        help="whether to load bitstreams")
    parser.add_argument("collect_power", type=int,
                        help="whether to collect power during running the exps")
    args = parser.parse_args(raw_args)

    sc_file = args.sc_config_file
    aec_file = args.aec_config_file
    init = args.init
    gen_bin = args.gen_bin
    copy_bitstreams = args.copy_bitstreams
    load_bitstreams = args.load_bitstreams
    collect_power = args.collect_power

    sc = utils.load_config(sc_file)  # system config
    aec = utils.load_config(aec_file)  # app evaluation config

    # Create output directory and log file
    out_dir = f"{sc['out_dir']}"
    os.makedirs(out_dir, exist_ok=True)
    if os.path.exists(out_dir):
        shutil.rmtree(out_dir)
        # create
        os.makedirs(out_dir, exist_ok=True)
        
    log = open(f"{out_dir}/process_flags_n_config.log", "w")

    log_out("-----------------------------------------------------------")
    log_out("Process Flags and Apps Configurations")
    log_out("-----------------------------------------------------------")
    
    log_out(f"  Loaded SC: {sc_file}")
    log_out(f"  Loaded AEC: {aec_file}")
    log_out(f"  Init: {init}")
    log_out(f"  Bin Gen: {gen_bin}")
    log_out(f"  Copy BitStreams: {copy_bitstreams}")
    log_out(f"  Load BitStreams: {load_bitstreams}")
    log_out(f"  Collect Power: {collect_power}")

    # remove sc['out_dir'] if exists

    
    # # Re-open log file after removing directory
    # log = open(f"{out_dir}/process_flags_n_config.log", "w")

    app_dict = create_app_dict(aec)
    board_list = utils.find_the_boards(f"{sc['secda_tflite_path']}/{sc['hw_configs']}", aec["hardware"])
    create_run_config(sc, aec, app_dict, load_bitstreams)
    
    if init > 0:
        log_out("!!!!!!!!!Initializing SECDA Apps Evaluation Suite...!!!!!!!!!")
        secda_app_eval_suite_init(sc, aec)
    
    if gen_bin > 0:
        log_out("!!!!!!!!!Generating scripts to create binaries...!!!!!!!!!")
        generate_bazel_build_scripts(sc, aec["hardware"], app_dict)
        log_out("!!!!!!!!!Generating binaries and Transferring...!!!!!!!!!")
        for br in board_list:
            log_subprocess_run(
                f"./generated/{br}/gen_bins.sh",
                check=False,
            )
    # copy the bit-stream file
    if copy_bitstreams > 0:
        log_out("!!!!!!!!!Copying bit-stream files to board...!!!!!!!!!")
        copyBitstreamFilesToBoard(
            system_config=sc, hw_arr=aec["hardware"])

    log_out("-----------------------------------------------------------")
    log_out("Transferring Configurations and Experiments to Target Board")
    log_out("-----------------------------------------------------------")
    transfer_exp_configs (sc, board_list)
    
    log_out("-----------------------------------------------------------")
    log_out("Running Experiments")
    log_out("-----------------------------------------------------------")
    run_exps (sc, board_list, collect_power)
    
    log_out("-----------------------------------------------------------")
    log_out("Transferring Results to Host")
    log_out("-----------------------------------------------------------")
    transfer_results_to_host(sc, board_list)
    
    
    log.close()

def transfer_results_to_host(sc, board_list):
    out_dir = f"{sc['out_dir']}"
    exp_output_dir =f"{out_dir}/exp_output"
    os.makedirs(out_dir, exist_ok=True)
    
    for board_name in board_list:
        board_dir = sc["boards"][board_name]["board_dir"]
        board_eval_dir = board_dir+"/secda_apps_evaluation_suite"
        board_user = sc["boards"][board_name]["board_user"]
        board_hostname = sc["boards"][board_name]["board_hostname"]
        board_port = sc["boards"][board_name]["board_port"]
        board_data_dir = board_dir.replace("/secda_tflite", "/data")
        
        log_subprocess_run(
            f"rsync -q -av -e 'ssh -p '{board_port} {board_user}@{board_hostname}:{board_eval_dir}/tmp {exp_output_dir}", 
            check = False
        )


def run_exps (sc, board_list, collect_power):
    for board_name in board_list:
        board_dir = sc["boards"][board_name]["board_dir"]
        board_eval_dir = board_dir+"/secda_apps_evaluation_suite"
        board_user = sc["boards"][board_name]["board_user"]
        board_hostname = sc["boards"][board_name]["board_hostname"]
        board_port = sc["boards"][board_name]["board_port"]
        board_data_dir = board_dir.replace("/secda_tflite", "/data")
        
        if board_name == "KRIA":
            log_subprocess_run(
                f"ssh -o LogLevel=QUIET -t -p {board_port} {board_user}@{board_hostname} 'cd {board_eval_dir}/ && ./run_collect.sh {collect_power}'", 
                check = False
            )
        else:
            log_subprocess_run(
                f"ssh -o LogLevel=QUIET -t -p {board_port} {board_user}@{board_hostname} 'cd {board_eval_dir}/ && ./run_collect.sh'", 
                check = False
            )

def transfer_exp_configs (sc, board_list):
    for board_name in board_list:
        board_dir = sc["boards"][board_name]["board_dir"]
        board_eval_dir = board_dir+"/secda_apps_evaluation_suite"
        board_user = sc["boards"][board_name]["board_user"]
        board_hostname = sc["boards"][board_name]["board_hostname"]
        board_port = sc["boards"][board_name]["board_port"]
        board_data_dir = board_dir.replace("/secda_tflite", "/data")
        
        log_subprocess_run(
            f"rsync -q -r -avz -e 'ssh -p {board_port}' ./generated/{board_name}/configs.sh {board_user}@{board_hostname}:{board_eval_dir}/", 
            check = False
        )
        log_subprocess_run(
            f"rsync -q -r -avz -e 'ssh -p {board_port}' ./generated/{board_name}/run_collect.sh {board_user}@{board_hostname}:{board_eval_dir}/", 
            check = False
        )
        log_subprocess_run(
            f"ssh -o LogLevel=QUIET -t -p {board_port} {board_user}@{board_hostname} 'cd {board_eval_dir}/ && chmod +x ./*.sh'", 
            check = False
        )
                        
def generate_bazel_build_scripts(sc, hw_arr, app_dict):
    cpu_paths = {
        "benchmark_model": ["tensorflow/lite/tools/benchmark", "benchmark_model"],
        "inference_diff": [
            "tensorflow/lite/tools/evaluation/tasks/inference_diff",
            "run_eval",
        ],
        "eval_model": ["tensorflow/lite/examples/secda_apps/eval_model", "eval_model"],
        "eval_model_accuracy": [
            "tensorflow/lite/examples/secda_apps/eval_model_accuracy",
            "eval_model_accuracy",
        ],
        "imagenet_image_classification": [
            "tensorflow/lite/examples/secda_apps/imagenet_image_classification",
            "run_eval",
        ],
    }

    bb_pr_pynq = "bazel6 build --config=elinux_armhf -c opt //"
    bb_po_pynq = "--copt='-DSECDA_LOGGING_DISABLED' --cxxopt='-march=armv7-a' --cxxopt='-mfpu=neon' --cxxopt='-funsafe-math-optimizations' --cxxopt='-ftree-vectorize' --copt='-DACC_PROFILE' --define tflite_with_xnnpack=false --copt='-DTFLITE_ENABLE_XNNPACK=OFF' --copt='-DTFLITE_WITHOUT_XNNPACK' --copt='-DACC_NEON' --@secda_tools//:config=fpga"
    bb_pr_kria = "bazel6 build --config=elinux_aarch64 -c opt //"
    bb_po_kria = "--copt='-DSECDA_LOGGING_DISABLED' --copt='-DACC_PROFILE' --define tflite_with_xnnpack=false --copt='-DTFLITE_ENABLE_XNNPACK=OFF' --copt='-DTFLITE_WITHOUT_XNNPACK' --copt='-DACC_NEON' --copt='-DKRIA' --@secda_tools//:config=fpga_arm64"

    board_list = utils.find_the_boards(f"{sc['secda_tflite_path']}/{sc['hw_configs']}", hw_arr)

    for board_name in board_list:
        output_path = f"{sc['out_dir']}/{board_name}/gen_bins.sh"
        board_dir = sc["boards"][board_name]["board_dir"]
        board_eval_dir = board_dir+"/secda_apps_evaluation_suite"
        board_user = sc["boards"][board_name]["board_user"]
        board_hostname = sc["boards"][board_name]["board_hostname"]
        board_port = sc["boards"][board_name]["board_port"]

        path_to_tf = sc["secda_tflite_path"] + "/tensorflow"
        rdel_path = sc["path_to_dels"]

        delegates_needed = {}
        for hw in hw_arr:
            hw_config_file = utils.find_hw_config(
                f"{sc['secda_tflite_path']}/{sc['hw_configs']}", hw
            )
            hw_config = utils.load_config(hw_config_file)
            if hw_config["board"]!=board_name:
                continue
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
                for tool, sn in app_dict.items():
                    del_path = f"{rdel_path}/{delegate}/v{ver}"
                    if not os.path.exists(sc["secda_tflite_path"] + "/" + del_path):
                        del_path = f"{rdel_path}/{delegate}"
                    del_path = del_path[del_path.index("/") + 1:]

                    name = f"{sn}_{delegate}_{ver}"
                    bin_name = f"{tool}_plus_{delegate}"

                    if delegate == "cpu":
                        del_path = cpu_paths[tool][0]
                        bin_name = cpu_paths[tool][1]

                    if board_name == "KRIA":
                        script += f"{bb_pr_kria}{del_path}:{bin_name} {bb_po_kria} \n"
                        script += f"rsync -r -avz -e 'ssh -p {board_port}' {path_to_tf}/bazel-out/aarch64-opt/bin/{del_path}/{bin_name} {board_user}@{board_hostname}:{board_eval_dir}/bins/{name}\n"
                    elif board_name == "Z1" or board_name == "Z2":
                        script += f"{bb_pr_pynq}{del_path}:{bin_name} {bb_po_pynq}  \n"
                        script += f"rsync -r -avz -e 'ssh -p {board_port}' {path_to_tf}/bazel-out/armhf-opt/bin/{del_path}/{bin_name} {board_user}@{board_hostname}:{board_eval_dir}/bins/{name}\n"
                    else:
                        raise ValueError("!!! Generating binary scripts: Unsupported Boards. Supported Board Name [Z1, Z2, KRIA]!!!")

        script += f"ssh -t -p {board_port} {board_user}@{board_hostname} 'cd {board_eval_dir}/bins/ && chmod 775 ./*'\n"
        script += "popd\n"
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, "w") as f:
            f.write(script)
        os.system(f"chmod +x {output_path}")

def secda_app_eval_suite_init(sc, aec):
    board_list = utils.find_the_boards(f"{sc['secda_tflite_path']}/{sc['hw_configs']}", aec["hardware"])
    for br in board_list:
        board_name = br
        board_dir = sc["boards"][board_name]["board_dir"]
        board_eval_dir = board_dir+"/secda_apps_evaluation_suite"
        board_user = sc["boards"][board_name]["board_user"]
        board_hostname = sc["boards"][board_name]["board_hostname"]
        board_port = sc["boards"][board_name]["board_port"]
        board_data_dir = board_dir.replace("/secda_tflite", "/data")
        
        # Create directories on board
        ssh_mkdir_cmd = f"ssh -o LogLevel=QUIET -t -p {board_port} {board_user}@{board_hostname} \"mkdir -p {board_eval_dir} && mkdir -p {board_eval_dir}/scripts && mkdir -p {board_eval_dir}/bitstreams && mkdir -p {board_eval_dir}/bins\""
        log_out(f"Creating directories on board {board_name}...")
        log_subprocess_run(ssh_mkdir_cmd, check=False)
        
        # Sync fpga_scripts to board
        rsync_cmd = f"rsync -q -r -avz -e 'ssh -p {board_port}' ./scripts/fpga_scripts/ {board_user}@{board_hostname}:{board_eval_dir}/scripts/"
        log_out(f"Syncing fpga_scripts to board {board_name}...")
        log_subprocess_run(rsync_cmd, check=False)

        # chmod the scripts
        log_subprocess_run(
            f"ssh -o LogLevel=QUIET -t -p {board_port} {board_user}@{board_hostname} 'cd {board_eval_dir}/scripts/ && chmod +x ./*.sh'", 
            check = False
        )
        
        # Transferring Data from Host to the Boards
        for host_data_dir in sc["models_dirs"]:
            subdirs = [d for d in os.listdir(host_data_dir) if os.path.isdir(os.path.join(host_data_dir, d))]
            
            if not subdirs:
                log_out(f"No subdirectories found in {host_data_dir}")
                continue
            
            log_out(f"\nSelect subdirectories to sync from {host_data_dir} (separate numbers by space, or type 'all' for all):")
            for i, subdir in enumerate(subdirs, 1):
                log_out(f"{i}) {subdir}")
            
            selection = input("Enter selection: ")
            
            if selection.lower() == "all":
                selected_subdirs = subdirs
            else:
                selected_subdirs = []
                for idx_str in selection.split():
                    try:
                        idx = int(idx_str)
                        if 1 <= idx <= len(subdirs):
                            selected_subdirs.append(subdirs[idx - 1])
                    except ValueError:
                        pass
            
            for subdir in selected_subdirs:
                subdir_path = os.path.join(host_data_dir, subdir)
                log_out(f"Syncing {subdir} ...")
                rsync_cmd = f"rsync -r -avz -e 'ssh -p {board_port}' {subdir_path} {board_user}@{board_hostname}:{board_data_dir}/"
                log_subprocess_run(rsync_cmd, check=False)
        
def copyBitstreamFilesToBoard(system_config, hw_arr):
    """
    hw_arr=[] list of arrays from the aec
    """
    
    secda_tflite_path = system_config["secda_tflite_path"]
    gen_hw_path = secda_tflite_path + "/hardware_automation/generated"
    hwc_folder_path =f"{system_config['secda_tflite_path']}/{system_config['hw_configs']}"
    
    for hw in hw_arr:
        hwc_path = utils.find_hw_config(hwc_folder_path, hw)
        hwc = utils.load_config(hwc_path)
        acc_name = hwc["acc_name"]+"_"+str(hwc["acc_version"])+"_"+str(hwc["acc_sub_version"]) 
        if not os.path.exists(f"{gen_hw_path}/{acc_name}"):
            log_out(f"Hardware automation path {gen_hw_path}/{acc_name} does not exist.")
            log_out("Please run the hardware automation first !!!")
            continue
        hw_path = f"{gen_hw_path}/{acc_name}"
        bitStreamPath = f"{hw_path}/generated_files/{acc_name}.bit"
        hwhPath = f"{hw_path}/generated_files/{acc_name}.hwh"
        if not os.path.exists(bitStreamPath) or not os.path.exists(hwhPath):
            log_out(f"!!!!!!!!!!!!!!!!Bitstream file or HWH file for {acc_name} does not exist in dir: {hw_path}/generated_files/!!!!!!!!!!!!")
            continue
        board_name = hwc["board"]
        board_dir = system_config["boards"][board_name]["board_dir"]
        board_eval_dir = board_dir+"/secda_apps_evaluation_suite"
        board_user = system_config["boards"][board_name]["board_user"]
        board_hostname = system_config["boards"][board_name]["board_hostname"]
        board_port = system_config["boards"][board_name]["board_port"]
        
        local_rsync_cmd = f"rsync -r -avz -e 'ssh -p {board_port}' {bitStreamPath} {board_user}@{board_hostname}:{board_eval_dir}/bitstreams/"
        log_out(f"Syncing bitstream {acc_name}.bit ...")
        log_subprocess_run(local_rsync_cmd, check=False)
        
        local_rsync_cmd = f"rsync -r -avz -e 'ssh -p {board_port}' {hwhPath} {board_user}@{board_hostname}:{board_eval_dir}/bitstreams/"
        log_out(f"Syncing hwh {acc_name}.hwh ...")
        log_subprocess_run(local_rsync_cmd, check=False)
        
def get_models_from_dir(board_data_dir, host_data_dirs, aec_model_dir):
    """
    - Check aec_model_dir exists in the host_data_dirs
    - retrun the model path by adding board_data_dir by replacing "$(d)"
    - if multiple host_data_dirs have the same aec_model_dir, we will chose the first one
    """
    models = []
    for m in host_data_dirs:
        if '$(d)' in aec_model_dir:
            host_model_path = aec_model_dir.replace("$(d)", m)
        if not os.path.exists(host_model_path):
            continue
        aec_model_dir = aec_model_dir.replace("$(d)", board_data_dir)
        for model in os.listdir(host_model_path):
            if model.endswith(".tflite"):
                models.append(aec_model_dir + model.replace(".tflite", ""))
        return models

def create_run_config(sc, aec, app_dict, load_bitstreams):
    hw_list = []
    app_list = []
    model_list = []
    cmd_list = []
    del_version_list = []
    delegate_list = []
    version_list = []
    taglist = []
    out_dir = f"./{sc['out_dir']}"
    board_list = utils.find_the_boards(f"{sc['secda_tflite_path']}/{sc['hw_configs']}", aec["hardware"])
    
    
    for br in board_list:
        for hw in aec["hardware"]:
            hw_config_file = utils.find_hw_config(
                f"{sc['secda_tflite_path']}/{sc['hw_configs']}", hw)
            hw_config = utils.load_config(hw_config_file)
            board_name = hw_config["board"]
            if board_name != br:
                continue
            board_dir = sc["boards"][board_name]["board_dir"]
            board_eval_dir = board_dir+"/secda_apps_evaluation_suite"
            board_data_dir = board_dir.replace("/secda_tflite", "/data")
            board_user = sc["boards"][board_name]["board_user"]
            host_data_dirs = sc["models_dirs"]
            delegate = hw_config["del"]
            for app in aec["apps"]:
                flags = aec["apps"][app].keys()
                config_arr = {}
                for f in flags:
                    fv = aec["apps"][app][f]
                    if f in model_alts and type(fv) == str and fv.endswith("/"):
                        models = get_models_from_dir(board_data_dir,host_data_dirs,aec["apps"][app][f])
                        models = [model + ".tflite" for model in models]
                        config_arr[f] = models
                        continue
                    elif f in model_alts and type(fv) == list:
                        models = []
                        for model in fv:
                            if model.endswith("/"):
                                models += get_models_from_dir(board_data_dir,host_data_dirs,model)
                            else:
                                models.append(model.replace("$(d)", board_data_dir))
                        models = [model + ".tflite" for model in models]
                        config_arr[f] = models
                        continue

                    if type(aec["apps"][app][f]) == list:
                        config_arr[f] = aec["apps"][app][f]
                    else:
                        config_arr[f] = [aec["apps"][app][f]]
                all_configs = list(product(*config_arr.values()))
                usedel = f"--use_{delegate}=true"
                if "CPU" in hw:
                    usedel = ""
                for config in all_configs:
                    flags_str = ""
                    tag = f"{app}_{hw_config['acc_name']}_{hw_config['del']}_{hw_config['del_version']}"
                    for i, f in enumerate(config_arr.keys()):
                        flags_str += f" --{f}={config[i]}"
                        s = config[i].split("/")[-1]
                        tag += f"_{s}"
                        if config[i].endswith(".tflite"):
                            model_list.append(s)
                    app_call = f"{board_eval_dir}/bins/{app_dict[app]}_{hw_config['del']}_{hw_config['del_version']}"
                    cmd = f"{app_call} {flags_str} {usedel}"
                    hw_list.append(hw_config["acc_name"])
                    version_list.append(str(hw_config["acc_version"])+'_'+str(hw_config["acc_sub_version"]))
                    del_version_list.append(hw_config["del_version"])
                    delegate_list.append(hw_config["del"])
                    app_list.append(app)
                    cmd_list.append(cmd)
                    taglist.append(tag)

            
            os.makedirs(f"{out_dir}/{br}", exist_ok=True)
            f = open(f"{out_dir}/{br}/configs.sh", "w+")
            utils.declare_array(f, "hw", hw_list)
            utils.declare_array(f, "tag", taglist)
            utils.declare_array(f, "app", app_list)
            utils.declare_array(f, "model", model_list)
            utils.declare_array(f, "cmd", cmd_list)
            utils.declare_array(f, "del", delegate_list)
            utils.declare_array(f, "del_version", del_version_list)
            utils.declare_array(f, "version", version_list)

            f.close()
        
        r_dict = {
            "board_dir": board_eval_dir,
            "board_user": board_user,
            "ld_bst": load_bitstreams,
        }

        with open("scripts/run_collect.tpl.sh") as f:
            template = Template(f.read())
            script = template.safe_substitute(r_dict)
        with open(f"{out_dir}/{br}/run_collect.sh", "w") as f:
            f.write(script)

def create_app_dict(aec):
    apps = aec["apps"].keys()
    app_dict = {}
    for app in apps:
        sn = ""
        for s in app.split("_"):
            sn += s[0]
        app_dict[app] = sn
    return app_dict

if __name__ == "__main__":
    main()