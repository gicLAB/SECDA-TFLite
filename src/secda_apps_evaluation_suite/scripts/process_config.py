import argparse
import shutil
import subprocess
from string import Template
from gen_bins import gen_bins
from itertools import product
from utils import load_config, find_hw_config, declare_array, mt
import os
import json
import sys
sys.dont_write_bytecode = True


cpu_paths = {
    "benchmark_model": ["tensorflow/lite/tools/benchmark", "benchmark_model"],
    "inference_diff": [
        "tensorflow/lite/tools/evaluation/tasks/inference_diff",
        "run_eval",
    ],
    "eval_model": ["tensorflow/lite/examples/secda_apps/eval_model", "eval_model"],
    "eval_model_accuracy": ["tensorflow/lite/examples/secda_apps/eval_model_accuracy", "eval_model_accuracy"],
    "imagenet_image_classification": ["tensorflow/lite/examples/secda_apps/imagenet_image_classification", "run_eval"],
}

model_alts = ["model", "tflite_model", "graph", "tflite_graph", "model_file"]

sc = load_config("../../config.json")  # system config
board_user = sc["board_user"]
data_dir_host = sc["data_dir"]
board_dir = sc["board_dir"]


def copyBitstreamFilesToBoard(system_config, hw_arr, board_eval_dir):
    secda_tflite_path = system_config["secda_tflite_path"]
    gen_hw_path = secda_tflite_path + "/hardware_automation/generated"
    # check this folder exists
    if not os.path.exists(gen_hw_path):
        print(f"Hardware automation path {gen_hw_path} does not exist.")
        print("Please run the hardware automation first !!!")
        sys.exit(1)
    for hw in hw_arr:
        # check the number "v" in the hw, if more than one create an error
        # flag
        if hw.count("v") > 1:
            print(
                f"Error: More than one 'v' char found in {hw} name: Make it one!!")
            sys.exit(1)
        hw = hw.replace("v", "_")
        hw_path = f"{gen_hw_path}/{hw}"
        bitStreamPath = f"{hw_path}/generated_files/{hw}.bit"
        hwhPath = f"{hw_path}/generated_files/{hw}.hwh"

        if not os.path.exists(bitStreamPath) or not os.path.exists(hwhPath):
            print(
                f"!!!!!!!!!!!!!!!!Bitstream file {bitStreamPath} or HWH file {hwhPath} does not exist.!!!!!!!!!!!!")
            continue

        # copy bitStream and hwh to board_eval_director/bitstreams
        # print("Bitstream Path:", bitStreamPath)
        # print("HWH Path:", hwhPath)
        # Copy bitstream file to board
        local_rsync_cmd = f"rsync -r -avz {bitStreamPath} {system_config['board_user']}@{system_config['board_hostname']}:{board_eval_dir}/bitstreams/"
        subprocess.run(local_rsync_cmd, shell=True)
        # Copy hwh file to board
        local_rsync_cmd = f"rsync -r -avz {hwhPath} {system_config['board_user']}@{system_config['board_hostname']}:{board_eval_dir}/bitstreams/"
        subprocess.run(local_rsync_cmd, shell=True)


def checkDataPathInBoard(ae_config, system_config):
    if "paths" in ae_config and "d" in ae_config["paths"]:
        data_path = ae_config["paths"]["d"]
        print("Data Path:", data_path)
    else:
        print("Data Path not found in AEC:", ae_config)
        sys.exit(1)
    # from the system_config load the board_user
    board_user = system_config["board_user"]
    board_hostname = system_config["board_hostname"]
    board_port = system_config["board_port"]
    # Implement the logic to check if the data path exists in the board
    ssh_command = f"ssh -o LogLevel=QUIET -p {board_port} {board_user}@{board_hostname} '[ -d {data_path} ] && echo \"exists\" || echo \"not found\"'"
    result = subprocess.run(
        ssh_command, shell=True, capture_output=True, text=True)
    if result.stdout.strip() == "exists":
        print("Data Path Provided in the AEC exists in the Board")
    else:
        print("Data Path provided in the AEC does not exist in the Board")
        sys.exit(1)


def replace_path(aec):
    paths = aec["paths"]
    for key in paths:
        for app in aec["apps"]:
            flags = aec["apps"][app].keys()
            for f in flags:
                fv = aec["apps"][app][f]
                if type(fv) == str:
                    aec["apps"][app][f] = fv.replace(f"$({key})", paths[key])
                elif type(fv) == list:
                    for i in range(len(fv)):
                        fv[i] = fv[i].replace(f"$({key})", paths[key])


def get_models_from_dir(model_dir):
    models = []
    for model in os.listdir(model_dir):
        if model.endswith(".tflite"):
            models.append(model_dir + model.replace(".tflite", ""))
    return models


def get_hw_info(
    sc,
    hardware,
):
    # Generate configs.sh
    config_list = []
    hw_list = []
    version_list = []
    del_version_list = []
    delegate_list = []
    for hw in hardware:
        hw_config_file = find_hw_config(
            f"{sc['secda_tflite_path']}/{sc['hw_configs']}", hw)
        hw_config = load_config(hw_config_file)
        hw_list.append(hw_config["acc_name"])
        version_list.append(hw_config["version"])
        del_version_list.append(hw_config["del_version"])
        delegate_list.append(hw_config["del"])
        config_list.append(hw_config)


def create_run_config(sc, aec, app_dict, board_eval_dir, load_bitstreams):
    hw_list = []
    app_list = []
    model_list = []
    cmd_list = []
    del_version_list = []
    delegate_list = []
    version_list = []
    taglist = []
    out_dir = f"./{sc['out_dir']}"
    for hw in aec["hardware"]:
        hw_config_file = find_hw_config(
            f"{sc['secda_tflite_path']}/{sc['hw_configs']}", hw)
        hw_config = load_config(hw_config_file)
        delegate = hw_config["del"]
        for app in aec["apps"]:
            # print(f"Generating config array for {app} on {hw}")
            flags = aec["apps"][app].keys()
            config_arr = {}

            for f in flags:
                fv = aec["apps"][app][f]
                if f in model_alts and type(fv) == str and fv.endswith("/"):
                    models = get_models_from_dir(aec["apps"][app][f])
                    # add data_dir_host to models
                    models = [model + ".tflite" for model in models]
                    config_arr[f] = models
                    continue
                elif f in model_alts and type(fv) == list:
                    models = []
                    for model in fv:
                        if model.endswith("/"):
                            models += get_models_from_dir(model)
                        else:
                            models.append(model)
                    # add data_dir_host to models
                    models = [model + ".tflite" for model in models]
                    config_arr[f] = models
                    continue

                # check if flag is a list
                if type(aec["apps"][app][f]) == list:
                    config_arr[f] = aec["apps"][app][f]
                else:
                    config_arr[f] = [aec["apps"][app][f]]
            all_configs = list(product(*config_arr.values()))
            # print(f"Total number of configurations: {len(all_configs)}")
            usedel = f"--use_{delegate}=true"
            if hw == "CPU" or hw == "CPU_KRIAv1_0":
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
                version_list.append(hw_config["version"])
                del_version_list.append(hw_config["del_version"])
                delegate_list.append(hw_config["del"])
                app_list.append(app)
                cmd_list.append(cmd)
                taglist.append(tag)

        os.makedirs(out_dir, exist_ok=True)
        f = open(f"{out_dir}/configs.sh", "w+")
        # list of all the config properties
        declare_array(f, "hw", hw_list)
        declare_array(f, "tag", taglist)
        declare_array(f, "app", app_list)
        declare_array(f, "model", model_list)
        declare_array(f, "cmd", cmd_list)
        declare_array(f, "del", delegate_list)
        declare_array(f, "del_version", del_version_list)
        declare_array(f, "version", version_list)

        f.close()

    # Generate run_collect.sh
    r_dict = {
        "board_dir": board_eval_dir,
        "board_user": board_user,
        "ld_bst": load_bitstreams,
    }
    # print("r_dict:", r_dict)

    with open("scripts/run_collect.tpl.sh") as f:
        template = Template(f.read())
        script = template.safe_substitute(r_dict)
    with open(f"{out_dir}/run_collect.sh", "w") as f:
        f.write(script)


def create_bin_config(aec):
    apps = aec["apps"].keys()
    app_dict = {}
    for app in apps:
        sn = ""
        for s in app.split("_"):
            sn += s[0]
        app_dict[app] = sn
    return app_dict


def main(raw_args=None):
    parser = argparse.ArgumentParser(description="Capture Experiment Video")
    parser.add_argument("config_file", type=str, help="config file")
    parser.add_argument("gen_bin", type=int,
                        help="whether to generate binaries")
    parser.add_argument("board_eval_dir", type=str,
                        help="board evaluation directory")
    parser.add_argument("copy_bitstreams", type=int,
                        help="whether to copy bitstreams")
    parser.add_argument("load_bitstreams", type=int,
                        help="whether to load bitstreams")
    args = parser.parse_args(raw_args)

    gen_bin = args.gen_bin
    aec_file = args.config_file
    board_eval_dir = args.board_eval_dir
    copy_bitstreams = args.copy_bitstreams
    load_bitstreams = args.load_bitstreams

    aec = load_config(aec_file)  # app evaluation config

    print("Gen Bins:", gen_bin)
    print("Loaded AEC:", aec_file)
    print("Board Evaluation Directory:", board_eval_dir)
    print("Copy BitStreams:", copy_bitstreams)
    print("Load BitStreams:", load_bitstreams)
    # print("AEC Contents before replacing:", json.dumps(aec, indent=2))

    # remove sc['out_dir'] if exists
    out_dir = f"{sc['out_dir']}"
    # print("Output Directory:", out_dir)
    if os.path.exists(out_dir):
        shutil.rmtree(out_dir)
        # create
        os.makedirs(out_dir, exist_ok=True)

    # check board_data_path provided in the AEC as key name "paths"
    # do exist in the board or not
    checkDataPathInBoard(ae_config=aec, system_config=sc)
    # if board_data_path exists then rebuild the model paths in AEC
    replace_path(aec)
    # print("AEC Contents after Replacing:", json.dumps(aec, indent=2))

    app_dict = create_bin_config(aec)
    hw = aec["hardware"]

    # copy the bit-stream file
    if copy_bitstreams > 0:
        print("!!!!!!!!!Copying bit-stream files to board...!!!!!!!!!")
        copyBitstreamFilesToBoard(
            system_config=sc, hw_arr=hw, board_eval_dir=board_eval_dir)

    # sys.exit(0)
    if gen_bin > 0:
        # print("Generating scripts create binaries...")
        gen_bins(sc, hw, app_dict, board_eval_dir)
    create_run_config(sc, aec, app_dict, board_eval_dir, load_bitstreams)
    # sys.exit(0)


if __name__ == "__main__":
    main()
