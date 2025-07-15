import sys
sys.dont_write_bytecode = True
import json
import os

from utils import load_config, find_hw_config, declare_array, mt
from itertools import product
from gen_bins import gen_bins
import argparse

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

model_alts = ["model", "tflite_model" , "graph", "tflite_graph", "model_file"]

sc = load_config("../../config.json") ## system config
board_user = sc["board_user"]
data_dir_host = sc["data_dir"]
board_dir = sc["board_dir"]

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
    ## Generate configs.sh
    config_list = []
    hw_list = []
    version_list = []
    del_version_list = []
    delegate_list = []
    for hw in hardware:
        hw_config_file= find_hw_config(f"{sc['secda_tflite_path']}/{sc['hw_configs']}", hw)
        hw_config = load_config(hw_config_file)
        hw_list.append(hw_config["acc_name"])
        version_list.append(hw_config["version"])
        del_version_list.append(hw_config["del_version"])
        delegate_list.append(hw_config["del"])
        config_list.append(hw_config)



def create_run_config(sc,aec ,app_dict):
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
        hw_config_file= find_hw_config(f"{sc['secda_tflite_path']}/{sc['hw_configs']}", hw)
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
            usedel=f"--use_{delegate}=true"
            if hw == "CPU":
                usedel=""
            for config in all_configs:
                flags_str = ""
                tag = f"{app}_{hw_config['acc_name']}_{hw_config['del']}_{hw_config['del_version']}"
                for i, f in enumerate(config_arr.keys()):
                    flags_str += f" --{f}={config[i]}" 
                    s = config[i].split("/")[-1]
                    tag += f"_{s}"
                    if config[i].endswith(".tflite"):
                        model_list.append(s)
                app_call = f"{board_dir}/apps_eval_suite/bins/{app_dict[app]}_{hw_config['del']}_{hw_config['del_version']}"
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

        ## Generate run_collect.sh
        r_dict = {
            "board_dir": board_dir,
            "board_user": board_user,
        }

        with open("scripts/run_collect.tpl.sh") as f:
            script = str(mt(f.read()).substitute(r_dict))
        with open(f"{out_dir}/run_collect.sh", "w+") as f:
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
    parser.add_argument("gen_bin", type=bool, help="whether to generate binaries")
    args = parser.parse_args(raw_args)

    gen_bin = args.gen_bin
    aec_file = args.config_file


    aec = load_config(aec_file) ## app evaluation config


    replace_path(aec)

    app_dict = create_bin_config(aec)
    hw = aec["hardware"]
    if gen_bin:
        gen_bins(sc, hw, app_dict)
    create_run_config(sc,aec,app_dict)


if __name__ == "__main__":
    main()