import sys
sys.dont_write_bytecode = True
import os   
from benchmark_utils import *


def gen_bench(sc, params):
    gen_bench_imp(
        sc,
        params[0],
        params[1],
        params[2],
        params[3],
        params[4],
        params[5],
        params[6],
        params[7],
    )


def gen_bench_imp(
    sc,
    models,
    hardware,
    threads,
    num_run,
    model_dir,
    bitstream_dir,
    bin_dir,
    board_user,
):
    ## Generate configs.sh
    config_list = []
    hw_list = []
    model_list = []
    thread_list = []
    num_run_list = []
    version_list = []
    del_version_list = []
    delegate_list = []
    out_dir = f"./{sc['out_dir']}"
    for hw in hardware:
        # hw_config_file = f"{sc['secda_tflite_path']}/{sc['hw_configs']}/{hw}"
        hw_config_file= find_hw_config(f"{sc['secda_tflite_path']}/{sc['hw_configs']}", hw)
        # print(f"hw_config_file: {hw_config_file}")
        hw_config = load_config(hw_config_file)
        for model in models:
            for thread in threads:
                hw_list.append(hw_config["acc_name"])
                model_list.append(model)
                thread_list.append(thread)
                num_run_list.append(num_run)
                version_list.append(hw_config["version"])
                del_version_list.append(hw_config["del_version"])
                delegate_list.append(hw_config["del"])
                config_list.append(hw_config)
    # fix this hard
    # print(f"Creating {out_dir}")
    os.makedirs(out_dir, exist_ok=True)

    f = open(f"{out_dir}/configs.sh", "w+")
    # list of all the config properties
    declare_array(f, "hw", hw_list)
    declare_array(f, "model", model_list)
    declare_array(f, "thread", thread_list)
    declare_array(f, "num_run", num_run_list)
    declare_array(f, "version", version_list)
    declare_array(f, "del_version", del_version_list)
    declare_array(f, "del", delegate_list)
    f.close()

    ## Generate run_collect.sh
    r_dict = {
        "model_dir": model_dir,
        "bitstream_dir": bitstream_dir,
        "bin_dir": bin_dir,
        "board_user": board_user,
        "board_dir": sc["board_dir"],
    }

    with open("scripts/run_collect.tpl.sh") as f:
        script = str(mt(f.read()).substitute(r_dict))
    with open(f"{out_dir}/run_collect.sh", "w+") as f:
        f.write(script)
