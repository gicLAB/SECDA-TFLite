import sys
sys.dont_write_bytecode = True
from string import Template
import json


def load_config(hw):
    with open(f"configs/{hw}.json") as f:
        config = json.load(f)
    return config


def declare_array(f, name, list):
    f.write("declare -a {}_array=(\n".format(name))
    for i in list:
        f.write('  "{}" \n'.format(i))
    f.write(")\n")


class mt(Template):
    delimiter = "£"
    idpattern = r"[a-z][_a-z0-9]*"


def gen_benchmark(params):
    gen_benchmark_imp(
        params[0],
        params[1],
        params[2],
        params[3],
        params[4],
        params[5],
        params[6],
        params[7],
    )


def gen_benchmark_imp(
    models, hardware, threads, num_run, model_dir, bitstream_dir, bin_dir, board_user
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
    for hw in hardware:
        config = load_config(hw)
        for model in models:
            for thread in threads:
                hw_list.append(config["hardware"])
                model_list.append(model)
                thread_list.append(thread)
                num_run_list.append(num_run)
                version_list.append(config["version"])
                del_version_list.append(config["del_version"])
                delegate_list.append(config["del"])
                config_list.append(config)

    f = open("generated/configs.sh", "w+")
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
    }
    with open("scripts/run_collect.tpl.sh") as f:
        script = str(mt(f.read()).substitute(r_dict))
    with open("generated/run_collect.sh", "w+") as f:
        f.write(script)
