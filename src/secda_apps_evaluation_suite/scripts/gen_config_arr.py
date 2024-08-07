import sys
import os
sys.dont_write_bytecode = True
import os   
from utils import *


def gen_config_arr_imp_bm(sc, params):
    
    models        = params[0]
    threads       = params[1]
    num_run       = params[2]
    hardware      = params[3]
    model_dir     = params[4]
    bitstream_dir = params[5]
    bin_dir       = params[6]
    board_user    = params[7]
    
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
                # config_list.append(hw_config) ## rpp- why??
    # fix this hard
    # print(f"Creating {out_dir}")
    os.makedirs(out_dir, exist_ok=True)

    f = open(f"{out_dir}/configs_bm.sh", "w+")
    # list of all the config properties
    declare_array(f, "hw", hw_list)
    declare_array(f, "model", model_list)
    declare_array(f, "thread", thread_list)
    declare_array(f, "num_run", num_run_list)
    declare_array(f, "version", version_list)
    declare_array(f, "del_version", del_version_list)
    declare_array(f, "del", delegate_list)
    f.close()

    # ## Generate run_collect.sh
    # r_dict = {
    #     "model_dir": model_dir,
    #     "bitstream_dir": bitstream_dir,
    #     "bin_dir": bin_dir,
    #     "board_user": board_user,
    # }

    # with open("scripts/run_collect.tpl.sh") as f:
    #     script = str(mt(f.read()).substitute(r_dict))
    # with open(f"{out_dir}/run_collect.sh", "w+") as f:
    #     f.write(script)
    
def gen_config_arr_imp_indiff(sc, params):
    
    models        = params[0]
    threads       = params[1]
    num_run       = params[2]
    hardware      = params[3]
    model_dir     = params[4]
    bitstream_dir = params[5]
    bin_dir       = params[6]
    board_user    = params[7]
    
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
                # config_list.append(hw_config) ## rpp- why??
    # fix this hard
    # print(f"Creating {out_dir}")
    os.makedirs(out_dir, exist_ok=True)

    f = open(f"{out_dir}/configs_indiff.sh", "w+")
    # list of all the config properties
    declare_array(f, "hw", hw_list)
    declare_array(f, "model", model_list)
    declare_array(f, "thread", thread_list)
    declare_array(f, "num_run", num_run_list)
    declare_array(f, "version", version_list)
    declare_array(f, "del_version", del_version_list)
    declare_array(f, "del", delegate_list)
    f.close()

    # ## Generate run_collect.sh
    # r_dict = {
    #     "model_dir": model_dir,
    #     "bitstream_dir": bitstream_dir,
    #     "bin_dir": bin_dir,
    #     "board_user": board_user,
    # }

    # with open("scripts/run_collect.tpl.sh") as f:
    #     script = str(mt(f.read()).substitute(r_dict))
    # with open(f"{out_dir}/run_collect.sh", "w+") as f:
    #     f.write(script)

def gen_config_arr_imp_em(sc, params):
    
    models        = params[0]
    image_names   = params[1]
    threads       = params[2]
    num_run       = params[3]
    hardware      = params[4]
    model_dir     = params[5]
    bitstream_dir = params[6]
    bin_dir       = params[7]
    board_user    = params[8]
    
    ## Generate configs.sh
    config_list = []
    hw_list = []
    model_list = []
    image_list = []
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
                image_list.append(image_names)
                thread_list.append(thread)
                num_run_list.append(num_run)
                version_list.append(hw_config["version"])
                del_version_list.append(hw_config["del_version"])
                delegate_list.append(hw_config["del"])
                # config_list.append(hw_config) ## rpp- why??
    # fix this hard
    # print(f"Creating {out_dir}")
    os.makedirs(out_dir, exist_ok=True)

    f = open(f"{out_dir}/configs_em.sh", "w+")
    # list of all the config properties
    declare_array(f, "hw", hw_list)
    declare_array(f, "model", model_list)
    declare_array(f, "image", image_list)
    declare_array(f, "thread", thread_list)
    declare_array(f, "num_run", num_run_list)
    declare_array(f, "version", version_list)
    declare_array(f, "del_version", del_version_list)
    declare_array(f, "del", delegate_list)
    f.close()

    # ## Generate run_collect.sh
    # r_dict = {
    #     "model_dir": model_dir,
    #     "bitstream_dir": bitstream_dir,
    #     "bin_dir": bin_dir,
    #     "board_user": board_user,
    # }

    # with open("scripts/run_collect.tpl.sh") as f:
    #     script = str(mt(f.read()).substitute(r_dict))
    # with open(f"{out_dir}/run_collect.sh", "w+") as f:
    #     f.write(script)

def gen_config_arr_imp_ema(sc, params):
    
    models        = params[0]
    threads       = params[1]
    image_no      = params[2]
    num_run       = params[3]
    hardware      = params[4]
    model_dir     = params[5]
    bitstream_dir = params[6]
    bin_dir       = params[7]
    board_user    = params[8]
    
    ## Generate configs.sh
    config_list = []
    hw_list = []
    model_list = []
    imageNo_list = []
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
                imageNo_list.append(image_no)
                thread_list.append(thread)
                num_run_list.append(num_run)
                version_list.append(hw_config["version"])
                del_version_list.append(hw_config["del_version"])
                delegate_list.append(hw_config["del"])
                # config_list.append(hw_config) ## rpp- why??
    # fix this hard
    # print(f"Creating {out_dir}")
    os.makedirs(out_dir, exist_ok=True)

    f = open(f"{out_dir}/configs_ema.sh", "w+")
    # list of all the config properties
    declare_array(f, "hw", hw_list)
    declare_array(f, "model", model_list)
    declare_array(f, "thread", thread_list)
    declare_array(f, "imageNo", imageNo_list)
    declare_array(f, "num_run", num_run_list)
    declare_array(f, "version", version_list)
    declare_array(f, "del_version", del_version_list)
    declare_array(f, "del", delegate_list)
    f.close()

    # ## Generate run_collect.sh
    # r_dict = {
    #     "model_dir": model_dir,
    #     "bitstream_dir": bitstream_dir,
    #     "bin_dir": bin_dir,
    #     "board_user": board_user,
    # }

    # with open("scripts/run_collect.tpl.sh") as f:
    #     script = str(mt(f.read()).substitute(r_dict))
    # with open(f"{out_dir}/run_collect.sh", "w+") as f:
    #     f.write(script)

def gen_config_arr_imp_iic(sc, params):
    
    models        = params[0]
    threads       = params[1]
    image_no      = params[2]
    num_run       = params[3]
    hardware      = params[4]
    model_dir     = params[5]
    bitstream_dir = params[6]
    bin_dir       = params[7]
    board_user    = params[8]
    
    ## Generate configs.sh
    config_list = []
    hw_list = []
    model_list = []
    imageNo_list = []
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
                imageNo_list.append(image_no)
                thread_list.append(thread)
                num_run_list.append(num_run)
                version_list.append(hw_config["version"])
                del_version_list.append(hw_config["del_version"])
                delegate_list.append(hw_config["del"])
                # config_list.append(hw_config) ## rpp- why??
    # fix this hard
    # print(f"Creating {out_dir}")
    os.makedirs(out_dir, exist_ok=True)

    f = open(f"{out_dir}/configs_iic.sh", "w+")
    # list of all the config properties
    declare_array(f, "hw", hw_list)
    declare_array(f, "model", model_list)
    declare_array(f, "thread", thread_list)
    declare_array(f, "imageNo", imageNo_list)
    declare_array(f, "num_run", num_run_list)
    declare_array(f, "version", version_list)
    declare_array(f, "del_version", del_version_list)
    declare_array(f, "del", delegate_list)
    f.close()

    # ## Generate run_collect.sh
    # r_dict = {
    #     "model_dir": model_dir,
    #     "bitstream_dir": bitstream_dir,
    #     "bin_dir": bin_dir,
    #     "board_user": board_user,
    # }

    # with open("scripts/run_collect.tpl.sh") as f:
    #     script = str(mt(f.read()).substitute(r_dict))
    # with open(f"{out_dir}/run_collect.sh", "w+") as f:
    #     f.write(script)