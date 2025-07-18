import sys

sys.dont_write_bytecode = True

import json
from string import Template
import os
import subprocess

# ============================================================
# helper.py
# ============================================================

import math


def find_len_of_needed_outputs_of_outrows(id, oh, ow, pl, pr, pt, ks, sx, sy):
    width_col = (ow + pl + pr - ks) // sy + 1
    cal_col = id % width_col
    cal_row = id // width_col
    h_pad = -pt + (sy * cal_row)
    w_pad = -pl + (sx * cal_col)
    h_high = h_pad + ks
    w_high = w_pad + ks

    h_len = min(h_high, oh) - max(h_pad, 0)
    w_len = min(w_high, ow) - max(w_pad, 0)
    print("h_pad:", h_pad, "h_high:", h_high)
    print("w_pad:", w_pad, "w_high:", w_high)
    print("h_len:", h_len)
    print("w_len:", w_len)
    return h_len * w_len


def col2im(
    depth,
    height,
    width,
    filter_h,
    filter_w,
    pad_t,
    pad_l,
    pad_b,
    pad_r,
    stride_h,
    stride_w,
):
    height_col = (height + pad_t + pad_b - filter_h) // stride_h + 1
    width_col = (width + pad_l + pad_r - filter_w) // stride_w + 1
    h_pad = -pad_t
    im_dex = 0
    map_dex = 0

    wasted_out = 0
    out_map = []
    for h in range(height_col):
        w_pad = -pad_l
        for w in range(width_col):
            im_dex = (h_pad * width + w_pad) * depth
            for ih in range(filter_h):
                for iw in range(filter_w):
                    if (
                        ih + h_pad >= 0
                        and ih + h_pad < height
                        and iw + w_pad >= 0
                        and iw + w_pad < width
                    ):
                        for i in range(depth):
                            map_dex += 1
                            # print(f"{im_dex:4}", ",", end="")
                            out_map.append(im_dex)
                            # if map_dex % ow == 0:
                            #     print("")
                            im_dex += 1
                    else:
                        for i in range(depth):
                            map_dex += 1
                            wasted_out += 1
                            # print(f"{-1:4}", ",", end="")
                            out_map.append(-1)
                            # if map_dex % ow == 0:
                            #     print("")
                            im_dex += 1
                im_dex += depth * (width - filter_w)
            w_pad += stride_w
        h_pad += stride_h

    return out_map, wasted_out


def col2imv2(
    depth,
    height,
    width,
    filter_h,
    filter_w,
    pad_t,
    pad_l,
    pad_b,
    pad_r,
    stride_h,
    stride_w,
):
    height_col = (height + pad_t + pad_b - filter_h) // stride_h + 1
    width_col = (width + pad_l + pad_r - filter_w) // stride_w + 1
    h_pad = -pad_t
    im_dex = 0
    map_dex = 0

    wasted_out = 0
    out_map = []
    for h in range(height_col):
        w_pad = -pad_l
        for w in range(width_col):
            im_dex = (h_pad * width + w_pad) * depth
            for i in range(depth):
                im_dex = (h_pad * width + w_pad) * depth + i
                for ih in range(filter_h):
                    for iw in range(filter_w):
                        if (
                            ih + h_pad >= 0
                            and ih + h_pad < height
                            and iw + w_pad >= 0
                            and iw + w_pad < width
                        ):
                            map_dex += 1
                            out_map.append(im_dex)
                            im_dex += depth
                        else:
                            map_dex += 1
                            wasted_out += 1
                            out_map.append(-1)
                            im_dex += depth
                    im_dex += depth * (width - filter_w)
            w_pad += stride_w
        h_pad += stride_h

    return out_map, wasted_out


def ComputeOutSize(padding, image_size, filter_size, stride, dilation_rate=1):
    effective_filter_size = (filter_size - 1) * dilation_rate + 1
    if stride == 0:
        return 0

    if padding == "same":
        return (image_size + stride - 1) // stride
    elif padding == "valid":
        return (image_size + stride - effective_filter_size) // stride
    else:
        return 0


def compute_padding_with_offset(
    stride, dilation_rate, in_size, filter_size, out_size, offset=0
):
    effective_filter_size = (filter_size - 1) * dilation_rate + 1
    total_padding = ((out_size - 1) * stride) + effective_filter_size - in_size
    total_padding = total_padding if total_padding > 0 else 0
    offset = total_padding % 2
    return offset, total_padding // 2


def compute_padding_height_width(
    padding,
    stride_height,
    stride_width,
    in_height,
    in_width,
    filter_height,
    filter_width,
):
    dilation_rate_height = 1
    dilation_rate_width = 1

    out_width = ComputeOutSize(
        padding,
        in_width,
        filter_width,
        stride_width,
        dilation_rate_width,
    )
    out_height = ComputeOutSize(
        padding, in_height, filter_height, stride_height, dilation_rate_height
    )

    offset, p_height = compute_padding_with_offset(
        stride_height, dilation_rate_height, in_height, filter_height, out_height, 0
    )
    h_offset = offset
    offset, p_width = compute_padding_with_offset(
        stride_width, dilation_rate_width, in_width, filter_width, out_width, offset
    )
    w_offset = offset
    return p_height, p_width, h_offset, w_offset


def calParams(params):
    stride_x = params[0]
    stride_y = params[1]
    filters = params[2]
    kernel_size = params[3]
    in1 = params[4]
    in2 = params[5]
    in3 = params[6]
    padding_val = params[7]
    out1 = in1 + kernel_size - stride_x
    out2 = in2 + kernel_size - stride_y
    out3 = filters
    rows = filters * kernel_size * kernel_size
    cols = in1 * in2
    depth = in3

    if padding_val == "same":
        out1 = in1 * stride_x
        out2 = in2 * stride_y
    else:
        out1 = in1 + kernel_size - stride_x
        out2 = in2 + kernel_size - stride_y

    ph, pw, pho, pwo = compute_padding_height_width(
        padding_val,
        stride_x,
        stride_y,
        out1,
        out2,
        kernel_size,
        kernel_size,
    )
    pt = ph
    pb = ph + pho
    pl = pw
    pr = pw + pwo
    return rows, cols, depth, out1, out2, out3, pt, pb, pl, pr


def nofSteps(length, stride, kernel_size):
    return int((length - (kernel_size - stride)) / stride)


def tconv_model_info(params):
    stride_x = params[0]
    stride_y = params[1]
    filters = params[2]
    kernel_size = params[3]
    in1 = params[4]
    in2 = params[5]
    in3 = params[6]
    padding_val = params[7]
    rows, cols, depth, out1, out2, out3, pt, pb, pl, pr = calParams(params)
    rdepth = math.ceil(depth / 16) * 16
    total_macs = rows * cols * rdepth
    mm2im_out = out1 * out2 * out3
    return (total_macs, mm2im_out)


def custom_tconv_cols(df):
    # add a new column for stride x
    df["stride_x"] = df["model"].str.split("_").str[1].astype(int)
    df["stride_y"] = df["model"].str.split("_").str[2].astype(int)
    df["filters"] = df["model"].str.split("_").str[3].astype(int)
    df["ks"] = df["model"].str.split("_").str[4].astype(int)
    df["ih"] = df["model"].str.split("_").str[5].astype(int)
    df["iw"] = df["model"].str.split("_").str[6].astype(int)
    df["ic"] = df["model"].str.split("_").str[7].astype(int)
    pf = lambda row: (
        tconv_model_info(
            [
                row["stride_x"],
                row["stride_y"],
                row["filters"],
                row["ks"],
                row["ih"],
                row["iw"],
                row["ic"],
                "same",
            ]
        )
    )
    df["MACs"] = df.apply(pf, axis=1, result_type="expand")[0]
    df["Outputs"] = df.apply(pf, axis=1, result_type="expand")[1]
    df["Compute Intensity"] = df["MACs"] / df["Outputs"]
    # df["Speedup vs. CPU"] = df["Compute Intensity"].astype(int)
    return df


# ============================================================

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

cpu_types = {"KRIA": "aarch64-opt", "Z1": "armhf-opt"}

global log


def log_out(string):
    log.write(string + "\n")
    print(string)


def load_model_config(filename="models.json"):
    with open(filename, "r") as f:
        model_config = json.load(f)

    categories = ["all_models"]
    for model in model_config["models"]:
        if model["category"] not in categories:
            categories.append(model["category"])
    return model_config


def declare_array(f, name, list):
    f.write("declare -a {}_array=(\n".format(name))
    for i in list:
        f.write('  "{}" \n'.format(i))
    f.write(")\n")


class mt(Template):
    delimiter = "£"
    idpattern = r"[a-z][_a-z0-9]*"


def find_hw_config(dir, hw):
    # check if the file.json exists in the directory
    for file in os.listdir(dir):
        if file.endswith(".json"):
            if file == hw + ".json":
                return dir + "/" + file
    # check subdirectories
    for subdir in os.listdir(dir):
        if os.path.isdir(dir + "/" + subdir):
            for file in os.listdir(dir + "/" + subdir):
                if file.endswith(".json"):
                    if file == hw + ".json":
                        return dir + "/" + subdir + "/" + file
    return


def load_config(config_file):
    if config_file.endswith(".json") == False:
        config_file += ".json"
    with open(config_file) as f:
        config = json.load(f)
    return config


def get_board_config(sc, board):
    board_user = sc["boards"][board]["board_user"]
    board_hostname = sc["boards"][board]["board_hostname"]
    board_port = sc["boards"][board]["board_port"]
    board_dir = sc["boards"][board]["board_dir"]
    bench_dir = f"{board_dir}/benchmark_suite"
    bitstream_dir = f"{board_dir}/benchmark_suite/bitstreams"
    bin_dir = f"{board_dir}/benchmark_suite/bins"
    model_dir = f"{board_dir}/benchmark_suite/models"
    board_config = {
        "board_user": board_user,
        "board_hostname": board_hostname,
        "board_port": board_port,
        "board_dir": board_dir,
        "bench_dir": bench_dir,
        "bitstream_dir": bitstream_dir,
        "bin_dir": bin_dir,
        "model_dir": model_dir,
    }
    return board_config


# ============================================================
# Benchmark Configuration Functions
# ============================================================


def find_models_in_path(model, paths):
    for path in paths:
        for root, dirs, files in os.walk(path):
            if f"{model}.tflite" in files:
                return os.path.join(root, f"{model}.tflite")
    return None

def find_relpath_for_model_from_models_dir(model, paths):
    for path in paths:
        for root, dirs, files in os.walk(path):
            if f"{model}.tflite" in files:
                model_path = os.path.relpath(root, start=path)
                model_path = "" if model_path == "." else model_path
                log_out(f"Found {model}.tflite in {root}")
                return model_path
    return None


def generate_benchmark_configs(
    sc, boards, models, layers, hardware, threads, num_runs, hardware_config, time_out
):
    board_hardware_map = {board: [] for board in boards}
    for hw in hardware:
        for folder, hw_list in hardware_config.items():
            for hw_item in hw_list:
                if hw_item["hardware"] == hw:
                    board = hw_item["config"].get("board", "Z1")
                    if board in board_hardware_map:
                        board_hardware_map[board].append(hw)
                    break

    experiment_configs = {}
    model_paths = {}
    for board, hws in board_hardware_map.items():
        log_out(f"Board: {board}")
        config_list = []
        hw_list = []
        model_list = []
        model_path_list = []
        layer_list = []
        thread_list = []
        num_run_list = []
        version_list = []
        del_version_list = []
        delegate_list = []
        out_dir = f"./{sc['out_dir']}"
        experiment_configs[board] = []
        board_config = get_board_config(sc, board)
        for hw in hws:
            log_out(f"  Generating configs for {hw}")
            hw_config_file = find_hw_config(
                f"{sc['secda_tflite_path']}/{sc['hw_configs']}", hw
            )
            hw_config = load_config(hw_config_file)

            # for model in models:
            # Need to append all the selected layers into layer_list per model
            for model in models:
                for thread in threads:
                    hw_list.append(hw_config["acc_name"])
                    model_list.append(model)
                    model_path = find_relpath_for_model_from_models_dir(model, [d.rstrip("/") for d in sc["models_dirs"]])
                    model_path_list.append(model_path)
                    layer_list.append(layers)
                    thread_list.append(thread)
                    num_run_list.append(num_runs)
                    version_str = str(hw_config["acc_version"]) + "_" + str(hw_config["acc_sub_version"])
                    version_list.append(version_str)
                    del_version_list.append(hw_config["del_version"])
                    delegate_list.append(hw_config["del"])
                    config_list.append(hw_config)
                    experiment_configs[board].append(
                        {
                            "hw": hw_config["acc_name"],
                            "model": model,
                            "model_path": model_path,
                            "layer": layers,
                            "thread": thread,
                            "num_runs": num_runs,
                            "version": version_str,
                            "del_version": hw_config["del_version"],
                            "del": hw_config["del"],
                        }
                    )
        uniq_model_list = list(set(model_list))
        for model in uniq_model_list:
            model_path = find_models_in_path(model, [d.rstrip("/") for d in sc["models_dirs"]])
            if model_path:
                model_paths[model] = model_path
                # log_out(f"Model {model} found in {model_path}")
            # else:
            # log_out(f"Model {model} not found in the specified paths.")

        os.makedirs(out_dir, exist_ok=True)
        f = open(f"{out_dir}/configs_{board}.sh", "w+")
        # list of all the config properties
        declare_array(f, "hw", hw_list)
        declare_array(f, "model", model_list)
        declare_array(f, "model_path", model_path_list)
        declare_array(f, "layer", layer_list)
        declare_array(f, "thread", thread_list)
        declare_array(f, "num_run", num_run_list)
        declare_array(f, "version", version_list)
        declare_array(f, "del_version", del_version_list)
        declare_array(f, "del", delegate_list)
        f.close()
        board_config["sudo_type"] = "sudo -i" if board == "KRIA" else "sudo"
        ## Generate run_collect.sh
        r_dict = {
            "board": board,
            "model_dir": board_config["model_dir"],
            "bitstream_dir": board_config["bitstream_dir"],
            "bin_dir": board_config["bin_dir"],
            "board_user": board_config["board_user"],
            "board_dir": board_config["board_dir"],
            "sudo_type": board_config["sudo_type"],
            "time_out": time_out,
        }

        with open("scripts/run_collect.tpl.sh") as f:
            script = str(mt(f.read()).substitute(r_dict))
        with open(f"{out_dir}/run_collect_{board}.sh", "w+") as f:
            f.write(script)

    return experiment_configs, model_paths

    #   log_out(f"Written {out_dir}/configs_{board}.sh")
    #   log_out(f"Written {out_dir}/run_collect_{board}.sh")


# ============================================================
# Binary Generation Functions
# ============================================================


def generate_bazel_buildsim_scripts(sc, hardware):
    bb_pr = "bazel6 build -c opt //"
    bb_po = "--cxxopt='-DSYSC' --cxxopt='-DACC_PROFILE' --copt='-DSECDA_LOGGING_DISABLED' --define tflite_with_xnnpack=false --copt='-DTFLITE_ENABLE_XNNPACK=OFF' --copt='-DTFLITE_WITHOUT_XNNPACK' --cxxopt='-DRUY_OPT_SET=0' --verbose_failures --@secda_tools//:config=sysc"
    path_to_tf = sc["secda_tflite_path"] + "/tensorflow"
    path_to_bench_suite = f"{sc['secda_tflite_path']}/src/benchmark_suite"
    output_path = f"{sc['out_dir']}/gen_bins_sim.sh"
    rdel_path = sc["path_to_dels"]
    delegates_needed = {}
    for hw in hardware:
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
    script += "bazel6 build --jobs 1 //tensorflow/lite/examples/systemc:hello_systemc --spawn_strategy=standalone --define tflite_with_xnnpack=false\n"
    for delegate, vers in delegates_needed.items():
        for ver in vers:
            sn = "bm"
            tool = "benchmark_model"
            del_path = f"{rdel_path}/{delegate}/v{ver}"
            # check if path exists
            if not os.path.exists(sc["secda_tflite_path"] + "/" + del_path):
                del_path = f"{rdel_path}/{delegate}"
            del_path = del_path[del_path.index("/") + 1 :]

            sim_name = f"{sn}_{delegate}_{ver}"
            bin_name = f"{tool}_plus_{delegate}"

            if delegate == "cpu":
                del_path = cpu_paths[tool][0]
                bin_name = cpu_paths[tool][1]

            script += f"{bb_pr}{del_path}:{bin_name} {bb_po} \n"
            script += f"mkdir -p {path_to_bench_suite}/{sc['out_dir']}/bins/ \n"
            script += f"rm -f {path_to_bench_suite}/{sc['out_dir']}/bins/{sim_name}\n"
            script += f"cp {path_to_tf}/bazel-out/k8-opt/bin/{del_path}/{bin_name} {path_to_bench_suite}/{sc['out_dir']}/bins/{sim_name}\n"

    script += "popd\n"

    # create folder to output
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w") as f:
        f.write(script)
    os.system(f"chmod +x {output_path}")
    return delegates_needed


def generate_bazel_build_scripts(sc, boards, hardware, hardware_config):
    board_hardware_map = {board: [] for board in boards}
    for hw in hardware:
        for folder, hw_list in hardware_config.items():
            for hw_item in hw_list:
                if hw_item["hardware"] == hw:
                    board = hw_item["config"].get("board", "Z1")
                    if board in board_hardware_map:
                        board_hardware_map[board].append(hw)
                    break

    for board, hw_list in board_hardware_map.items():
        output_path = f"{sc['out_dir']}/gen_bins_{board}.sh"
        board_config = get_board_config(sc, board)
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
            # bb_po = "--copt='-DSECDA_LOGGING_DISABLED' --copt='-DACC_PROFILE' --define tflite_with_xnnpack=false --copt='-DTFLITE_ENABLE_XNNPACK=OFF' --copt='-DTFLITE_WITHOUT_XNNPACK' --copt='-DACC_NEON' --@secda_tools//:config=fpga"
            bb_po = "--copt='-DSECDA_LOGGING_DISABLED' --cxxopt='-march=armv7-a' --cxxopt='-mfpu=neon' --cxxopt='-funsafe-math-optimizations' --cxxopt='-ftree-vectorize' --copt='-DACC_PROFILE' --define tflite_with_xnnpack=false --copt='-DTFLITE_ENABLE_XNNPACK=OFF' --copt='-DTFLITE_WITHOUT_XNNPACK' --copt='-DACC_NEON' --@secda_tools//:config=fpga"

        elif board == "KRIA":
            bb_pr = "bazel6 build --config=elinux_aarch64 -c opt //"
            bb_po = "--copt='-DSECDA_LOGGING_DISABLED' --copt='-DACC_PROFILE' --define tflite_with_xnnpack=false --copt='-DTFLITE_ENABLE_XNNPACK=OFF' --copt='-DTFLITE_WITHOUT_XNNPACK' --copt='-DACC_NEON' --copt='-DKRIA' --@secda_tools//:config=fpga_arm64"
            # bb_po = "--copt='-DSECDA_LOGGING_DISABLED' --copt='-DACC_PROFILE' --define tflite_with_xnnpack=false --copt='-DTFLITE_ENABLE_XNNPACK=OFF' --copt='-DTFLITE_WITHOUT_XNNPACK' --copt='-DKRIA'"

        delegates_needed = {}
        for hw in hw_list:
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


# ============================================================
# Simulation Functions
# ============================================================


def run_simulations(sc, selected_models, selected_delegated):
    path_to_tf = sc["secda_tflite_path"] + "/tensorflow"
    benchmark_suite_path = f"{sc['secda_tflite_path']}/src/benchmark_suite"
    bin_path = f"{benchmark_suite_path}/{sc['out_dir']}/bins/"
    out_dir = f"{benchmark_suite_path}/results/SIM/"
    os.makedirs(out_dir, exist_ok=True)
    total_runs = len(selected_models) * sum(
        [len(vers) for vers in selected_delegated.values()]
    )
    models_folder = f"{path_to_tf}/models"
    current_run = 0
    for model in selected_models:
        for delegate, vers in selected_delegated.items():
            for ver in vers:
                sn = "bm"
                tool = "benchmark_model"
                sim_name = f"{sn}_{delegate}_{ver}"
                sim_bin = f"{bin_path}/{sim_name}"
                if not os.path.exists(sim_bin):
                    log_out(f"Binary {sim_bin} does not exist")
                    continue
                sim_out = f"{out_dir}{sim_name}.txt"

                command = (
                    f"{sim_bin} --use_gpu=false --num_threads=1 "
                    f"--enable_op_profiling=true --graph={models_folder}/{model}.tflite "
                    f"--num_runs=1 --warmup_runs=0 --warmup_min_secs=0 "
                    f"--use_{delegate}=true --print_postinvoke_state=true "
                )
                log_out(
                    "========================================================================"
                )
                log_out(f"{sim_name} {current_run}/{total_runs}")
                log_out(
                    "========================================================================"
                )
                subprocess.run(
                    f"{command} > {sim_out} 2>&1",
                    shell=True,
                    executable="/bin/bash",
                    # f"{command} 2>&1 | tee {sim_out}", shell=True, executable="/bin/bash"
                )
                current_run += 1
    if os.path.exists(f"prf.csv"):
        os.remove(f"prf.csv")
    if os.path.exists(f"runtime.txt"):
        os.remove(f"runtime.txt")


# ============================================================
# Benchmark Suite Functions
# ============================================================
from datetime import datetime
from scripts.configure_utils import *


def ssh_board(board_info):
    board_user = board_info["board_user"]
    board_hostname = board_info["board_hostname"]
    board_port = board_info["board_port"]
    response = os.system(
        f"ssh -o BatchMode=yes -o ConnectTimeout=2 {board_user}@{board_hostname} -p {board_port} 'exit' > /dev/null 2>&1"
    )
    return response == 0


def ping_board(board_hostname, board_port):
    response = os.system(f"ping -c 1 -p {board_port} {board_hostname} > /dev/null 2>&1")
    log_out(f"ping -c 1 -p {board_port} {board_hostname}")
    return response == 0


def send_models_to_board(
    sc, board, board_hostname, board_port, board_user, bench_dir, board_dir
):  
    models_dir = [d.rstrip("/") for d in sc["models_dirs"]]
    resout = ""
    reserr = ""
    resbool = 0
    for model_dir in models_dir:
        result = subprocess.run(
            f"rsync --include='*.tflite' -r -avz -e 'ssh -p {board_port}' {model_dir}/ {board_user}@{board_hostname}:{bench_dir}/models/",
            shell=True,
            check=False,
            capture_output=True,
            text=True,
        )
        resout += result.stdout
        reserr += result.stderr
        resbool = result.returncode or resbool

    if result.returncode != 0:
        log_out("-----------------------------------------------------------")
        log_out("Error in Transferring Models")
        log_out("-----------------------------------------------------------")
        log_out(result.stdout)
        log_out(result.stderr)
        log.close()


def create_dir(sc, board, board_hostname, board_port, board_user, bench_dir, board_dir):
    resout = ""
    reserr = ""
    resbool = 0
    result = subprocess.run(
        f"ssh -o LogLevel=QUIET -t -p {board_port} {board_user}@{board_hostname} 'mkdir -p {bench_dir} && mkdir -p {board_dir}/bitstreams && mkdir -p {bench_dir}/bins && mkdir -p {bench_dir}/models'",
        shell=True,
        check=False,
        capture_output=True,
        text=True,
    )
    resout += result.stdout
    reserr += result.stderr
    resbool = result.returncode or resbool
    send_models_to_board(
        sc, board, board_hostname, board_port, board_user, bench_dir, board_dir
    )

    result = subprocess.run(
        f"rsync -r -avz -e 'ssh -p {board_port}' ./bitstreams/{board}/ {board_user}@{board_hostname}:{board_dir}/bitstreams/",
        shell=True,
        check=False,
        capture_output=True,
        text=True,
    )
    resout += result.stdout
    reserr += result.stderr
    resbool = result.returncode or resbool

    result = subprocess.run(
        f"rsync -q -r -avz -e 'ssh -p {board_port}' ./scripts/fpga_scripts/ {board_user}@{board_hostname}:{board_dir}/scripts/",
        shell=True,
        check=False,
        capture_output=True,
        text=True,
    )
    resout += result.stdout
    reserr += result.stderr
    resbool = result.returncode or resbool
    if resbool != 0:
        log_out("-----------------------------------------------------------")
        log_out("Error in Transferring Experiment Configs")
        log_out("-----------------------------------------------------------")
        log_out(resout)
        log_out(reserr)
        log.close()

    log_out(f"Initialization Done for {board}")


def verify_board_connections(sc, selected_boards):
    for board in selected_boards:
        if not ssh_board(sc["boards"][board]):
            log_out(f"Could not SSH into {board}")
            return 0
        else:
            log_out(f"Successful SSH into {board}")
    return 1


def init_boards(sc, selected_boards):
    for board in selected_boards:
        board_info = get_board_config(sc, board)
        board_user = board_info["board_user"]
        board_hostname = board_info["board_hostname"]
        board_port = board_info["board_port"]
        bench_dir = board_info["bench_dir"]
        board_dir = board_info["board_dir"]
        create_dir(sc, board, board_hostname, board_port, board_user, bench_dir, board_dir)


def transfer_exp_configs(sc, board):
    board_info = get_board_config(sc, board)
    board_user = board_info["board_user"]
    board_hostname = board_info["board_hostname"]
    board_port = board_info["board_port"]
    bench_dir = board_info["bench_dir"]
    resout = ""
    reserr = ""
    resbool = 0
    result = subprocess.run(
        f"rsync -q -r -avz -e 'ssh -p {board_port}' ./generated/configs_{board}.sh {board_user}@{board_hostname}:{bench_dir}/",
        shell=True,
        check=False,
        capture_output=True,
        text=True,
    )
    resout += result.stdout
    reserr += result.stderr
    resbool = result.returncode or resbool

    result = subprocess.run(
        f"rsync -q -r -avz -e 'ssh -p {board_port}' ./generated/run_collect_{board}.sh {board_user}@{board_hostname}:{bench_dir}/",
        shell=True,
        check=False,
        capture_output=True,
        text=True,
    )
    resout += result.stdout
    reserr += result.stderr
    resbool = result.returncode or resbool
    result = subprocess.run(
        f"ssh -o LogLevel=QUIET -t -p {board_port} {board_user}@{board_hostname} 'cd {bench_dir}/ && chmod +x ./*.sh'",
        shell=True,
        check=False,
        capture_output=True,
        text=True,
    )
    resout += result.stdout
    reserr += result.stderr
    resbool = result.returncode or resbool
    if resbool != 0:
        log_out("-----------------------------------------------------------")
        log_out("Error in Transferring Experiment Configs")
        log_out("-----------------------------------------------------------")
        log_out(resout)
        log_out(reserr)
        log.close()


power_cap_sh_prefix = """
    echo '-----------------------------------------------------------'
    echo 'Initializing Power Capture'
    python3 scripts/record_power.py $name &
    echo $! >/tmp/record_power.py.pid
    echo '-----------------------------------------------------------'
    """

power_cap_sh_postfix = """
    if [[ -e /tmp/record_power.py.pid ]]; then
        kill $(cat /tmp/record_power.py.pid)
        echo "-----------------------------------------------------------"
        echo "Power Capture Done"
        echo "-----------------------------------------------------------"
    else
        echo $(cat /tmp/record_power.py.pid) "not found"
    fi
    """

# power_cap_sh_postfix = """
#     if [[ -e /tmp/record_power.py.pid ]]; then
#         kill $(cat /tmp/record_power.py.pid)
#         echo "-----------------------------------------------------------"
#         echo "Power Capture Done"
#         echo "-----------------------------------------------------------"
#         echo "Processing Power Data"
#         echo "-----------------------------------------------------------"

#         python3 scripts/process_power.py $name $length
#         echo "Power Processing Done"
#         echo "-----------------------------------------------------------"
#     else
#         echo $(cat /tmp/record_power.py.pid) "not found"
#     fi
#     echo "Simple csv:" ./files/${name}_clean.csv
#     echo "-----------------------------------------------------------"
#     """

def run_exp(sc, board, skip_inf_diff, collect_power, test_run, gen_script, name):
    board_info = get_board_config(sc, board)
    board_user = board_info["board_user"]
    board_hostname = board_info["board_hostname"]
    board_port = board_info["board_port"]
    bench_dir = board_info["bench_dir"]
    if gen_script:
        # log_out("-----------------------------------------------------------")
        # log_out(f"Saved Experiment {board} to the run_exp.sh script")
        # log_out("-----------------------------------------------------------")
        script_path = f"./generated/run_exp.sh"
        os.makedirs(os.path.dirname(script_path), exist_ok=True)
        with open(script_path, "a+") as script_file:
            script_file.seek(0)
            first_line = script_file.readline()
            if not first_line.startswith("#!"):
                script_file.write("#!/bin/bash\n")
            if board == "Z1":
                if collect_power:
                    script_file.write(power_cap_sh_prefix)
                script_file.write(
                    f"ssh -o LogLevel=QUIET -t -p {board_port} {board_user}@{board_hostname} 'cd {bench_dir}/ && ./run_collect_{board}.sh 0 {int(skip_inf_diff)} {int(collect_power)} {int(test_run)}'\n"
                )
            elif board == "KRIA":
                script_file.write(
                    f"ssh -o LogLevel=QUIET -t -p {board_port} {board_user}@{board_hostname} '((ls /etc/profile.d/pynq_venv.sh >> /dev/null 2>&1 && source /etc/profile.d/pynq_venv.sh) || echo '') && cd {bench_dir}/ && ./run_collect_{board}.sh 0 {int(skip_inf_diff)} {int(collect_power)} {int(test_run)}'\n"
                )
            if not test_run:
                script_file.write(
                    f"rsync --mkpath -q -r -av -e 'ssh -p {board_port}' {board_user}@{board_hostname}:{bench_dir}/tmp/* ./tmp/\n"
                )
                script_file.write(
                    f"rsync --mkpath -q -r -av -e 'ssh -p {board_port}' {board_user}@{board_hostname}:{bench_dir}/tmp/* ./tmp/{name}_{board}/\n"
                )
        os.chmod(script_path, 0o775)
    else:
        log_out("-----------------------------------------------------------")
        log_out(f"Running {board} Experiments")
        log_out("-----------------------------------------------------------")
        if board == "Z1":
            subprocess.run(
                f"ssh -o LogLevel=QUIET -t -p {board_port} {board_user}@{board_hostname} 'cd {bench_dir}/ && ./run_collect_{board}.sh 0 {int(skip_inf_diff)} {int(collect_power)} {int(test_run)}'",
                shell=True,
            )
        elif board == "KRIA":
            subprocess.run(
                f"ssh -o LogLevel=QUIET -t -p {board_port} {board_user}@{board_hostname} '((ls /etc/profile.d/pynq_venv.sh >> /dev/null 2>&1 && source /etc/profile.d/pynq_venv.sh) || echo '') && cd {bench_dir}/ && ./run_collect_{board}.sh 0 {int(skip_inf_diff)} {int(collect_power)} {int(test_run)}'",
                shell=True,
            )
        if not test_run:
            subprocess.run(
                f"rsync --mkpath -q -r -av -e 'ssh -p {board_port}' {board_user}@{board_hostname}:{bench_dir}/tmp/* ./tmp/",
                shell=True,
            )
            subprocess.run(
                f"rsync --mkpath -q -r -av -e 'ssh -p {board_port}' {board_user}@{board_hostname}:{bench_dir}/tmp/* ./tmp/{name}_{board}/",
                shell=True,
            )
        log_out("-----------------------------------------------------------")
        log_out(f"Finished Running {board} Experiments")
        log_out("-----------------------------------------------------------")


def check_valid(file):
    open_file = open(file, "r")
    avg_error = 0
    # find the line that contains the word "avg_error"
    for line in open_file:
        if "avg_error" in line:
            avg_error = line.split("avg_error=")[1].split(",")[0]

    avg_error = float(avg_error)
    # log_out("Average error: ", avg_error)
    if avg_error < 0.01:
        # log_out("No errors found")
        return 1
    else:
        # log_out("Error found")
        return 0


def get_df_from_csv(csv_path):
    try:
        f = open(csv_path, "r")
    except:
        log_out(csv_path + " does not exist")
        exit(1)
    save = 0
    savedString = ""
    while True:
        line = f.readline()
        if save == 2:
            savedString += line
        if (
            line
            == "============================== Summary by node type ==============================\n"
        ):
            save += 1
        if not line:
            break
    node_table = savedString.split("\n\n")[0].split("\n")
    node_table = [x.replace(" ", "").split(",") for x in node_table]
    df = pd.DataFrame.from_records(node_table)
    df.columns = df.iloc[0]
    df = df[1:]
    return df


def add_new_columns(df, column, value):
    df[column] = value
    return df


def process_layer_details(df, run_dict, run_name):
    # runtime is combination of driver, del and hardware
    runtime = (
        run_dict["hardware"]
        + "_"
        + run_dict["acc_version"]
        + "_"
        + run_dict["del"]
        + "_"
        + run_dict["del_version"]
    )
    df = add_new_columns(df, "Runtime", runtime)
    run_dict["total_latency"] = int(
        df["avg_ms"].astype(float).sum() * 1000
    )  # convert to us
    run_dict["acc_layer"] = int(
        df[df["nodetype"].str.lower().str.contains("del")]["avg_ms"].astype(float).sum()
        * 1000
    )
    run_dict["cpu_layers"] = int(run_dict["total_latency"] - run_dict["acc_layer"])
    return run_dict


# Reports in MICROSECONDS
def process_layer_details_custom(df, run_dict, acc_layer):
    # runtime is combination of driver, del and hardware
    runtime = (
        run_dict["hardware"]
        + "_"
        + run_dict["acc_version"]
        + "_"
        + run_dict["del"]
        + "_"
        + run_dict["del_version"]
    )
    df = add_new_columns(df, "Runtime", runtime)
    run_dict["total_latency"] = int(
        df["avg_ms"].astype(float).sum() * 1000
    )  # convert to us
    run_dict["acc_layer"] = int(
        df[df["nodetype"].str.lower().str.contains("del")]["avg_ms"].astype(float).sum()
        * 1000
    )

    # This for the case where we are running CPU only and no delegate is used  or
    # when accelerator does not cover all the layers we want to accelerate
    # if run_dict["acc_layer"] == 0:
    for layer in acc_layer:
        run_dict["acc_layer"] += int(
            df[df["nodetype"].str.lower()==(layer.lower())]["avg_ms"]
            .astype(float)
            .sum()
            * 1000
        )

    run_dict["cpu_layers"] = int(run_dict["total_latency"] - run_dict["acc_layer"])
    
    # Add a dictionary of all nodetype and their avg_ms
    run_dict["layer_times"] = dict(zip(df["nodetype"], df["avg_ms"].astype(float) * 1000))
    return run_dict


def process_run(
    model,
    thread,
    num_run,
    hardware,
    acc_version,
    delegate,
    del_version,
    valid,
    board,
    runname,
    name,
    acc_layer,
):
    run_dict = {}
    run_dict["model"] = model
    run_dict["thread"] = thread
    run_dict["num_run"] = num_run
    run_dict["hardware"] = hardware
    run_dict["acc_version"] = acc_version
    run_dict["del"] = delegate
    run_dict["del_version"] = del_version
    run_dict["valid"] = valid
    run_dict["runname"] = runname
    run_dict["name"] = name
    # open file called prf.csv and read it line by line
    if run_dict["valid"] == 1:
        acc_prf = {}
        file_missing = False
        if run_dict["hardware"] != "CPU":
            # open file called prf.csv, handle if it does not exist
            try:
                open_file = open(f"tmp/{runname}_prf.csv", "r")
                for line in open_file:
                    # separate each line by comma
                    line = line.replace("\n", "").split(",")
                    acc_prf[line[0]] = line[1]
            except:
                log_out(f"tmp/{runname}_prf.csv" + " does not exist")
                file_missing = True
                pass
        if file_missing:
            log_out(f"tmp/{runname}_prf.csv" + " does not exist")
            run_dict["total_latency"] = 0
            run_dict["acc_layer"] = 0
            run_dict["cpu_layers"] = 0
        else:
            run_dict = {**run_dict, **acc_prf}
            # run_dict = process_layer_details(
            #     get_df_from_csv(f"tmp/{runname}_layer.csv"), run_dict, runname
            # )
            df = get_df_from_csv(f"tmp/{runname}_layer.csv")
            run_dict = process_layer_details_custom(df, run_dict, acc_layer)
            if "CPU" in run_dict["hardware"]:
                # Save pie chart of avg_ms using plotly
                import plotly.express as px

                # pie_chart = px.pie(
                #     df,
                #     values="avg_ms",
                #     names="nodetype",
                #     title=f"Layer-wise Execution Time Breakdown for {runname}",
                # )

                # df["avg_ms"] = df["avg_ms"].astype(float)
                # pie_chart.update_traces(
                #     textinfo="label+percent",
                #     texttemplate="%{label}: %{percent:.1%}",
                #     insidetextorientation="horizontal",
                #     textposition="inside",
                #     pull=[
                #         0.1 if v > 10 else 0
                #         for v in (df["avg_ms"] / (df["avg_ms"].sum() * 100))
                #     ],
                # )
                # pie_chart.update_layout(
                #     title={
                #         "text": f"Layer-wise Execution Time Breakdown for {runname}",
                #         "x": 0.5,  # Center the title
                #         "xanchor": "center",
                #     }
                # )
                # os.makedirs(f"results/{board}/{name}", exist_ok=True)
                # pie_chart.write_image(
                #     f"results/{board}/{name}/{runname}_pie_chart.png",
                #     scale=2,
                #     width=1280,
                #     height=720,
                # )
                # display(pie_chart)

    else:
        log_out(f"Invalid run : {runname}")
        run_dict["total_latency"] = 0
        run_dict["acc_layer"] = 0
        run_dict["cpu_layers"] = 0

    # save dict to json
    os.makedirs(f"results/{board}/{name}", exist_ok=True)
    with open(f"results/{board}/{name}/" + runname + ".json", "w") as fp:
        json.dump(run_dict, fp,indent=4)


import pandas as pd
import glob


def process_all_runs(name, board):
    # get all json files from out folder
    json_files = glob.glob(f"results/{board}/{name}/*.json")
    json_files.sort(key=lambda x: os.path.getmtime(x))
    runs = []
    if len(json_files) == 0:
        log_out("No json files found")
        exit(1)
    for json_file in json_files:
        with open(json_file, "r") as f:
            runs.append(json.load(f))

    common_keys = set.intersection(*map(set, runs))
    # find all the unique keys
    unique_keys = set.union(*map(set, runs)) - common_keys

    # create csv file with common keys as
    common_keys = list(common_keys)
    unique_keys = list(unique_keys)
    cols = common_keys + unique_keys
    df = pd.DataFrame(columns=cols)
    for run in runs:
        df = pd.concat([df, pd.DataFrame(run, index=[0])], ignore_index=True)

    # specialised ordering of columns
    ordered_keys = [
        "model",
        "thread",
        "num_run",
        "hardware",
        "acc_version",
        "del",
        "del_version",
        "valid",
        "acc_layer",
        "cpu_layers",
        "total_latency",
    ]
    unique_keys = unique_keys + [key for key in common_keys if key not in ordered_keys]
    unique_keys.sort()
    final_keys = ordered_keys + unique_keys
    # re-order columns
    df = df[final_keys]

    # save df to csv
    df.to_csv(f"results/{board}/benchmark_summary_{name}.csv")
    df.to_csv(f"results/{board}/latest.csv")
    log_out(
        "Benchmark summary saved to ./results/{}/benchmark_summary_{}.csv".format(
            board, name
        )
    )
    return df


# ============================================================
# Benchmark Suite Functions
# ============================================================
import subprocess
from datetime import datetime
import os
import signal
from IPython.display import display, Markdown


def ctrl_c_handler(signum, frame):
    log_out("Exiting")
    exit(1)


def run_benchmarking_suite(
    sc,
    selected_boards,
    selected_models,
    selected_layers,
    selected_hardware,
    selected_threads,
    selected_num_runs,
    hardware_config,
    exp_config,
    board_results,
    out,
):
    now = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    init = exp_config["init"]
    send_models = exp_config["send_models"]
    skip_bench = exp_config["skip_bench"]
    bin_gen = exp_config["bin_gen"]
    skip_inf_diff = exp_config["skip_inf_diff"]
    collect_power = exp_config["collect_power"]
    gen_script = exp_config["gen_script"]
    time_out = exp_config["time_out"]
    test_run = exp_config["test_run"]
    sim_mode = exp_config["sim_mode"]
    name = f"run_{now}" if exp_config["name"] == "" else exp_config["name"]
    log_file_name = f"logs/{name}.log"
    # make sure the logs directory exists and file is created
    os.makedirs("logs", exist_ok=True)
    global log
    log = open(log_file_name, "w")

    log_out("-----------------------------------------------------------")
    log_out("-- SECDA-TFLite Benchmark Suite --")
    log_out("-----------------------------------------------------------")

    # Verify the connection to the selected boards
    if (init or not skip_bench) and not sim_mode:
        if not verify_board_connections(sc, selected_boards):
            log_out("Benchmark Failed")
            return 1
    selected_boards = ["SIM"] if sim_mode else selected_boards

    # Handle SIGINT
    signal.signal(signal.SIGINT, ctrl_c_handler)

    # Display configurations
    log_out("-----------------------------------------------------------")
    log_out("Configurations")
    log_out("-----------------------------------------------------------")
    log_out(f"Selected Boards: {selected_boards}")
    log_out(f"Skip Bench: {skip_bench}")
    log_out(f"Bin Gen: {bin_gen}")
    log_out(f"Skip Inf Diff: {skip_inf_diff}")
    log_out(f"Collect Power: {collect_power}")
    log_out(f"Generate Run Script: {gen_script}")
    log_out(f"Test Run: {test_run}")
    log_out(f"Time Out: {time_out}")
    log_out(f"Sim Mode: {sim_mode}")
    log_out(f"Name: {name}")
    log_out("-----------------------------------------------------------")

    if sim_mode:
        log_out("-----------------------------------------------------------")
        log_out("Configuring Benchmark for Simulation")
        log_out("-----------------------------------------------------------")
        # Configure the benchmark
        selected_delegated = generate_bazel_buildsim_scripts(sc, selected_hardware)
        if bin_gen:
            log_out("-----------------------------------------------------------")
            log_out("Running Bazel Build Scripts")
            log_out("-----------------------------------------------------------")
            result = subprocess.run(
                f"{sc['out_dir']}/gen_bins_sim.sh",
                shell=True,
                check=False,
                capture_output=True,
                text=True,
            )
            if result.returncode != 0:
                log_out("-----------------------------------------------------------")
                log_out("Error in Bazel Build Scripts")
                log_out("-----------------------------------------------------------")
                log_out(result.stdout)
                log_out(result.stderr)
                log.close()
                return 1

            log_out("-----------------------------------------------------------")

        log_out("-----------------------------------------------------------")
        log_out("Running Simulation")
        log_out("-----------------------------------------------------------")
        run_simulations(sc, selected_models, selected_delegated)
        log_out("-----------------------------------------------------------")
        log_out("Finished Running Simulation")
        log_out("-----------------------------------------------------------")

    else:
        log_out("-----------------------------------------------------------")
        log_out("Configuring Benchmark")
        log_out("-----------------------------------------------------------")
        # Configure the benchmark
        experiment_configs, models_path = generate_benchmark_configs(
            sc,
            selected_boards,
            selected_models,
            selected_layers,
            selected_hardware,
            selected_threads,
            selected_num_runs,
            hardware_config,
            time_out,
        )

        if init:
            log_out("-----------------------------------------------------------")
            log_out("Initialising Boards")
            log_out("Models Sent to Boards")
            log_out("-----------------------------------------------------------")
            init_boards(sc, selected_boards)

        if send_models and not init:
            log_out("-----------------------------------------------------------")
            log_out("Models Sent to Boards")
            log_out("-----------------------------------------------------------")
            for board in selected_boards:
                board_info = get_board_config(sc, board)
                board_hostname = board_info["board_hostname"]
                board_port = board_info["board_port"]
                board_user = board_info["board_user"]
                bench_dir = board_info["bench_dir"]
                board_dir = board_info["board_dir"]
                send_models_to_board(sc,
                    board, board_hostname, board_port, board_user, bench_dir, board_dir
                )

        if not skip_bench:
            if bin_gen:
                log_out("-----------------------------------------------------------")
                log_out("Configuring Bazel Build Scripts")
                log_out("-----------------------------------------------------------")
                # Configure the benchmark
                generate_bazel_build_scripts(
                    sc, selected_boards, selected_hardware, hardware_config
                )
                log_out("-----------------------------------------------------------")
                log_out("Running Bazel Build Scripts")
                log_out("-----------------------------------------------------------")
                for board in selected_boards:
                    result = subprocess.run(
                        f"./generated/gen_bins_{board}.sh",
                        shell=True,
                        check=False,
                        capture_output=True,
                        text=True,
                    )
                    if result.returncode != 0:
                        log_out(
                            "-----------------------------------------------------------"
                        )
                        log_out("Error in Bazel Build Scripts for " + board)
                        log_out(
                            "-----------------------------------------------------------"
                        )
                        log_out(result.stdout)
                        log_out(result.stderr)
                        log.close()
                        return 1

            if gen_script:
                log_out("-----------------------------------------------------------")
                log_out("Generating Run Script")
                log_out("-----------------------------------------------------------")
                script_path = f"./generated/run_exp.sh"
                os.makedirs(os.path.dirname(script_path), exist_ok=True)
                with open(script_path, "w") as script_file:
                    script_file.write("#!/bin/bash\n")
                    script_file.write(f"name=" + name + "\n")
                    script_file.write(f"pushd {sc['secda_tflite_path']}/src/benchmark_suite\n")

                os.chmod(script_path, 0o775)

            for board in selected_boards:
                log_out("-----------------------------------------------------------")
                log_out(f"Transferring Experiment Configurations to the {board} board")
                log_out("-----------------------------------------------------------")
                transfer_exp_configs(sc, board)

                # I can addedd in automatic copying of selected models to the board here
                # if needed

                # Runs exp or generates the run_exp.sh script
                run_exp(
                    sc, board, skip_inf_diff, collect_power, test_run, gen_script, name
                )

            if gen_script:
                script_path = f"./generated/run_exp.sh"
                with open(script_path, "a") as script_file:
                    if collect_power:
                        script_file.write(power_cap_sh_postfix)
                log_out("-----------------------------------------------------------")
                log_out("Saved Experiment to the run_exp.sh script")
                log_out("-----------------------------------------------------------")
                log_out(f"Run the following command to execute the experiments:")
                log_out(f"./generated/run_exp.sh")
                log_out("-----------------------------------------------------------")

        if not test_run and not gen_script:
            log_out("-----------------------------------------------------------")
            log_out("Processing Results")
            log_out("-----------------------------------------------------------")
            for board in selected_boards:
                i = 0
                length = len(experiment_configs[board])
                for exp in experiment_configs[board]:
                    i += 1
                    hw = exp["hw"]
                    model = exp["model"]
                    layer = exp["layer"]
                    thread = exp["thread"]
                    num_runs = exp["num_runs"]
                    version = exp["version"]
                    delegate = exp["del"]
                    del_version = exp["del_version"]
                    runname = f"{hw}_{version}_{delegate}_{del_version}_{model}_{thread}_{num_runs}_{board}"
                    # if i % 1 == 0:
                    #     log_out("========================================================================")
                    #     log_out(f"{runname} {i}/{length}")
                    #     log_out("========================================================================")
                    valid = 1
                    if "CPU" not in hw and not skip_inf_diff:
                        if not os.path.exists(f"tmp/{runname}_id.txt"):
                            log_out(
                                f"Correct Check File Missing for: tmp/{runname}_id.txt"
                            )
                            valid = 0
                        else:
                            valid = check_valid(f"./tmp/{runname}_id.txt")
                            if not valid:
                                log_out(f"Failed Inference Check for: {runname}")

                    process_run(
                        model,
                        thread,
                        num_runs,
                        hw,
                        version,
                        delegate,
                        del_version,
                        valid,
                        board,
                        runname,
                        name,
                        layer,
                    )

                results_df = process_all_runs(name, board)
                board_results[board] = results_df
            log_out("-----------------------------------------------------------")
            log_out("Finished Processing Results")
            log_out("-----------------------------------------------------------")
    log_out("-----------------------------------------------------------")
    log_out("Exiting SECDA-TFLite Benchmark Suite")
    log_out("-----------------------------------------------------------")
    log.close()
    # sys.stdout = sys.__stdout__
    # sys.stderr = sys.__stderr__


import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.ticker as ticker
import matplotlib
import os


def plot_profiles(folder, show_all=False, show_x=1):
    # cm = plt.get_cmap("Accent").reversed()
    cm = plt.get_cmap("tab20").reversed()
    NUM_COLORS = 7
    font = {"family": "sans-serif", "weight": "normal", "size": 7}
    matplotlib.rc("font", **font)

    def plot_h(df):
        for index, row in df.iterrows():
            for col in df.columns:
                if not col.startswith("T_"):
                    row = row.drop([col])

            T_rows = [row.split("_")[1] for row in df.columns if row.startswith("T_")]
            T_rows = list(set(T_rows))
            sdf = pd.DataFrame()
            Tlen = T_rows.__len__()
            fig, axs = plt.subplots(Tlen, 1, figsize=(10, Tlen * 2))
            # reorder T_rows alphabetically
            T_rows = sorted(T_rows)
            for id, T_row in enumerate(T_rows):
                all_T_row_cols = [
                    row for row in df.columns if row.startswith("T_" + T_row)
                ]
                ndf = df[all_T_row_cols].sort_values(by=0, axis=1, ascending=True)
                # ndf = df[all_T_row_cols]

                ndf = ndf.iloc[index]
                ndf = ndf.rename(T_row)
                ndf = ndf.to_frame().T
                ax = axs[id]
                pd.DataFrame(ndf).plot(
                    kind="barh", stacked=True, ax=ax, colormap=cm, width=0.3
                )
                ax.set_prop_cycle(color=[cm(5 + 1.0 * i) for i in range(NUM_COLORS)])
                legends = [
                    i.replace("T_", "").replace(T_row, "S") for i in list(ndf.columns)
                ]
                total_cycles = ndf.sum(axis=1).values[0]

                ax.legend(
                    legends,
                    loc="upper center",
                    bbox_to_anchor=(0.5, 1),
                    prop={"size": 7},
                    ncol=10,
                )

                ax.set_xlabel(f"{T_row} | Total Clock Cycles: {total_cycles}")
                ax.set_xlabel(ax.get_xlabel(), fontweight="normal", fontsize=9)
                ax.set_yticklabels("")
                ax.xaxis.set_major_formatter(
                    ticker.FuncFormatter(lambda x, _: "{:,.0f}K".format(x / 1000))
                )
                ax.tick_params(axis="both", which="major", labelsize=9)
                # add vertical line between all bar colour
                for i in range(1, len(legends)):
                    ax.axvline(x=i - 0.5, color="black", linewidth=0.5)

            # plt.subplots_adjust(wspace=0.5, hspace=0.5/ Tlen)
            # make save between subplots bigger
            plt.tight_layout()
            plt.show()
            break

    profs = sorted(os.listdir(folder), key=lambda x: os.path.getctime(folder + x))
    profs.reverse()
    for prof in profs:
        filename = folder + prof
        df = pd.read_csv(filename, sep=",", header=0)
        print(prof)
        plot_h(df)
        show_x -= 1
        if show_x == 0 and not show_all:
            break


def merge_rows(df, custom_cols=False, board="Z1"):
    df["acc_version"] = df["acc_version"].astype(str)
    df["hardware"] = df["hardware"].astype(str)
    df["acc_version_hardware"] = df["acc_version"] + "_" + df["hardware"]
    acc_version_hardware = df["acc_version_hardware"].unique()
    model = df["model"].unique()
    if "CPU" not in acc_version_hardware:
        return

    df2 = pd.DataFrame(columns=["model", "thread"] + list(acc_version_hardware))
    for m in model:
        for t in [1, 2]:
            df3 = df.loc[(df["model"] == m) & (df["thread"] == t)]
            df2 = pd.concat(
                [df2, pd.DataFrame([[m, t]], columns=["model", "thread"])],
                ignore_index=True,
            )
            for a in acc_version_hardware:
                df4 = df3.loc[df3["acc_version_hardware"] == a]
                if df4.empty:
                    total_latency = 0
                else:
                    total_latency = df4["total_latency"].sum()
                df2.loc[(df2["model"] == m) & (df2["thread"] == t), a] = total_latency
                if total_latency == 0:
                    return
    df2 = df2.sort_values(by=["1_0_CPU", "model", "thread"])

    if custom_cols:
        df2 = custom_cols(df2)
    name = df.iloc[0]["name"]
    df2.to_csv(f"results/{board}/merged_{name}.csv")
    return df2


def process_compare_with_cpu(board_results):
    for board, result_df in board_results.items():
        if result_df.empty:
            continue
        name = result_df.iloc[0]["name"]
        if "tconv" in name:
            merge_rows(result_df, custom_tconv_cols, board)
        result_df = result_df[
            [
                "model",
                "thread",
                "num_run",
                "hardware",
                "acc_version",
                "del",
                "del_version",
                "valid",
                "acc_layer",
                "cpu_layers",
                "total_latency",
                "name",
                "runname",
            ]
        ]
        cpu_df = result_df[result_df["hardware"] == "CPU"]
        name = result_df.iloc[0]["name"]

        final_df = result_df.copy()
        final_df["acc_layer_cpu"] = 0
        final_df["cpu_layers_cpu"] = 0
        final_df["total_latency_cpu"] = 0
        final_df["acc_layer_speedup"] = 0
        final_df["cpu_layers_speedup"] = 0
        final_df["total_latency_speedup"] = 0

        for index, row in final_df.iterrows():
            related_cpu_row = cpu_df[
                (cpu_df["model"] == row["model"])
                & (cpu_df["thread"] == row["thread"])
                & (cpu_df["num_run"] == row["num_run"])
                & (cpu_df["name"] == row["name"])
            ]
            if not related_cpu_row.empty:
                final_df.at[index, "acc_layer_cpu"] = related_cpu_row.iloc[0][
                    "acc_layer"
                ]
                final_df.at[index, "cpu_layers_cpu"] = related_cpu_row.iloc[0][
                    "cpu_layers"
                ]
                final_df.at[index, "total_latency_cpu"] = related_cpu_row.iloc[0][
                    "total_latency"
                ]
                if row["acc_layer"] > 0:
                    final_df.at[index, "acc_layer_speedup"] = round(
                        related_cpu_row.iloc[0]["acc_layer"] / row["acc_layer"], 3
                    )
                if row["cpu_layers"] > 0:
                    final_df.at[index, "cpu_layers_speedup"] = round(
                        related_cpu_row.iloc[0]["cpu_layers"] / row["cpu_layers"], 3
                    )
                if row["total_latency"] > 0:
                    final_df.at[index, "total_latency_speedup"] = round(
                        related_cpu_row.iloc[0]["total_latency"] / row["total_latency"],
                        3,
                    )

        final_df["acc_layer"] = (final_df["acc_layer"].astype(float) / 1000).round(2)
        final_df["cpu_layers"] = (final_df["cpu_layers"].astype(float) / 1000).round(2)
        final_df["total_latency"] = (
            final_df["total_latency"].astype(float) / 1000
        ).round(2)

        final_df["acc_layer_cpu"] = (
            final_df["acc_layer_cpu"].astype(float) / 1000
        ).round(2)
        final_df["cpu_layers_cpu"] = (
            final_df["cpu_layers_cpu"].astype(float) / 1000
        ).round(2)
        final_df["total_latency_cpu"] = (
            final_df["total_latency_cpu"].astype(float) / 1000
        ).round(2)

        final_df["acc_layer_speedup"] = final_df["acc_layer_speedup"].apply(
            lambda x: round(x, 2)
        )
        final_df["cpu_layers_speedup"] = final_df["cpu_layers_speedup"].apply(
            lambda x: round(x, 2)
        )
        final_df["total_latency_speedup"] = final_df["total_latency_speedup"].apply(
            lambda x: round(x, 2)
        )

        final_df = final_df[
            [
                "model",
                "thread",
                "num_run",
                "hardware",
                "acc_version",
                "del",
                "del_version",
                "valid",
                "acc_layer",
                "cpu_layers",
                "total_latency",
                "acc_layer_cpu",
                "cpu_layers_cpu",
                "total_latency_cpu",
                "acc_layer_speedup",
                "cpu_layers_speedup",
                "total_latency_speedup",
                "name",
                "runname",
            ]
        ]

        board_results[board] = final_df
        final_df.to_csv(f"results/{board}/cpu_comparison_{name}.csv")
    return board_results
