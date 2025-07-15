import sys

sys.dont_write_bytecode = True
import json
from gen_benchmark import gen_bench
from gen_bins import gen_bins
from benchmark_utils import *

import argparse


# parse arguments if any
# arglen = len(sys.argv)
# if arglen > 1:
#     gen_config = sys.argv[1]
# if arglen > 2:
#     gen_config = sys.argv[1]
#     gen_bin = sys.argv[2]


## argument parsing using argparse
parser = argparse.ArgumentParser()
# add integer arguments with short and long names
parser.add_argument("-c", "--gen_config", type=int, help="Generate config", default=0)
parser.add_argument("-b", "--gen_bin", type=int, help="Generate bin", default=0)
parser.add_argument("-e", "--gen_bench", type=int, help="Generate benchmark", default=0)


args = parser.parse_args()
f_gen_config = args.gen_config
f_gen_bin = args.gen_bin
f_gen_bench = args.gen_bench


def create_exp(sc, exp):
    board_config_keys = [
        "board",
        "board_user",
        "board_hostname",
        "board_port",
        "board_dir",
    ]
    exp_dict = dict(zip(board_config_keys, exp[7]))
    exp[7] = exp_dict
    if f_gen_config:
        print("Loading Board Config")
        with open(f"{sc['out_dir']}/board_config.json", "w") as f:
            json.dump(exp_dict, f)

    if f_gen_bench:
        print("Creating Experiment")
        print("Generating benchmark")
        gen_bench(sc, exp)

    if f_gen_bin:
        print("Generating Binaries")
        gen_bins(sc, exp, exp_dict)


def get_board_config(sc, board):
    board_user = sc["boards"][board]["board_user"]
    board_hostname = sc["boards"][board]["board_hostname"]
    board_port = sc["boards"][board]["board_port"]
    board_dir = sc["boards"][board]["board_dir"]
    bitstream_dir = f"{board_dir}/benchmark_suite/bitstreams"
    bin_dir = f"{board_dir}/benchmark_suite/bins"
    return board_user, board_hostname, board_port, board_dir, bitstream_dir, bin_dir


####################################################
## MODELS
####################################################
# All manually supported models
all_models = [
    "mobilenetv1",
    "resnet18v1",
    "albert_int8",
    "efficientnet_lite",
    "inceptionv1",
    "inceptionv3",
    "mobilebert_quant",
    "mobilenetv2",
    # "mobilenetv3", # not supported
    "resnet50v2",
    "dcgan_gen",
    "new_cycle_gan_f",
    "new_cycle_gan_g",
    "pix2pix_g",
    "magenta_gen",
    "add_simple",
]

# manually supported conv models
conv_models = [
    # "mobilenetv1",
    "mobilenetv2",
    "resnet18v1",
    # "efficientnet_lite",
    "inceptionv1",
    # "inceptionv3",
    # "resnet50v2",
]

# manually supported bert models
bert_models = ["albert_int8", "mobilebert_quant"]

# manually supported gan models
gan_models = [
    "dcgan_gen",
    # "new_cycle_gan_f",
    # "new_cycle_gan_g",
    # "pix2pix_g",
    # "magenta_gen",
    # "esrgan"
]

add_models = ["add_simple"]

# automatically generated tconv models
with open("model_gen/configs/tconv_models_synth.json") as f:
    tconv_models_synth = json.load(f)["tconv_models_synth"]

with open("model_gen/configs/dcgan_layers.json") as f:
    dcgan_layers = json.load(f)["dcgan_layers"]

with open("model_gen/configs/tf_dcgan_layers.json") as f:
    tf_dcgan_layers = json.load(f)["tf_dcgan_layers"]

with open("model_gen/configs/conv_models.json") as f:
    conv_models_pot_exp = json.load(f)["conv_models"]

with open("model_gen/configs/pix2pix_models.json") as f:
    pix2pix_models = json.load(f)["pix2pix_models"]

# with open("model_gen/configs/mnk_broke.json") as f:
#     mnk_models = json.load(f)["mnk_broke"]

####################################################
## HARDWARE
####################################################
all_supported_hardware = [
    "CPU",
    "VMv3_0",
    "VMv4_0",
    "SAv3_0",
    "MM2IMv2_3",
    "MM2IMv2_4",
]
# conv_only = ["vm_3_0", "sa_2_0"]
# tconv_only = ["mm2im_1_0", "mm2im_2_0", "mm2im_2_1", "mm2im_2_2", "mm2im_2_3"]
# add_only = ["toyadd_1_0", "cpu"]


####################################################
## EXPERIMENT CONFIGS
####################################################
sc = load_config("../../config.json")
# board_user = sc["board_user"]
# bitstream_dir = f"/home/{board_user}/Workspace/secda_tflite/benchmark_suite/bitstreams"
# bin_dir = f"/home/{board_user}/Workspace/secda_tflite/benchmark_suite/bins"

# TCONV Synth Experiment
models = tconv_models_synth
hardware = ["MM2IMv2_4", "CPU"]
threads = [1, 2]
num_run = 1
board = "Z1"
board_user, board_hostname, board_port, board_dir, bitstream_dir, bin_dir = (
    get_board_config(sc, board)
)
board_config = [board, board_user, board_hostname, board_port, board_dir]

model_dir = f"{board_dir}/benchmark_suite/models/tconv"
tconv_synth_exp = [
    models,
    hardware,
    threads,
    num_run,
    model_dir,
    bitstream_dir,
    bin_dir,
    board_config,
]


# CONV Experiment
# models = conv_models
models = ["mobilenetv2"]
# hardware = ["CPU","VMRPPv2_0","VMRPP_SH_QKv2_0"]
hardware = ["VMv3_0", "CPU"]
threads = [2]
num_run = 10
board = "Z1"
board_user, board_hostname, board_port, board_dir, bitstream_dir, bin_dir = (
    get_board_config(sc, board)
)
board_config = [board, board_user, board_hostname, board_port, board_dir]

model_dir = f"{board_dir}/benchmark_suite/models/"
conv_exp = [
    models,
    hardware,
    threads,
    num_run,
    model_dir,
    bitstream_dir,
    bin_dir,
    board_config,
]




# GAN Experiment
models = gan_models
hardware = ["MM2IMv2_3", "CPU"]
threads = [1, 2]
num_run = 1
board = "Z1"
board_user, board_hostname, board_port, board_dir, bitstream_dir, bin_dir = (
    get_board_config(sc, board)
)
board_config = [board, board_user, board_hostname, board_port, board_dir]
model_dir = f"{board_dir}/benchmark_suite/models/gans"
gan_exp = [
    models,
    hardware,
    threads,
    num_run,
    model_dir,
    bitstream_dir,
    bin_dir,
    board_config,
]


# Test Experiment

## Hardware Config
# hardware = ["VMv4_0_KRIA", "CPU_KRIA"]
# board = "KRIA"
# board_user, board_hostname, board_port, board_dir, bitstream_dir, bin_dir = (
#     get_board_config(sc, board)
# )
# board_config = [board, board_user, board_hostname, board_port, board_dir]

hardware = ["VMv4_0"]
board = "Z1"
board_user, board_hostname, board_port, board_dir, bitstream_dir, bin_dir = (
    get_board_config(sc, board)
)
board_config = [board, board_user, board_hostname, board_port, board_dir]


## Inference Parameters
models = ["mobilenetv1"]
threads = [1]
num_run = 1
model_dir = f"{board_dir}/benchmark_suite/models/"
test_exp = [
    models,
    hardware,
    threads,
    num_run,
    model_dir,
    bitstream_dir,
    bin_dir,
    board_config,
]





# DCGAN Experiment

# Works
# models = ["dcgan_gen"]
# models = ["tconv_2_2_7_4_14_14_64"]
# models = ["tconv_2_2_3_5_4_4_64"]
# models = ["tconv_2_2_1_8_14_14_64"]


# Doesn't work
models = ["tconv_2_2_8_4_14_14_64"] # produces error  



# models = pix2pix_models
# models = ["pix2pix_g"]

# hardware = ["MM2IMv2_3", "MM2IMv2_4", "CPU", "MM2IMv2_4"]
hardware = ["MM2IMv2_7"]
# hardware = ["MM2IMv2_61"]

threads = [1]
num_run = 1
board = "Z1"
board_user, board_hostname, board_port, board_dir, bitstream_dir, bin_dir = (
    get_board_config(sc, board)
)
board_config = [board, board_user, board_hostname, board_port, board_dir]
# model_dir = f"/home/{board_user}/Workspace/secda_tflite/benchmark_suite/models/tconv"
# model_dir = f"{board_dir}/benchmark_suite/models/pix2/"
model_dir = f"{board_dir}/benchmark_suite/models/"
dc_gan_exp = [
    models,
    hardware,
    threads,
    num_run,
    model_dir,
    bitstream_dir,
    bin_dir,
    board_config,
]
####################################################
####################################################

# Current experiment

create_exp(sc, dc_gan_exp)


# pix2pix_models = [
#     "tconv_2_2_512_4_1_1_512",
#     "tconv_2_2_512_4_2_2_1024",
#     "tconv_2_2_512_4_4_4_1024",
#     "tconv_2_2_512_4_8_8_1024",
#     "tconv_2_2_256_4_16_16_1024",
#     "tconv_2_2_128_4_32_32_512",
#     "tconv_2_2_64_4_64_64_256"
# ]

# f,1
# ks,5
# ih,14
# iw,14
# ic,64
# oh,28
# ow,28
# oc,1
# rows: 25, cols: 196, depth: 64
# stride_x: 2
# stride_y: 2
# *******************
# f,1
# ks,8
# ih,14
# iw,14
# ic,64
# oh,28
# ow,28
# oc,1
# rows: 64, cols: 196, depth: 64
# stride_x: 2
# stride_y: 2
# *******************