import sys
sys.dont_write_bytecode = True
import json
from gen_benchmark import gen_bench
from gen_bins import gen_bins
from benchmark_utils import *


# parse arguments if any
arglen = len(sys.argv)
if arglen > 1:
    gen_bin = sys.argv[1]

sc = load_config("../../config.json")
board_user = sc["board_user"]


def create_exp(sc, exp):
    print("Creating experiment")
    print("Generating benchmark")
    gen_bench(sc, exp)
    if gen_bin:
        print("Generating bins")
        gen_bins(sc, exp)


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
bitstream_dir = f"/home/{board_user}/Workspace/secda_tflite/benchmark_suite/bitstreams"
bin_dir = f"/home/{board_user}/Workspace/secda_tflite/benchmark_suite/bins"

# TCONV Synth Experiment
models = tconv_models_synth
hardware = ["MM2IMv2_4", "CPU"]
threads = [1, 2]
num_run = 1
model_dir = f"/home/{board_user}/Workspace/secda_tflite/benchmark_suite/models/tconv"
tconv_synth_exp = [
    models,
    hardware,
    threads,
    num_run,
    model_dir,
    bitstream_dir,
    bin_dir,
    board_user,
]


# CONV Experiment
# models = conv_models
models=["mobilenetv2"]
# hardware = ["CPU","VMRPPv2_0","VMRPP_SH_QKv2_0"]
hardware = ["VMv3_0","CPU"]
threads = [2]
num_run = 10
model_dir = f"/home/{board_user}/Workspace/secda_tflite/benchmark_suite/models/"
conv_exp = [
    models,
    hardware,
    threads,
    num_run,
    model_dir,
    bitstream_dir,
    bin_dir,
    board_user,
]


# DCGAN Experiment
models = ["dcgan_gen"]
hardware = ["MM2IMv2_3", "MM2IMv2_4", "CPU", "MM2IMv2_4"]
# hardware = ["MM2IMv2_51"]
threads = [1]
num_run = 10
# model_dir = f"/home/{board_user}/Workspace/secda_tflite/benchmark_suite/models/tconv"
model_dir = f"/home/{board_user}/Workspace/secda_tflite/benchmark_suite/models"
dc_gan_exp = [
    models,
    hardware,
    threads,
    num_run,
    model_dir,
    bitstream_dir,
    bin_dir,
    board_user,
]

# GAN Experiment
models = gan_models
hardware = ["MM2IMv2_3", "CPU"]
threads = [1, 2]
num_run = 1
model_dir = f"/home/{board_user}/Workspace/secda_tflite/benchmark_suite/models/gans"
gan_exp = [
    models,
    hardware,
    threads,
    num_run,
    model_dir,
    bitstream_dir,
    bin_dir,
    board_user,
]





# Test Experiment
# models=["conv_1_1_96_1_112_112_16"]
# models=["mobilenetv2"]

models=["mobilenetv1"]


# hardware = ["VMRPP_GEMM2_200Mv4_1"]
hardware = ["VMv3_0"]
threads = [1]
num_run = 1
model_dir = f"/home/{board_user}/Workspace/secda_tflite/benchmark_suite/models/"
test_exp = [
    models,
    hardware,
    threads,
    num_run,
    model_dir,
    bitstream_dir,
    bin_dir,
    board_user,
]

####################################################
####################################################

# Current experiment
create_exp(sc, test_exp)
