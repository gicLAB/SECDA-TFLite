from string import Template
import json
from gen_benchmark import gen_benchmark

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
    "mobilenetv1",
    "mobilenetv2",
    "resnet18v1",
    "efficientnet_lite",
    "inceptionv1",
    "inceptionv3",
    "resnet50v2",
]

# manually supported bert models
bert_models = ["albert_int8", "mobilebert_quant"]

# manually supported gan models
gan_models = [
    "dcgan_gen",
    "new_cycle_gan_f",
    "new_cycle_gan_g",
    "pix2pix_g",
    "magenta_gen",
]

add_models = ["add_simple"]

# automatically generated tconv models
with open("model_gen/configs/tconv_models_synth.json") as f:
    tconv_models_synth = json.load(f)["tconv_models_synth"]

with open("model_gen/configs/dcgan_layers.json") as f:
    dcgan_layers = json.load(f)["dcgan_layers"]

####################################################
## HARDWARE
####################################################
all_supported_hardware = [
    "vm_3_0",
    "sa_2_0",
    "cpu",
    "mm2im_1_0",
    "toyadd_1_0",
    "mm2im_2_0",
    "mm2im_2_1",
    "mm2im_2_2",
    "mm2im_2_3",
]
conv_only = ["vm_3_0", "sa_2_0"]
tconv_only = ["mm2im_1_0", "mm2im_2_0", "mm2im_2_1", "mm2im_2_2", "mm2im_2_3"]
add_only = ["toyadd_1_0", "cpu"]


####################################################
## EXPERIMENT CONFIGS
####################################################
bitstream_dir = "/home/xilinx/Workspace/secda_benchmark_suite/bitstreams"
bin_dir = "/home/xilinx/Workspace/secda_benchmark_suite/bins"

# TCONV Synth Experiment
models = tconv_models_synth
# hardware = ["mm2im_1_0", "cpu", "mm2im_2_0"]
hardware = ["mm2im_2_1"]
threads = [1, 2]
num_run = 1
model_dir = "/home/xilinx/Workspace/secda_benchmark_suite/models/tconv"
tconv_synth_exp = [
    models,
    hardware,
    threads,
    num_run,
    model_dir,
    bitstream_dir,
    bin_dir,
]


# ADD Experiment
hardware = add_only
models = add_models
threads = [1, 2]
num_run = 1
model_dir = "/home/xilinx/Workspace/secda_benchmark_suite/models"
add_exp = [models, hardware, threads, num_run, model_dir, bitstream_dir, bin_dir]

# CONV Experiment
models = conv_models
# hardware = ["vm_3_0", "cpu", "sa_2_0"]
hardware = ["vm_3_0", "cpu"]
threads = [1]
num_run = 1
model_dir = "/home/xilinx/Workspace/secda_benchmark_suite/models"
conv_exp = [models, hardware, threads, num_run, model_dir, bitstream_dir, bin_dir]


# DCGAN Experiment
# models = dcgan_layers
models = ["tconv_2_2_512_5_4_4_1024"]
hardware = ["mm2im_2_3", "cpu"]
threads = [1, 2]
num_run = 1
model_dir = "/home/xilinx/Workspace/secda_benchmark_suite/models/tconv"
dc_gan_exp = [models, hardware, threads, num_run, model_dir, bitstream_dir, bin_dir]


####################################################
####################################################

# Current experiment
gen_benchmark(dc_gan_exp)

