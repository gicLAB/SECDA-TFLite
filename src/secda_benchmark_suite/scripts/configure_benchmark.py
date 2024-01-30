from string import Template
import json

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

# automatically generated tconv models
with open("model_gen/configs/tconv_models.json") as f:
    tconv_models = json.load(f)["tconv_models"]

####################################################
## HARDWARE
####################################################
all_supported_hardware = ["vm_3_0", "sa_2_0", "cpu", "mm2im_1_0"]
cpu_only = ["cpu"]
conv_only = ["vm_3_0", "sa_2_0"]
tconv_only = ["mm2im_1_0", "mm2im_2_0", "mm2im_2_1"]

####################################################
## CURRENT CONFIG
####################################################
# Current benchmark suite config
# models = conv_models
# hardware = ["vm_3_0", "cpu", "sa_2_0"]
# threads = [1, 2]
# num_run = 1

models = tconv_models
# models = ["dcgan_gen"]
# hardware = ["mm2im_1_0", "cpu", "mm2im_2_0"]
hardware = ["mm2im_2_1"]
threads = [1, 2]
num_run = 1

# directories within the target board
# model_dir = "/home/xilinx/Workspace/secda_benchmark_suite/models"
model_dir = "/home/xilinx/Workspace/secda_benchmark_suite/models/tconv"
bitstream_dir = "/home/xilinx/Workspace/secda_benchmark_suite/bitstreams"
bin_dir = "/home/xilinx/Workspace/secda_benchmark_suite/bins"


####################################################
####################################################


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
r_dict = {"model_dir": model_dir, "bitstream_dir": bitstream_dir, "bin_dir": bin_dir}
with open("scripts/run_collect.tpl.sh") as f:
    script = str(mt(f.read()).substitute(r_dict))
with open("generated/run_collect.sh", "w+") as f:
    f.write(script)
