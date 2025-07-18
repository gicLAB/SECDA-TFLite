import os
from os import walk
import sys
import json
import argparse

supported_layers = {
    "CONV2D": "isCONV2D",
    "DWCONV2D": "isDWCONV2D",
    "ADD": "isADD",
    "FULLY_CONNECTED": "isFC",
    "TCONV": "isTCONV",
    "SHAPE": "isSHAPE",
    "SOFTMAX": "isSOFTMAX",
    "PAD": "isPAD",
    "MEAN": "isMEAN",
    "QUANTIZE": "isQUANTIZE",
    "DEQUANTIZE": "isDEQUANTIZE",
}


## Create a new file with new name and if line contains search_text, replace with new_text
def create_new_file(file_path, new_file_name, replace_dict):
    with open(file_path, "r") as f:
        lines = (line.rstrip() for line in f)
        altered_lines = []
        for line in lines:
            altered_line = line
            for key, value in replace_dict.items():
                altered_line = altered_line.replace(key, value)
            altered_lines.append(altered_line)

    with open(new_file_name, "w") as f:
        f.writelines("\n".join(altered_lines))


## Create a new directory with new name and copy all files from template
def create_new_dir(dir_path, new_dir_name, replace_dict):
    # print(new_dir_name)
    os.makedirs(new_dir_name, exist_ok=True)
    # make new_file_name old file name replacing "template" with "test"
    files = []
    folders = []
    for dirpath, dirnames, filenames in walk(dir_path):
        files.extend(filenames)
        folders.extend(dirnames)
        break

    # print(files)
    # print(folders)
    for file in files:
        new_file_name = file
        # print(new_file_name)
        for key, value in replace_dict.items():
            new_file_name = new_file_name.replace(key, value)

        create_new_file(
            file_path=dir_path + "/" + file,
            new_file_name=new_dir_name + "/" + new_file_name,
            replace_dict=replace_dict,
        )
        # print(new_file_name)

    for dir in folders:
        new_subdir_name = dir
        for key, value in replace_dict.items():
            new_subdir_name = new_subdir_name.replace(key, value)
        create_new_dir(
            dir_path=dir_path + "/" + dir,
            new_dir_name=new_dir_name + "/" + new_subdir_name,
            replace_dict=replace_dict,
        )
        # print(dir)


## take a template directory and create a new directory with new name and new files
def generate_delegate(template_dir_path, new_dir_name, config, layers):
    layer_sup = ""
    for idx, layer in enumerate(layers):
        if layer not in supported_layers.keys():
            print(f"Layer {layer} is not supported.")
            sys.exit(1)
        if idx == len(layers) - 1:
            layer_sup += f"{supported_layers.get(layer)}"
        else:
            layer_sup += f"{supported_layers.get(layer)},"

    replace_dict = {
        "Tempdel": config["delegate_name"].capitalize(),
        "tempdel": config["delegate_name"].lower(),
        "TEMPDEL": config["delegate_name"].upper(),
        "Acc_name": config["acc_name"].capitalize(),
        "acc_name": config["acc_name"].lower(),
        "ACC_NAME": config["acc_name"].upper(),
        "Hw_submodule": config["hw_submodule"].capitalize(),
        "hw_submodule": config["hw_submodule"].lower(),
        "HW_SUBMODULE": config["hw_submodule"].upper(),
        "Driver_name": config["driver_name"].capitalize(),
        "driver_name": config["driver_name"].lower(),
        "DRIVER_NAME": config["driver_name"].upper(),
        "secda_delegates_path": config["secda_delegates_path"],
        "secda_apps_path": config["secda_apps_path"],
        "layer_sup": layer_sup,
    }
    create_new_dir(
        dir_path=template_dir_path,
        new_dir_name=new_dir_name,
        replace_dict=replace_dict,
    )
    print("Delegate created at: " + new_dir_name)
    print("We recommend you to move this folder to /src/secda_delegates folder")


## load config from json file
def load_config(template_config):
    try:
        with open(f"configs/{template_config}.json") as f:
            config = json.load(f)
        return config
    except:
        print("Config not found")
        sys.exit(1)


def main():
    parser = argparse.ArgumentParser(description="Generate a delegate from template.")
    parser.add_argument(
        "config_file", type=str, help="Name of the config file (without .json)"
    )
    args = parser.parse_args()

    # Prepare config dictionary
    config = load_config(args.config_file)
    delegate_name = config["delegate_name"]
    layers = config["supported_layers"]
    template = os.path.abspath("templates/tempdel_delegate")
    output_dir = os.path.abspath(f"generated/{delegate_name}_delegate")
    generate_delegate(template, output_dir, config, layers)


if __name__ == "__main__":
    main()
