import os
from os import walk
import sys
import json

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
def generate_delegate(template_dir_path, new_dir_name, replace_dict):
    create_new_dir(
        dir_path=template_dir_path,
        new_dir_name=new_dir_name,
        replace_dict=replace_dict,
    )
    print("Delegate created at: " + new_dir_name)
    print("We recommend you to move this folder to /src/secda_delegates folder")


if len(sys.argv) != 2:
    print("Usage: python generate_delegate.py <config>")
    sys.exit(1)

args = sys.argv[1:]
config_file = args[0]


## load config from json file
def load_config(hw):
    try:
        with open(f"configs/{hw}.json") as f:
            config = json.load(f)
        return config
    except:
        print("Config not found")
        sys.exit(1)
    
config = load_config(config_file)
template = os.path.abspath("templates/temp_delegate")
output_dir = os.path.abspath(f"generated/{config['temp']}_delegate")
generate_delegate(template, output_dir, config)