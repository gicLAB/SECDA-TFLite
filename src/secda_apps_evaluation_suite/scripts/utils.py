import sys
sys.dont_write_bytecode = True
import os
import json
from string import Template


def declare_array(f, name, list):
    f.write("declare -a {}_array=(\n".format(name))
    for i in list:
        f.write('  "{}" \n'.format(i))
    f.write(")\n")


class mt(Template):
    delimiter = "£"
    idpattern = r"[a-z][_a-z0-9]*"


def load_config(config_file):
    if config_file.endswith(".json") == False:
        config_file += ".json"
    with open(config_file) as f:
        config = json.load(f)
    return config


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

def find_the_boards(hwc_path, hw):
    """
    hw is the list of hardwaeres in the aec_path
    It will return the board list
        - Spported borads ["Z1", "Z2", "KRIA"]
    """
    supported_boards = ["Z1", "Z2", "KRIA"]
    board_list = []

    for h in hw:
        hw_config_file = find_hw_config(hwc_path, h)
        if hw_config_file is None:
            continue

        hw_config = load_config(hw_config_file)
        board = hw_config.get("board")

        if board is None:
            continue

        if isinstance(board, list):
            boards_to_add = board
        else:
            boards_to_add = [board]

        for b in boards_to_add:
            if b not in supported_boards:
                raise ValueError("!!! Unsupported Boards. Supported Board Name [Z1, Z2, KRIA]!!!")
            if b not in board_list:
                board_list.append(b)
                
    return board_list