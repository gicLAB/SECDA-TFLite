import sys
sys.dont_write_bytecode = True
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
