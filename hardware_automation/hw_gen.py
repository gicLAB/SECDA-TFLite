
import sys
sys.dont_write_bytecode = True

import os
from string import Template
import json


vivado_path = "/home/jude/Xilinx_2019.2/Vivado/2019.2/bin/"
board_user = "jude"
board_hostname = "jharis.ddns.net"
board_port = "2202"
run_template = "run_template.sh"
out_dir = "generated"
script_dir = "hlx_scripts/"
config_dir = "configs/"
acc_link_dir = "acc_srcs/"

# open c++ header file and parse all the #define and their values into a dictionary
def parse_defines(file_name="test.h"):
    with open(file_name) as f:
        defines = {}
        for line in f:
            if line.startswith("#define"):
                tokens = line.split()
                if "BITS" in tokens[1]:
                    continue
                tokens[1] = tokens[1].replace("XVM_INT8_V3_0_SLV0_ADDR_", "")
                tokens[1] = tokens[1].replace("_DATA", "")
                defines[tokens[1]] = int(tokens[2], 16)
        # print(defines)
        print(f"rm = {defines}")


# print bitmap for an integer , put space between every 8 bits
def print_bitmap(num):
    s = bin(num)[2:]
    s = s.zfill(32)
    for i in range(0, len(s), 8):
        print(s[i : i + 8], end=" ")
    print()


def load_config(config_file):
    with open(config_file) as f:
        config = json.load(f)
    return config


class mt(Template):
    delimiter = "&"
    idpattern = r"[a-z][_a-z0-9]*"

class hardware_exp:
    def __init__(
        self,
        config,
    ):
        self.acc_name = config["acc_name"]
        self.acc_version = config["acc_version"]
        self.acc_sub_version = config["acc_sub_version"]
        self.acc_test = config["acc_test"]
        self.acc_test_id = ("_" + self.acc_test_id) if self.acc_test else ""
        self.acc_test_desc = config["acc_test_desc"]
        self.acc_link_folder = os.path.abspath(acc_link_dir + config["acc_link_folder"])
        self.acc_part = "xc7z020clg400-1"
        self.hlx_tcl_script = config["hlx_tcl_script"]
        self.top = config["top"]
        self.pynq_dir = config["pynq_dir"]
        self.board_script = config["board_script"]
        self.bitstream = config["acc_name"]+ "_"+  str(config["acc_version"])+ "_" + str(config["acc_sub_version"])
        
        self.vp = vivado_path
        self.acc_tag = (
            str(self.acc_name)
            + "_"
            + str(self.acc_version)
            + "_"
            + str(self.acc_sub_version)
            + str(self.acc_test_id)
        )

    def generate_hls_tcl(self, output_dir):
        s = ""
        s += "open_project -reset " + self.acc_tag + "\n"
        s += "set_top " + self.top + "\n"
        for file in os.listdir(self.acc_link_folder):
            if file.endswith(".cc") or file.endswith(".h"):
                s += "add_files " + "src/" + file + ' -cflags "-D__SYNTHESIS__"\n'
        s += 'open_solution "' + self.acc_tag + '"\n'
        s += "set_part " + self.acc_part + "\n"
        s += "create_clock -period " + "5" + " -name default\n"
        s += "config_export -format ip_catalog -rtl verilog -taxonomy /s -vendor xilinx\n"
        s += "csynth_design\n"
        s += "export_design -format ip_catalog\n"
        s += "exit\n"
        with open(f"{output_dir}/hls_script.tcl", "w") as f:
            f.write(s)

    def generate_hlx_tcl(self, output_dir):
        with open((script_dir + self.hlx_tcl_script)) as f:
            tcl_script = str(mt(f.read()).substitute({"top": self.top}))
        with open(f"{output_dir}/hlx_script.tcl", "w") as f:
            f.write(tcl_script)

    def create_run_script(self, output_path):
        output_dir = output_path + "/" + self.acc_tag
        run_dict = {
            "acc_link_folder": self.acc_link_folder,
            "acc_tag": self.acc_tag,
            "vp": self.vp,
            "bitstream": self.bitstream,
            "pynq_dir": self.pynq_dir,
            "board_user": board_user,
            "board_hostname": board_hostname,
            "board_port": board_port,
            "board_script": self.board_script,
        }
        with open(run_template) as f:
            run_script = str(mt(f.read()).substitute(run_dict))

        with open(f"{output_dir}/run.sh", "w") as f:
            f.write(run_script)
        os.system(f"chmod +x {output_dir}/run.sh")

    def create_project(self, output_path):
        output_dir = output_path + "/" + self.acc_tag
        output_acc_src_dir = output_dir + "/src"
        os.system(f"mkdir -p {output_dir}")
        os.system(f"mkdir -p {output_acc_src_dir}")
        self.generate_hls_tcl(output_dir)
        self.generate_hlx_tcl(output_dir)
        self.create_run_script(output_path)


def main():
    args = sys.argv[1:]
    if len(args) != 1:
        print("Usage: hw_gen.py <config_file>")
        sys.exit(1)

    config_file = args[0]
    if config_file.endswith(".json") == False:
        config_file += ".json"

    config_file = config_dir + config_file
    config = load_config(config_file)
  
    # create acc_link_folder if it does not exist
    if not os.path.exists(acc_link_dir + config["acc_link_folder"]):
        os.makedirs(acc_link_dir + config["acc_link_folder"])

    # if output directory does not exist create it
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    # then force symlink all the files in the acc_src to the acc_link_folder
    for file in os.listdir(config["acc_src"]):
        if file.endswith(".cc") or file.endswith(".h"):
            source = os.path.abspath(config["acc_src"] + "/" + file)
            target = os.path.abspath(acc_link_dir + config["acc_link_folder"] + "/")
            os.system(f"ln -sf {source} {target}")
    
    acc_proj = hardware_exp(config)
    acc_proj.create_project(out_dir)
    # print out cli for running the script
    # absolute path to the project
    acc_proj_path = os.path.abspath(out_dir + "/" + acc_proj.acc_tag)
    print(f"The project has been created in {acc_proj_path}")
    print(f"To run the project HLS and HLX, run the following commands:")
    print(f"cd {acc_proj_path}")
    print(f"{acc_proj_path}/run.sh")


if __name__ == '__main__':
  abspath = os.path.abspath(__file__)
  dname = os.path.dirname(abspath)
  os.chdir(dname)
  main()